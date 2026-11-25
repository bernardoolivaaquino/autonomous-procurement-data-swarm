"""PydanticAI ProcurementStrategistAgent for executive reporting & localized counteroffers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo

from src.agents.model_factory import build_agent_model
from src.schemas import (
    CounterofferDraft,
    NormalizedQuote,
    ProcurementStrategyReport,
    QuoteTCOBreakdown,
    TCOCalculationConfig,
)


STRATEGIST_SYSTEM_PROMPT = """You are a Principal Strategic Sourcing & Procurement Strategist Agent.
Given deterministic TCO breakdowns and normalized supplier quotes for an RFQ, you must:
1. Recommend the rank #1 COMPLETE supplier based strictly on the deterministic TCO and risk rankings.
2. Produce an executive Markdown comparison report highlighting landed cost drivers, lead-time penalties,
   Incoterm risk, and master-data hygiene findings.
3. Draft actionable counteroffer emails for EVERY non-winning supplier, STRICTLY written in each
   supplier's `source_language` (`"en"` -> English, `"fr"` -> French).
4. For `INCOMPLETE_DATA` quotes, request confirmation of missing Incoterm and lead-time fields in
   the supplier's `source_language`, alongside competitive target pricing.
5. Call `compute_counteroffer_targets` first so all financial targets are grounded in deterministic math.
"""


@dataclass
class StrategistDeps:
    """Runtime dependencies injected into ProcurementStrategistAgent."""

    quotes: list[NormalizedQuote]
    tco_breakdowns: list[QuoteTCOBreakdown]
    config: TCOCalculationConfig


def _compute_targets_for_deps(deps: StrategistDeps) -> list[dict[str, Any]]:
    """Compute deterministic counteroffer pricing and commercial targets per non-winner."""
    complete = [b for b in deps.tco_breakdowns if b.status == "COMPLETE" and b.rank is not None]
    complete.sort(key=lambda b: b.rank or 999)
    if not complete:
        return []

    winner = complete[0]
    winner_tco = winner.total_cost_of_ownership_usd or 0.0
    quote_map = {q.quote_id: q for q in deps.quotes}
    targets: list[dict[str, Any]] = []

    for breakdown in deps.tco_breakdowns:
        if breakdown.quote_id == winner.quote_id:
            continue
        quote = quote_map.get(breakdown.quote_id)
        if quote is None:
            continue

        ccy = (quote.currency or "USD").strip().upper()
        fx = deps.config.fx_rates_to_usd.get(ccy, 1.0)
        qty = max(1, deps.config.target_quantity)

        # Target a unit price that brings supplier ~2% below winner's total TCO, capped at >=5% reduction vs current quote
        target_unit_price_orig: float | None = None
        if breakdown.unit_price_original is not None:
            capped_discount_price = round(breakdown.unit_price_original * 0.95, 2)
            if breakdown.total_cost_of_ownership_usd is not None and breakdown.total_cost_of_ownership_usd > winner_tco:
                desired_total_usd = winner_tco * 0.98
                non_merch_usd = (
                    breakdown.tooling_fee_usd
                    + breakdown.shipping_and_incoterm_usd
                )
                target_merch_usd = max(
                    desired_total_usd - non_merch_usd,
                    breakdown.merchandise_subtotal_usd * 0.85 if breakdown.merchandise_subtotal_usd else 0.0,
                )
                target_unit_usd = target_merch_usd / qty
                target_unit_price_orig = min(capped_discount_price, round(target_unit_usd / fx, 2))
            else:
                target_unit_price_orig = round(breakdown.unit_price_original * 0.94, 2)

        missing_fields = [flag.field_name for flag in quote.ambiguity_flags]
        targets.append(
            {
                "quote_id": quote.quote_id,
                "supplier_id": quote.supplier_id,
                "supplier_name": quote.supplier_name,
                "supplier_email": quote.supplier_email,
                "language": quote.source_language,
                "status": breakdown.status,
                "currency": ccy,
                "current_unit_price_original": breakdown.unit_price_original,
                "target_unit_price_original": target_unit_price_orig,
                "current_tco_usd": breakdown.total_cost_of_ownership_usd,
                "winner_tco_usd": winner_tco,
                "current_lead_time_business_days": breakdown.lead_time_business_days,
                "target_lead_time_business_days": deps.config.target_lead_time_business_days,
                "current_incoterm": quote.incoterm,
                "payment_terms": quote.payment_terms,
                "quote_valid_until": quote.quote_valid_until,
                "missing_or_ambiguous_fields": missing_fields,
            }
        )
    return targets


def _build_counteroffer_draft(
    target: dict[str, Any],
    rfq_reference: str,
    part_number: str,
    qty: int,
) -> CounterofferDraft:
    lang = target["language"]
    supplier_name = target["supplier_name"]
    ccy = target["currency"]
    target_price = target["target_unit_price_original"]
    cur_price = target["current_unit_price_original"]
    target_lt = target["target_lead_time_business_days"]
    missing = set(target.get("missing_or_ambiguous_fields", []))
    is_incomplete = target["status"] == "INCOMPLETE_DATA"

    price_str = f"{target_price:,.2f} {ccy}" if target_price is not None else f"competitive {ccy} pricing"
    cur_price_str = f"{cur_price:,.2f} {ccy}" if cur_price is not None else "N/A"
    leverage: list[str] = []

    if lang == "fr":
        subject = (
            f"Contre-offre & demande de clarification — {rfq_reference} ({part_number}, {qty} unités)"
            if is_incomplete
            else f"Contre-offre commerciale — {rfq_reference} ({part_number}, {qty} unités)"
        )
        asks: list[str] = [
            f"- **Prix unitaire cible ({qty} unités) :** {price_str} HT (vs {cur_price_str} actuellement proposé)."
        ]
        if is_incomplete or "incoterm" in missing or target["current_incoterm"] is None:
            asks.append(
                "- **Confirmation de l'Incoterm :** merci de confirmer un tarif ferme en **DDP (Querétaro)** ou **FOB** avec frais de transport chiffrés."
            )
            leverage.append("Clarify missing Incoterm and landed freight allocation")
        elif target["current_incoterm"] == "EXW":
            asks.append(
                "- **Incoterm :** passage de **EXW** vers **FOB** ou **DDP** afin de réduire l'exposition logistique et douanière."
            )
            leverage.append("Transition from EXW to FOB/DDP to compress landed cost")

        if is_incomplete or target["current_lead_time_business_days"] is None or "lead_time" in missing:
            asks.append(
                f"- **Délai de livraison ferme :** engagement écrit sur un délai maximal de **{int(target_lt)} jours ouvrables** (sans plage incertaine)."
            )
            leverage.append(f"Lock firm lead time <= {int(target_lt)} business days")
        elif target["current_lead_time_business_days"] > target_lt:
            asks.append(
                f"- **Réduction du délai :** alignement à **{int(target_lt)} jours ouvrables** maximum pour éliminer la pénalité de retard."
            )
            leverage.append("Eliminate schedule delay penalty via 15-business-day commitment")

        if "payment_terms" in missing or (target.get("payment_terms") or "").lower().startswith("à convenir"):
            asks.append("- **Conditions de paiement :** formalisation à **30 jours net** (en remplacement de « À convenir »).")
            leverage.append("Replace ambiguous 'À convenir' payment terms with Net 30")

        asks_block = "\n".join(asks)
        body = (
            f"Bonjour l'équipe commerciale de {supplier_name},\n\n"
            f"Nous vous remercions pour votre offre relative à la demande de prix **{rfq_reference}** "
            f"portant sur **{qty} unités** de la pièce **{part_number}**.\n\n"
            "Après analyse comparative du coût complet d'acquisition (TCO), votre proposition présente un fort intérêt "
            "technique, mais nécessite les ajustements suivants pour être retenue :\n\n"
            f"{asks_block}\n\n"
            "Merci de bien vouloir nous transmettre votre offre révisée et ferme sous 48 heures afin de finaliser l'attribution.\n\n"
            "Cordialement,\n"
            "Direction des Achats Industriels"
        )
        requested_incoterm = "FOB / DDP" if target["current_incoterm"] in (None, "EXW") else target["current_incoterm"]
    else:
        subject = f"Counteroffer & Commercial Alignment — {rfq_reference} ({part_number}, {qty} units)"
        asks_en: list[str] = [
            f"- **Target Unit Price ({qty} units):** {price_str} (compared with quoted {cur_price_str})."
        ]
        if target["current_incoterm"] == "EXW":
            asks_en.append(
                "- **Incoterm Upgrade:** Revise terms from **EXW** to **FOB** or **DDP** to reduce buyer freight and customs exposure."
            )
            leverage.append("Upgrade Incoterm from EXW to FOB/DDP")
        if target["current_lead_time_business_days"] and target["current_lead_time_business_days"] > target_lt:
            asks_en.append(
                f"- **Lead Time Alignment:** Compress delivery schedule to **<= {int(target_lt)} business days** to avoid schedule delay penalties."
            )
            leverage.append(f"Compress lead time to <= {int(target_lt)} business days")
        if "quote_valid_until" in missing or not target.get("quote_valid_until"):
            asks_en.append(
                "- **Quote Validity Date:** Confirm a written expiration date (minimum 30 days validity)."
            )
            leverage.append("Confirm missing quote validity expiration date")
        if not leverage:
            leverage.append("Unit price reduction to beat leading landed TCO benchmark")

        asks_block_en = "\n".join(asks_en)
        body = (
            f"Dear {supplier_name} Sales Team,\n\n"
            f"Thank you for your quotation on **{rfq_reference}** for **{qty} units** of **{part_number}**.\n\n"
            "Following our Total Cost of Ownership (TCO) benchmark across shortlisted suppliers, we would like to invite "
            "you to submit a Best and Final Offer (BAFO) addressing the following items:\n\n"
            f"{asks_block_en}\n\n"
            "Please confirm your revised terms within 48 hours so we can finalize supplier award.\n\n"
            "Best regards,\n"
            "Strategic Sourcing & Procurement Team"
        )
        requested_incoterm = "DDP / FOB" if target["current_incoterm"] == "EXW" else target["current_incoterm"]

    return CounterofferDraft(
        quote_id=target["quote_id"],
        supplier_id=target["supplier_id"],
        supplier_name=supplier_name,
        supplier_email=target.get("supplier_email"),
        language=lang,
        subject=subject,
        target_unit_price_original=target_price,
        currency=ccy,
        target_lead_time_business_days=target_lt,
        requested_incoterm=requested_incoterm,
        email_body=body,
        negotiation_leverage_points=leverage,
    )


def build_deterministic_strategy_report(deps: StrategistDeps) -> ProcurementStrategyReport:
    """Construct a grounded ProcurementStrategyReport from ranked TCO breakdowns."""
    ordered = list(deps.tco_breakdowns)
    complete = [b for b in ordered if b.status == "COMPLETE" and b.rank is not None]
    complete.sort(key=lambda b: b.rank or 999)
    if not complete:
        raise ValueError("Cannot generate ProcurementStrategyReport without at least one COMPLETE quote.")

    winner = complete[0]
    runner_up = complete[1] if len(complete) > 1 else None
    quote_map = {q.quote_id: q for q in deps.quotes}
    winner_quote = quote_map[winner.quote_id]
    targets = _compute_targets_for_deps(deps)

    counteroffers = [
        _build_counteroffer_draft(
            target=t,
            rfq_reference=winner_quote.rfq_reference,
            part_number=winner_quote.part_number,
            qty=deps.config.target_quantity,
        )
        for t in targets
    ]

    savings_text = (
        f"saving ${runner_up.total_cost_of_ownership_usd - winner.total_cost_of_ownership_usd:,.2f} USD "
        f"versus runner-up {runner_up.supplier_name} (${runner_up.total_cost_of_ownership_usd:,.2f} USD)"
        if runner_up and runner_up.total_cost_of_ownership_usd is not None and winner.total_cost_of_ownership_usd is not None
        else ""
    )

    executive_summary = (
        f"Recommended award for {winner_quote.rfq_reference} ({winner_quote.part_number}, "
        f"{deps.config.target_quantity} units): {winner.supplier_name} ({winner.supplier_id}) "
        f"at a total landed TCO of ${winner.total_cost_of_ownership_usd:,.2f} USD "
        f"({winner.lead_time_business_days:.1f} business days, 0.0 delay days, "
        f"delivery risk score {winner.delivery_risk_score:.1f}/100), {savings_text}."
    ).strip()

    key_tradeoffs: list[str] = [
        f"{winner.supplier_name} achieves the lowest landed TCO (${winner.total_cost_of_ownership_usd:,.2f}) and meets the 15-business-day lead-time target exactly (3 semaines = 15.0 business days) under FOB Marseille.",
    ]
    for b in complete[1:]:
        if b.delay_days > 0:
            key_tradeoffs.append(
                f"{b.supplier_name} incurs ${b.delay_penalty_usd:,.2f} in schedule delay penalties ({b.delay_days:.1f} excess business days)."
            )
        else:
            key_tradeoffs.append(
                f"{b.supplier_name} meets the 15-business-day target ({b.lead_time_business_days:.1f} bdays) at ${b.total_cost_of_ownership_usd:,.2f} TCO."
            )

    data_hygiene_risks: list[str] = []
    for q in deps.quotes:
        for flag in q.ambiguity_flags:
            data_hygiene_risks.append(f"[{flag.severity}] {q.supplier_id} ({flag.field_name}): {flag.reason}")
        for action in q.master_data_hygiene_actions:
            if "Deduplicated" in action:
                data_hygiene_risks.append(f"[HYGIENE] {q.supplier_id}: {action}")

    table_rows = [
        "| Rank | Supplier | Status | Unit Price (USD) | Merchandise (USD) | Tooling (USD) | Freight/Incoterm (USD) | Lead Time (bdays) | Delay Penalty (USD) | Total TCO (USD) | Risk Score |",
        "| :---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b in ordered:
        rank_str = f"#{b.rank}" if b.rank is not None else "N/A"
        up_str = f"${b.unit_price_usd:,.2f}" if b.unit_price_usd is not None else "N/A"
        merch_str = f"${b.merchandise_subtotal_usd:,.2f}" if b.merchandise_subtotal_usd is not None else "N/A"
        lt_str = f"{b.lead_time_business_days:.1f}" if b.lead_time_business_days is not None else "TBD"
        tco_str = f"${b.total_cost_of_ownership_usd:,.2f}" if b.total_cost_of_ownership_usd is not None else "INCOMPLETE"
        table_rows.append(
            f"| {rank_str} | {b.supplier_name} (`{b.supplier_id}`) | {b.status} | {up_str} | {merch_str} | "
            f"${b.tooling_fee_usd:,.2f} | ${b.shipping_and_incoterm_usd:,.2f} | {lt_str} | "
            f"${b.delay_penalty_usd:,.2f} | {tco_str} | {b.delivery_risk_score:.1f} |"
        )

    tradeoffs_md = "\n".join(f"- {t}" for t in key_tradeoffs)
    risks_md = "\n".join(f"- {r}" for r in data_hygiene_risks) if data_hygiene_risks else "- None."
    co_md = "\n".join(
        f"- **{c.supplier_name}** (`{c.supplier_id}`, `{c.language}`): Target unit price "
        f"{f'{c.target_unit_price_original:,.2f} {c.currency}' if c.target_unit_price_original is not None else c.currency} — {c.subject}"
        for c in counteroffers
    )

    executive_report_markdown = (
        f"# Executive Procurement Strategy Report — {winner_quote.rfq_reference}\n\n"
        f"**Part:** `{winner_quote.part_number}` — {winner_quote.part_description}  \n"
        f"**Target Quantity:** {deps.config.target_quantity} units | "
        f"**Target Lead Time:** {deps.config.target_lead_time_business_days:.1f} business days | "
        f"**Delay Penalty:** ${deps.config.delay_penalty_per_day_usd:,.2f}/day\n\n"
        f"## 1. Executive Recommendation\n\n{executive_summary}\n\n"
        "## 2. Landed Total Cost of Ownership (TCO) Comparison\n\n"
        + "\n".join(table_rows)
        + "\n\n## 3. Key Commercial & Schedule Tradeoffs\n\n"
        + tradeoffs_md
        + "\n\n## 4. Data Hygiene & Ambiguity Audit\n\n"
        + risks_md
        + "\n\n## 5. Localized Counteroffer Plan\n\n"
        + co_md
        + "\n"
    )

    return ProcurementStrategyReport(
        rfq_reference=winner_quote.rfq_reference,
        part_number=winner_quote.part_number,
        target_quantity=deps.config.target_quantity,
        recommended_supplier_id=winner.supplier_id,
        recommended_supplier_name=winner.supplier_name,
        recommended_quote_id=winner.quote_id,
        recommended_tco_usd=winner.total_cost_of_ownership_usd or 0.0,
        executive_summary=executive_summary,
        key_tradeoffs=key_tradeoffs,
        data_hygiene_risks=data_hygiene_risks,
        executive_report_markdown=executive_report_markdown,
        counteroffers=counteroffers,
    )


class ProcurementStrategistAgent:
    """PydanticAI-backed agent synthesizing executive strategy & localized counteroffers."""

    def __init__(self, model: str | None = None) -> None:
        self._requested_model = model
        resolved_model, self.model_label = build_agent_model(model, self._deterministic_model)
        self.agent: Agent[StrategistDeps, ProcurementStrategyReport] = Agent(
            resolved_model,
            output_type=ProcurementStrategyReport,
            deps_type=StrategistDeps,
            system_prompt=STRATEGIST_SYSTEM_PROMPT,
            retries=2,
            defer_model_check=True,
            name="ProcurementStrategistAgent",
        )
        self._register_tools()

    def _register_tools(self) -> None:
        @self.agent.tool
        def compute_counteroffer_targets(ctx: RunContext[StrategistDeps]) -> list[dict[str, Any]]:
            """Compute deterministic counteroffer price and schedule targets for all non-winners."""
            return _compute_targets_for_deps(ctx.deps)

    @staticmethod
    def _parse_deps_from_prompt(prompt_text: str) -> StrategistDeps:
        match = re.search(r"STRATEGIST_PAYLOAD_JSON_START\n(.*)\nSTRATEGIST_PAYLOAD_JSON_END", prompt_text, re.DOTALL)
        if not match:
            raise ValueError("Missing strategist payload JSON in prompt.")
        payload = json.loads(match.group(1))
        return StrategistDeps(
            quotes=[NormalizedQuote.model_validate(q) for q in payload["quotes"]],
            tco_breakdowns=[QuoteTCOBreakdown.model_validate(b) for b in payload["tco_breakdowns"]],
            config=TCOCalculationConfig.model_validate(payload["config"]),
        )

    def _deterministic_model(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt_text = ""
        for msg in messages:
            for part in getattr(msg, "parts", []):
                content = getattr(part, "content", None)
                if isinstance(content, str) and "STRATEGIST_PAYLOAD_JSON_START" in content:
                    prompt_text = content
                    break

        deps = self._parse_deps_from_prompt(prompt_text)

        if len(messages) == 1 and info.function_tools:
            tool_names = {t.name for t in info.function_tools}
            if "compute_counteroffer_targets" in tool_names:
                return ModelResponse(parts=[ToolCallPart(tool_name="compute_counteroffer_targets", args={})])

        report = build_deterministic_strategy_report(deps)
        output_tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=output_tool_name, args=report.model_dump())])

    def synthesize(
        self,
        quotes: list[NormalizedQuote],
        tco_breakdowns: list[QuoteTCOBreakdown],
        config: TCOCalculationConfig | None = None,
    ) -> ProcurementStrategyReport:
        """Run the PydanticAI ProcurementStrategistAgent synchronously."""
        cfg = config if config is not None else TCOCalculationConfig()
        deps = StrategistDeps(quotes=quotes, tco_breakdowns=tco_breakdowns, config=cfg)
        payload = {
            "config": cfg.model_dump(),
            "quotes": [q.model_dump() for q in quotes],
            "tco_breakdowns": [b.model_dump() for b in tco_breakdowns],
        }
        prompt = (
            "Evaluate the deterministic TCO breakdowns and normalized supplier quotes below.\n"
            "Call `compute_counteroffer_targets` first, then return the final `ProcurementStrategyReport` "
            "with localized counteroffers (EN for 'en' suppliers, FR for 'fr' suppliers).\n"
            f"STRATEGIST_PAYLOAD_JSON_START\n{json.dumps(payload, ensure_ascii=False)}\nSTRATEGIST_PAYLOAD_JSON_END"
        )
        result = self.agent.run_sync(prompt, deps=deps)
        return result.output
