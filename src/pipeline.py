"""End-to-end orchestration pipeline for the Autonomous Procurement Data Swarm."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from src.agents.ingestion_agent import QuoteIngestionAgent
from src.agents.strategist_agent import ProcurementStrategistAgent
from src.calculator import calculate_quote_tco, rank_quotes
from src.schemas import (
    NormalizedQuote,
    PipelineMetrics,
    PipelineResult,
    ProcurementStrategyReport,
    QuoteTCOBreakdown,
    TCOCalculationConfig,
)


SUPPORTED_QUOTE_EXTENSIONS = {".csv", ".md", ".txt"}


def discover_quote_files(quotes_dir: str | Path) -> list[Path]:
    """Return sorted list of supported quote files inside ``quotes_dir``."""
    directory = Path(quotes_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Quotes directory not found: {directory}")
    files = [
        p
        for p in sorted(directory.iterdir(), key=lambda item: item.name.lower())
        if p.is_file() and p.suffix.lower() in SUPPORTED_QUOTE_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No .csv, .md, or .txt quote files found in {directory}")
    return files


def write_master_data_csv(quotes: list[NormalizedQuote], path: Path) -> None:
    """Export canonical normalized supplier quotes to a flat master-data CSV."""
    fieldnames = [
        "quote_id",
        "source_file",
        "supplier_id",
        "supplier_name",
        "supplier_email",
        "source_language",
        "rfq_reference",
        "part_number",
        "currency",
        "incoterm",
        "incoterm_location",
        "lead_time_value",
        "lead_time_max_value",
        "lead_time_unit",
        "price_tiers_json",
        "tooling_fee",
        "shipping_fee",
        "payment_terms",
        "certifications",
        "quote_valid_until",
        "ambiguity_flag_count",
        "hygiene_action_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for q in quotes:
            writer.writerow(
                {
                    "quote_id": q.quote_id,
                    "source_file": q.source_file,
                    "supplier_id": q.supplier_id,
                    "supplier_name": q.supplier_name,
                    "supplier_email": q.supplier_email or "",
                    "source_language": q.source_language,
                    "rfq_reference": q.rfq_reference,
                    "part_number": q.part_number,
                    "currency": q.currency,
                    "incoterm": q.incoterm or "",
                    "incoterm_location": q.incoterm_location or "",
                    "lead_time_value": q.lead_time_value if q.lead_time_value is not None else "",
                    "lead_time_max_value": q.lead_time_max_value if q.lead_time_max_value is not None else "",
                    "lead_time_unit": q.lead_time_unit or "",
                    "price_tiers_json": json.dumps([t.model_dump() for t in q.price_tiers]),
                    "tooling_fee": q.tooling_fee if q.tooling_fee is not None else "",
                    "shipping_fee": q.shipping_fee if q.shipping_fee is not None else "",
                    "payment_terms": q.payment_terms or "",
                    "certifications": "; ".join(q.certifications),
                    "quote_valid_until": q.quote_valid_until or "",
                    "ambiguity_flag_count": len(q.ambiguity_flags),
                    "hygiene_action_count": len(q.master_data_hygiene_actions),
                }
            )


def write_tco_csv(breakdowns: list[QuoteTCOBreakdown], path: Path) -> None:
    """Export ranked deterministic TCO breakdowns to CSV."""
    fieldnames = [
        "rank",
        "quote_id",
        "supplier_id",
        "supplier_name",
        "source_language",
        "status",
        "applicable_tier_range",
        "unit_price_original",
        "original_currency",
        "unit_price_usd",
        "merchandise_subtotal_usd",
        "tooling_fee_usd",
        "shipping_and_incoterm_usd",
        "lead_time_business_days",
        "delay_days",
        "delay_penalty_usd",
        "total_cost_of_ownership_usd",
        "delivery_risk_score",
        "composite_score",
        "warnings_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for b in breakdowns:
            writer.writerow(
                {
                    "rank": b.rank if b.rank is not None else "",
                    "quote_id": b.quote_id,
                    "supplier_id": b.supplier_id,
                    "supplier_name": b.supplier_name,
                    "source_language": b.source_language,
                    "status": b.status,
                    "applicable_tier_range": b.applicable_tier_range or "",
                    "unit_price_original": b.unit_price_original if b.unit_price_original is not None else "",
                    "original_currency": b.original_currency or "",
                    "unit_price_usd": b.unit_price_usd if b.unit_price_usd is not None else "",
                    "merchandise_subtotal_usd": b.merchandise_subtotal_usd if b.merchandise_subtotal_usd is not None else "",
                    "tooling_fee_usd": b.tooling_fee_usd,
                    "shipping_and_incoterm_usd": b.shipping_and_incoterm_usd,
                    "lead_time_business_days": b.lead_time_business_days if b.lead_time_business_days is not None else "",
                    "delay_days": b.delay_days,
                    "delay_penalty_usd": b.delay_penalty_usd,
                    "total_cost_of_ownership_usd": b.total_cost_of_ownership_usd if b.total_cost_of_ownership_usd is not None else "",
                    "delivery_risk_score": b.delivery_risk_score,
                    "composite_score": b.composite_score if b.composite_score is not None else "",
                    "warnings_count": len(b.calculation_warnings),
                }
            )


class ProcurementSwarmPipeline:
    """Coordinates quote ingestion, deterministic TCO modeling, strategy synthesis, and artifact export."""

    def __init__(self, model: str | None = None) -> None:
        self.ingestion_agent = QuoteIngestionAgent(model=model)
        self.strategist_agent = ProcurementStrategistAgent(model=model)

    def ingest_quotes(self, quotes_dir: str | Path) -> list[NormalizedQuote]:
        """Ingest and normalize all dirty quote files in ``quotes_dir``."""
        files = discover_quote_files(quotes_dir)
        return [self.ingestion_agent.extract(file_path) for file_path in files]

    @staticmethod
    def evaluate_tco(
        quotes: list[NormalizedQuote],
        config: TCOCalculationConfig | None = None,
    ) -> list[QuoteTCOBreakdown]:
        """Run deterministic TCO engine and rank all normalized quotes."""
        cfg = config if config is not None else TCOCalculationConfig()
        breakdowns = [calculate_quote_tco(q, cfg) for q in quotes]
        return rank_quotes(breakdowns)

    def run(
        self,
        quotes_dir: str | Path = "data/quotes",
        output_dir: str | Path = "output",
        config: TCOCalculationConfig | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline and export all JSON, CSV, and Markdown deliverables."""
        start = time.perf_counter()
        cfg = config if config is not None else TCOCalculationConfig()
        out_path = Path(output_dir)
        counteroffers_dir = out_path / "counteroffers"
        out_path.mkdir(parents=True, exist_ok=True)
        counteroffers_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Ingestion Agent
        normalized_quotes = self.ingest_quotes(quotes_dir)
        (out_path / "normalized_quotes.json").write_text(
            json.dumps([q.model_dump() for q in normalized_quotes], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_master_data_csv(normalized_quotes, out_path / "master_data_cleaned.csv")

        # Stage 2: Pure Deterministic TCO Engine
        tco_breakdowns = self.evaluate_tco(normalized_quotes, cfg)
        (out_path / "tco_comparison.json").write_text(
            json.dumps([b.model_dump() for b in tco_breakdowns], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_tco_csv(tco_breakdowns, out_path / "tco_comparison.csv")

        # Stage 3: Procurement Strategist Agent
        strategy_report: ProcurementStrategyReport = self.strategist_agent.synthesize(
            quotes=normalized_quotes,
            tco_breakdowns=tco_breakdowns,
            config=cfg,
        )
        (out_path / "executive_report.md").write_text(
            strategy_report.executive_report_markdown,
            encoding="utf-8",
        )

        # Clean existing *.md in counteroffers_dir before writing current batch
        for old_md in counteroffers_dir.glob("*.md"):
            old_md.unlink()

        for draft in strategy_report.counteroffers:
            filename = f"{draft.quote_id.lower()}_{draft.supplier_id.lower()}_{draft.language}.md"
            md_content = (
                f"# Counteroffer Draft — {draft.supplier_name} (`{draft.supplier_id}`)\n\n"
                f"- **Quote ID:** {draft.quote_id}\n"
                f"- **To:** {draft.supplier_email or 'N/A'}\n"
                f"- **Language:** `{draft.language}`\n"
                f"- **Subject:** {draft.subject}\n"
                f"- **Target Unit Price:** "
                f"{f'{draft.target_unit_price_original:,.2f} {draft.currency}' if draft.target_unit_price_original is not None else draft.currency}\n"
                f"- **Target Lead Time:** "
                f"{f'{draft.target_lead_time_business_days:.1f} business days' if draft.target_lead_time_business_days is not None else 'TBD'}\n"
                f"- **Requested Incoterm:** {draft.requested_incoterm or 'N/A'}\n\n"
                "---\n\n"
                f"{draft.email_body}\n"
            )
            (counteroffers_dir / filename).write_text(md_content, encoding="utf-8")

        elapsed = round(time.perf_counter() - start, 4)
        complete_tcos = [
            b.total_cost_of_ownership_usd
            for b in tco_breakdowns
            if b.status == "COMPLETE" and b.total_cost_of_ownership_usd is not None
        ]
        second_best = complete_tcos[1] if len(complete_tcos) > 1 else None
        savings_vs_next = (
            round(second_best - strategy_report.recommended_tco_usd, 2)
            if second_best is not None
            else None
        )
        spread = (
            round(max(complete_tcos) - min(complete_tcos), 2)
            if len(complete_tcos) >= 2
            else None
        )

        # Conservative manual baseline: ~45 min per multilingual quote (extraction + FX/Incoterm TCO + localized emails)
        manual_hours_saved = round(len(normalized_quotes) * 0.75, 2)

        metrics = PipelineMetrics(
            rfq_reference=strategy_report.rfq_reference,
            model_used=self.ingestion_agent.model_label,
            quotes_processed=len(normalized_quotes),
            complete_quotes=sum(1 for b in tco_breakdowns if b.status == "COMPLETE"),
            incomplete_quotes=sum(1 for b in tco_breakdowns if b.status == "INCOMPLETE_DATA"),
            ambiguity_flags_total=sum(len(q.ambiguity_flags) for q in normalized_quotes),
            hygiene_actions_total=sum(len(q.master_data_hygiene_actions) for q in normalized_quotes),
            counteroffers_generated=len(strategy_report.counteroffers),
            recommended_supplier_id=strategy_report.recommended_supplier_id,
            recommended_tco_usd=strategy_report.recommended_tco_usd,
            second_best_tco_usd=second_best,
            tco_savings_vs_next_best_usd=savings_vs_next,
            tco_spread_usd=spread,
            execution_time_seconds=elapsed,
            estimated_manual_hours_saved=manual_hours_saved,
        )
        (out_path / "pipeline_metrics.json").write_text(
            json.dumps(metrics.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return PipelineResult(
            config=cfg,
            normalized_quotes=normalized_quotes,
            tco_breakdowns=tco_breakdowns,
            strategy_report=strategy_report,
            metrics=metrics,
        )
