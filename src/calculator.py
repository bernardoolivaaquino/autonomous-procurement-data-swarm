"""Pure deterministic Python financial engine for Total Cost of Ownership (TCO).

Zero LLM calls are executed in this module. All FX conversions, lead-time
standardization, Incoterm landed-cost uplifts, delay penalties, risk scoring,
and ranking are fully deterministic and defensively guarded against null or
malformed inputs.
"""

from __future__ import annotations

import math
from typing import Iterable

from src.schemas import (
    NormalizedQuote,
    QuoteTCOBreakdown,
    TCOCalculationConfig,
    VolumePriceTier,
)


def _lookup_rate(mapping: dict[str, float], key: str | None) -> float | None:
    """Case-insensitive trimmed lookup into a rate/uplift dictionary."""
    if key is None:
        return None
    normalized = key.strip().upper()
    if not normalized:
        return None
    if normalized in mapping:
        return mapping[normalized]
    for k, v in mapping.items():
        if k.strip().upper() == normalized:
            return v
    return None


def convert_to_usd(
    amount: float | None,
    currency: str | None,
    fx_rates_to_usd: dict[str, float] | None = None,
) -> tuple[float | None, list[str]]:
    """Convert an amount in a source currency to USD without raising on bad input.

    Returns ``(usd_amount, warnings)``. When conversion cannot be performed,
    ``usd_amount`` is ``None`` and ``warnings`` explains why.
    """
    warnings: list[str] = []
    if amount is None:
        warnings.append("Missing monetary amount; cannot convert to USD.")
        return None, warnings
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount):
        warnings.append(f"Non-finite or invalid monetary amount '{amount}'; cannot convert to USD.")
        return None, warnings
    if amount < 0:
        warnings.append(f"Negative monetary amount ({amount}) is invalid; cannot convert to USD.")
        return None, warnings

    if currency is None or not str(currency).strip():
        warnings.append("Missing currency code; cannot convert to USD.")
        return None, warnings

    rates = fx_rates_to_usd if fx_rates_to_usd is not None else TCOCalculationConfig().fx_rates_to_usd
    clean_currency = str(currency).strip().upper()
    rate = _lookup_rate(rates, clean_currency)
    if rate is None or not math.isfinite(rate) or rate <= 0:
        warnings.append(f"Unknown or invalid currency '{currency}'; no FX rate available.")
        return None, warnings

    return amount * rate, warnings


def standardize_lead_time_to_business_days(
    lead_time_value: float | None,
    lead_time_max_value: float | None,
    lead_time_unit: str | None,
) -> tuple[float | None, list[str], float]:
    """Standardize a quote's lead time (and optional range) into business days.

    Returns ``(business_days, warnings, risk_penalty_points)``.
    """
    warnings: list[str] = []
    if lead_time_value is None or lead_time_unit is None or not str(lead_time_unit).strip():
        warnings.append("Missing lead time value or unit; lead time cannot be standardized.")
        return None, warnings, 35.0

    if not math.isfinite(lead_time_value) or lead_time_value < 0:
        warnings.append(f"Invalid negative/non-finite lead time value ({lead_time_value}).")
        return None, warnings, 35.0

    unit_norm = str(lead_time_unit).strip().lower()
    factor_map = {
        "business_days": 1.0,
        "calendar_days": 5.0 / 7.0,
        "weeks": 5.0,
    }
    factor = factor_map.get(unit_norm)
    if factor is None:
        warnings.append(f"Unsupported lead time unit '{lead_time_unit}'.")
        return None, warnings, 35.0

    risk_penalty = 0.0
    effective_value = float(lead_time_value)

    if lead_time_max_value is not None:
        if not math.isfinite(lead_time_max_value) or lead_time_max_value < 0:
            warnings.append(
                f"Invalid upper lead time bound ({lead_time_max_value}); using minimum bound {lead_time_value}."
            )
            risk_penalty += 15.0
        elif lead_time_max_value != lead_time_value:
            low, high = (
                (lead_time_value, lead_time_max_value)
                if lead_time_max_value >= lead_time_value
                else (lead_time_max_value, lead_time_value)
            )
            if lead_time_max_value < lead_time_value:
                warnings.append(
                    f"Inverted lead time range ({lead_time_value}-{lead_time_max_value} {unit_norm}); "
                    "normalized bounds before computing midpoint."
                )
            effective_value = (low + high) / 2.0
            risk_penalty += 15.0
            warnings.append(
                f"Lead time specified as range ({low:g}-{high:g} {unit_norm}); "
                f"using midpoint ({effective_value:g} {unit_norm}) with +15.0 range-uncertainty risk penalty."
            )

    business_days = round(effective_value * factor, 4)
    return business_days, warnings, risk_penalty


def select_volume_tier(
    price_tiers: Iterable[VolumePriceTier] | None,
    quantity: int,
) -> tuple[VolumePriceTier | None, str | None, list[str]]:
    """Select the applicable volume price tier for ``quantity``.

    Returns ``(tier, tier_range_label, warnings)``. Never raises on empty lists,
    non-positive quantities, or malformed tier bounds.
    """
    warnings: list[str] = []
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) or quantity <= 0:
        warnings.append(f"Target quantity ({quantity}) must be a positive integer.")
        return None, None, warnings

    if not price_tiers:
        warnings.append("No volume price tiers available on quote.")
        return None, None, warnings

    valid_tiers: list[VolumePriceTier] = []
    for idx, tier in enumerate(price_tiers):
        if tier is None:
            warnings.append(f"Skipped null price tier at index {idx}.")
            continue
        if tier.min_qty <= 0:
            warnings.append(f"Ignored price tier with non-positive min_qty ({tier.min_qty}).")
            continue
        if tier.max_qty is not None and tier.max_qty < tier.min_qty:
            warnings.append(
                f"Ignored malformed price tier ({tier.min_qty}-{tier.max_qty}) where max_qty < min_qty."
            )
            continue
        if not math.isfinite(tier.unit_price) or tier.unit_price < 0:
            warnings.append(
                f"Ignored price tier ({tier.min_qty}-{tier.max_qty}) with invalid unit_price ({tier.unit_price})."
            )
            continue
        valid_tiers.append(tier)

    if not valid_tiers:
        warnings.append("No valid volume price tiers remained after validation.")
        return None, None, warnings

    matching = [
        t
        for t in valid_tiers
        if t.min_qty <= quantity and (t.max_qty is None or quantity <= t.max_qty)
    ]
    if not matching:
        warnings.append(f"No volume price tier covers quantity {quantity}.")
        return None, None, warnings

    if len(matching) > 1:
        warnings.append(
            f"Multiple overlapping price tiers matched quantity {quantity}; selected lowest unit price."
        )
        matching.sort(key=lambda t: (t.unit_price, -t.min_qty))

    chosen = matching[0]
    label = f"{chosen.min_qty}-{chosen.max_qty}" if chosen.max_qty is not None else f"{chosen.min_qty}+"
    return chosen, label, warnings


def _compute_shipping_and_incoterm_usd(
    quote: NormalizedQuote,
    merchandise_subtotal_usd: float | None,
    config: TCOCalculationConfig,
    audit_trail: list[str],
    warnings: list[str],
) -> tuple[float, float]:
    """Compute landed freight/customs adjustment in USD and Incoterm risk points."""
    incoterm_key = quote.incoterm.strip().upper() if quote.incoterm else "UNKNOWN"
    incoterm_risk_map = {"DDP": 0.0, "FOB": 4.0, "EXW": 8.0, "UNKNOWN": 15.0}
    incoterm_risk = incoterm_risk_map.get(incoterm_key, 15.0)

    if quote.incoterm is None:
        warnings.append(
            "Incoterm is missing/unconfirmed; applied conservative UNKNOWN landed-cost buffer and +15.0 risk points."
        )

    explicit_shipping_usd = 0.0
    has_explicit_shipping = False

    if quote.shipping_fee is not None:
        if quote.shipping_fee < 0:
            warnings.append(
                f"Negative shipping_fee ({quote.shipping_fee}) ignored; falling back to Incoterm uplift."
            )
        elif quote.shipping_fee > 0:
            converted, ship_warnings = convert_to_usd(
                quote.shipping_fee, quote.currency, config.fx_rates_to_usd
            )
            warnings.extend(ship_warnings)
            if converted is not None:
                explicit_shipping_usd = converted
                has_explicit_shipping = True

    if merchandise_subtotal_usd is None:
        return round(explicit_shipping_usd, 2), incoterm_risk

    if has_explicit_shipping:
        buffer_pct = _lookup_rate(config.incoterm_customs_buffer_pct, incoterm_key)
        if buffer_pct is None:
            buffer_pct = 0.050
        customs_buffer_usd = merchandise_subtotal_usd * buffer_pct
        total_logistics = explicit_shipping_usd + customs_buffer_usd
        audit_trail.append(
            f"Logistics ({incoterm_key}, explicit freight): "
            f"${explicit_shipping_usd:,.2f} freight + ${customs_buffer_usd:,.2f} "
            f"({buffer_pct * 100:.1f}% customs/duties buffer) = ${total_logistics:,.2f}."
        )
        return total_logistics, incoterm_risk

    uplift_pct = _lookup_rate(config.incoterm_uplift_pct, incoterm_key)
    if uplift_pct is None:
        uplift_pct = 0.120
    uplift_usd = merchandise_subtotal_usd * uplift_pct
    audit_trail.append(
        f"Logistics ({incoterm_key}, no explicit freight): "
        f"{uplift_pct * 100:.1f}% Incoterm uplift on ${merchandise_subtotal_usd:,.2f} = ${uplift_usd:,.2f}."
    )
    return uplift_usd, incoterm_risk


def calculate_quote_tco(
    quote: NormalizedQuote,
    config: TCOCalculationConfig | None = None,
) -> QuoteTCOBreakdown:
    """Compute full deterministic TCO and delivery risk score for a NormalizedQuote."""
    cfg = config if config is not None else TCOCalculationConfig()
    warnings: list[str] = []
    audit_trail: list[str] = []

    clean_currency = quote.currency.strip().upper() if quote.currency else None
    if quote.currency and quote.currency != clean_currency:
        warnings.append(f"Normalized currency '{quote.currency}' to '{clean_currency}'.")

    # 1. Select volume tier
    tier, tier_label, tier_warnings = select_volume_tier(quote.price_tiers, cfg.target_quantity)
    warnings.extend(tier_warnings)

    unit_price_original: float | None = tier.unit_price if tier is not None else None
    unit_price_usd: float | None = None
    merchandise_subtotal_usd: float | None = None

    if unit_price_original is not None:
        unit_price_usd, fx_warnings = convert_to_usd(
            unit_price_original, clean_currency, cfg.fx_rates_to_usd
        )
        warnings.extend(fx_warnings)
        if unit_price_usd is not None:
            merchandise_subtotal_usd = unit_price_usd * cfg.target_quantity
            audit_trail.append(
                f"Unit price ({tier_label} tier): {unit_price_original:,.2f} {clean_currency} "
                f"= ${unit_price_usd:,.4f} USD x {cfg.target_quantity} = ${merchandise_subtotal_usd:,.2f}."
            )

    # 2. Tooling fee
    tooling_fee_usd = 0.0
    if quote.tooling_fee is None:
        warnings.append("Tooling fee not specified; assumed $0.00.")
        audit_trail.append("Tooling fee: None -> $0.00.")
    elif quote.tooling_fee < 0:
        warnings.append(f"Negative tooling_fee ({quote.tooling_fee}) treated as $0.00.")
        audit_trail.append(f"Tooling fee: invalid ({quote.tooling_fee}) -> $0.00.")
    else:
        converted_tooling, tooling_warnings = convert_to_usd(
            quote.tooling_fee, clean_currency, cfg.fx_rates_to_usd
        )
        warnings.extend(tooling_warnings)
        if converted_tooling is not None:
            tooling_fee_usd = converted_tooling
            audit_trail.append(
                f"Tooling fee: {quote.tooling_fee:,.2f} {clean_currency} = ${tooling_fee_usd:,.2f}."
            )

    # 3. Shipping & Incoterm uplift
    shipping_and_incoterm_usd, incoterm_risk = _compute_shipping_and_incoterm_usd(
        quote=quote,
        merchandise_subtotal_usd=merchandise_subtotal_usd,
        config=cfg,
        audit_trail=audit_trail,
        warnings=warnings,
    )

    # 4. Lead time standardization & delay penalty
    lead_time_bd, lt_warnings, lt_risk = standardize_lead_time_to_business_days(
        quote.lead_time_value,
        quote.lead_time_max_value,
        quote.lead_time_unit,
    )
    warnings.extend(lt_warnings)

    delay_days = 0.0
    delay_penalty_usd = 0.0
    if lead_time_bd is not None:
        delay_days = max(0.0, round(lead_time_bd - cfg.target_lead_time_business_days, 4))
        delay_penalty_usd = delay_days * cfg.delay_penalty_per_day_usd
        audit_trail.append(
            f"Lead time: {lead_time_bd:.2f} business days vs {cfg.target_lead_time_business_days:.2f} target "
            f"-> {delay_days:.2f} delay days x ${cfg.delay_penalty_per_day_usd:,.2f}/day = ${delay_penalty_usd:,.2f}."
        )

    # 5. Validity & ambiguity risk scoring
    validity_risk = 0.0
    if not quote.quote_valid_until:
        validity_risk = 5.0
        warnings.append("Missing quote_valid_until date; added +5.0 validity risk points.")

    flag_risk = 0.0
    for flag in quote.ambiguity_flags:
        if flag.severity == "HIGH":
            flag_risk += 5.0
        elif flag.severity == "MEDIUM":
            flag_risk += 2.5
        else:
            flag_risk += 1.0

    delay_risk = min(35.0, delay_days * 3.0)
    base_risk = 10.0
    raw_risk = base_risk + incoterm_risk + lt_risk + delay_risk + validity_risk + flag_risk
    delivery_risk_score = round(max(0.0, min(100.0, raw_risk)), 2)

    # 6. Completeness gate & final TCO
    is_complete = (
        merchandise_subtotal_usd is not None
        and unit_price_usd is not None
        and lead_time_bd is not None
    )

    total_tco_usd: float | None = None
    status: str = "INCOMPLETE_DATA"
    if is_complete:
        status = "COMPLETE"
        total_tco_usd = round(
            merchandise_subtotal_usd
            + tooling_fee_usd
            + shipping_and_incoterm_usd
            + delay_penalty_usd,
            2,
        )
        audit_trail.append(f"Total TCO: ${total_tco_usd:,.2f} USD (status=COMPLETE).")
    else:
        warnings.append(
            "Quote marked INCOMPLETE_DATA because required pricing, FX, or lead-time inputs are missing."
        )
        audit_trail.append("Total TCO: N/A (status=INCOMPLETE_DATA).")

    return QuoteTCOBreakdown(
        quote_id=quote.quote_id,
        supplier_id=quote.supplier_id,
        supplier_name=quote.supplier_name,
        source_language=quote.source_language,
        status=status,  # type: ignore[arg-type]
        applicable_tier_range=tier_label,
        unit_price_original=unit_price_original,
        original_currency=clean_currency,
        unit_price_usd=round(unit_price_usd, 4) if unit_price_usd is not None else None,
        merchandise_subtotal_usd=(
            round(merchandise_subtotal_usd, 2) if merchandise_subtotal_usd is not None else None
        ),
        tooling_fee_usd=round(tooling_fee_usd, 2),
        shipping_and_incoterm_usd=round(shipping_and_incoterm_usd, 2),
        lead_time_business_days=round(lead_time_bd, 2) if lead_time_bd is not None else None,
        delay_days=round(delay_days, 2),
        delay_penalty_usd=round(delay_penalty_usd, 2),
        total_cost_of_ownership_usd=total_tco_usd,
        delivery_risk_score=delivery_risk_score,
        composite_score=None,
        rank=None,
        calculation_warnings=warnings,
        audit_trail=audit_trail,
    )


def rank_quotes(breakdowns: list[QuoteTCOBreakdown]) -> list[QuoteTCOBreakdown]:
    """Score and rank QuoteTCOBreakdown records deterministically.

    Complete quotes are ranked 1..N by lowest risk-adjusted ``composite_score``
    (with ``total_cost_of_ownership_usd`` and ``delivery_risk_score`` tie-breakers).
    ``INCOMPLETE_DATA`` quotes are placed after all complete quotes with
    ``rank=None`` and ``composite_score=None``.
    """
    if not breakdowns:
        return []

    complete: list[QuoteTCOBreakdown] = []
    incomplete: list[QuoteTCOBreakdown] = []

    for item in breakdowns:
        if item.status == "COMPLETE" and item.total_cost_of_ownership_usd is not None:
            composite = round(
                item.total_cost_of_ownership_usd * (1.0 + 0.0015 * item.delivery_risk_score),
                2,
            )
            complete.append(item.model_copy(update={"composite_score": composite}))
        else:
            incomplete.append(item.model_copy(update={"composite_score": None, "rank": None}))

    complete.sort(
        key=lambda b: (
            b.composite_score if b.composite_score is not None else float("inf"),
            b.total_cost_of_ownership_usd if b.total_cost_of_ownership_usd is not None else float("inf"),
            b.delivery_risk_score,
            b.quote_id,
        )
    )

    ranked_complete = [
        item.model_copy(update={"rank": idx}) for idx, item in enumerate(complete, start=1)
    ]
    incomplete.sort(key=lambda b: (b.delivery_risk_score, b.quote_id))
    return ranked_complete + incomplete
