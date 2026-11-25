"""Exhaustive deterministic unit tests for the TCO financial calculation engine."""

from __future__ import annotations

import math

import pytest

from src.calculator import (
    calculate_quote_tco,
    convert_to_usd,
    rank_quotes,
    select_volume_tier,
    standardize_lead_time_to_business_days,
)
from src.schemas import AmbiguityFlag, NormalizedQuote, TCOCalculationConfig, VolumePriceTier


def _make_quote(**overrides) -> NormalizedQuote:
    """Build a valid baseline NormalizedQuote with optional field overrides."""
    defaults = {
        "quote_id": "Q-TEST-01",
        "source_file": "test_quote.csv",
        "supplier_id": "SUP-TEST-01",
        "supplier_name": "Test Precision Inc.",
        "supplier_email": "quotes@test.example",
        "source_language": "en",
        "rfq_reference": "RFQ-IND-2026-088",
        "part_number": "PART-HMB-7075-088",
        "part_description": "CNC-Machined 7075-T6 Hydraulic Manifold Block",
        "currency": "USD",
        "incoterm": "DDP",
        "incoterm_location": "Querétaro Plant",
        "lead_time_value": 15.0,
        "lead_time_max_value": None,
        "lead_time_unit": "business_days",
        "price_tiers": [
            VolumePriceTier(min_qty=1, max_qty=99, unit_price=180.0),
            VolumePriceTier(min_qty=100, max_qty=499, unit_price=150.0),
            VolumePriceTier(min_qty=500, max_qty=None, unit_price=135.0),
        ],
        "tooling_fee": 0.0,
        "shipping_fee": 0.0,
        "payment_terms": "Net 30",
        "certifications": ["ISO 9001:2015"],
        "quote_valid_until": "2026-12-31",
        "ambiguity_flags": [],
        "extraction_notes": [],
        "master_data_hygiene_actions": [],
    }
    defaults.update(overrides)
    return NormalizedQuote(**defaults)


class TestCurrencyConversion:
    def test_known_currencies(self) -> None:
        rates = TCOCalculationConfig().fx_rates_to_usd
        usd, w1 = convert_to_usd(100.0, "USD", rates)
        eur, w2 = convert_to_usd(100.0, "EUR", rates)
        mxn, w3 = convert_to_usd(1000.0, "MXN", rates)
        assert usd == pytest.approx(100.0)
        assert eur == pytest.approx(108.5)
        assert mxn == pytest.approx(58.0)
        assert w1 == [] and w2 == [] and w3 == []

    def test_lowercase_and_whitespace_currency(self) -> None:
        rates = TCOCalculationConfig().fx_rates_to_usd
        eur, warnings = convert_to_usd(100.0, " eur ", rates)
        assert eur == pytest.approx(108.5)
        assert warnings == []

    def test_none_negative_nonfinite_and_unknown_currency_safe(self) -> None:
        rates = TCOCalculationConfig().fx_rates_to_usd
        for bad_amount, bad_ccy in [
            (None, "USD"),
            (100.0, "JPY"),
            (100.0, None),
            (100.0, "   "),
            (-50.0, "USD"),
            (float("nan"), "USD"),
            (float("inf"), "EUR"),
            (True, "USD"),
        ]:
            val, warnings = convert_to_usd(bad_amount, bad_ccy, rates)  # type: ignore[arg-type]
            assert val is None
            assert len(warnings) >= 1


class TestLeadTimeStandardization:
    def test_units_conversion(self) -> None:
        bd, warnings, risk = standardize_lead_time_to_business_days(12.0, None, "business_days")
        assert bd == pytest.approx(12.0)
        assert warnings == []
        assert risk == pytest.approx(0.0)

        bd, warnings, risk = standardize_lead_time_to_business_days(28.0, None, "calendar_days")
        assert bd == pytest.approx(20.0)
        assert risk == pytest.approx(0.0)

        bd, warnings, risk = standardize_lead_time_to_business_days(3.0, None, "weeks")
        assert bd == pytest.approx(15.0)
        assert risk == pytest.approx(0.0)

    def test_range_midpoint_and_risk_penalty(self) -> None:
        bd, warnings, risk = standardize_lead_time_to_business_days(4.0, 6.0, "weeks")
        assert bd == pytest.approx(25.0)
        assert risk == pytest.approx(15.0)
        assert len(warnings) >= 1

    def test_inverted_range_normalized_safely(self) -> None:
        bd, warnings, risk = standardize_lead_time_to_business_days(6.0, 4.0, "weeks")
        assert bd == pytest.approx(25.0)
        assert risk == pytest.approx(15.0)
        assert any("Inverted" in w for w in warnings)

    def test_missing_negative_or_unknown_unit_safe(self) -> None:
        for val, max_val, unit in [
            (None, None, None),
            (10.0, None, None),
            (-5.0, None, "days"),
            (-5.0, None, "weeks"),
            (10.0, None, "months"),
            (float("nan"), None, "weeks"),
        ]:
            bd, warnings, risk = standardize_lead_time_to_business_days(val, max_val, unit)
            assert bd is None
            assert risk == pytest.approx(35.0)
            assert len(warnings) >= 1


class TestVolumeTierSelection:
    def test_selects_matching_tier_and_open_ended_tier(self) -> None:
        tiers = [
            VolumePriceTier(min_qty=1, max_qty=99, unit_price=185.0),
            VolumePriceTier(min_qty=100, max_qty=499, unit_price=162.5),
            VolumePriceTier(min_qty=500, max_qty=None, unit_price=148.0),
        ]
        tier, label, warnings = select_volume_tier(tiers, 250)
        assert tier is not None
        assert tier.unit_price == pytest.approx(162.5)
        assert label == "100-499"
        assert warnings == []

        tier_hi, label_hi, _ = select_volume_tier(tiers, 600)
        assert tier_hi is not None
        assert tier_hi.unit_price == pytest.approx(148.0)
        assert label_hi == "500+"

    def test_overlapping_tiers_selects_lowest_price_with_warning(self) -> None:
        tiers = [
            VolumePriceTier(min_qty=100, max_qty=499, unit_price=165.0),
            VolumePriceTier(min_qty=200, max_qty=499, unit_price=155.0),
        ]
        tier, _, warnings = select_volume_tier(tiers, 250)
        assert tier is not None
        assert tier.unit_price == pytest.approx(155.0)
        assert len(warnings) >= 1

    def test_empty_tiers_bad_qty_and_malformed_tiers_safe(self) -> None:
        assert select_volume_tier([], 250)[0] is None
        assert select_volume_tier(None, 250)[0] is None
        tiers = [VolumePriceTier(min_qty=1, max_qty=99, unit_price=185.0)]
        assert select_volume_tier(tiers, 0)[0] is None
        assert select_volume_tier(tiers, -10)[0] is None
        assert select_volume_tier(tiers, True)[0] is None  # type: ignore[arg-type]

        bad_tiers = [
            VolumePriceTier(min_qty=0, max_qty=50, unit_price=100.0),
            VolumePriceTier(min_qty=100, max_qty=50, unit_price=100.0),
            VolumePriceTier(min_qty=1, max_qty=500, unit_price=-10.0),
        ]
        chosen, _, warnings = select_volume_tier(bad_tiers, 250)
        assert chosen is None
        assert len(warnings) >= 3


class TestCalculateQuoteTCOAndRanking:
    def test_ddp_zero_delay_exact_tco(self) -> None:
        quote = _make_quote(
            incoterm="DDP",
            lead_time_value=12.0,
            lead_time_unit="business_days",
            tooling_fee=450.0,
            shipping_fee=0.0,
        )
        res = calculate_quote_tco(quote, TCOCalculationConfig())
        assert res.status == "COMPLETE"
        assert res.unit_price_usd == pytest.approx(150.0)
        assert res.merchandise_subtotal_usd == pytest.approx(37500.0)
        assert res.tooling_fee_usd == pytest.approx(450.0)
        assert res.shipping_and_incoterm_usd == pytest.approx(0.0)
        assert res.delay_days == pytest.approx(0.0)
        assert res.delay_penalty_usd == pytest.approx(0.0)
        assert res.total_cost_of_ownership_usd == pytest.approx(37950.0)

    def test_fob_no_explicit_shipping_eur_uplift(self) -> None:
        quote = _make_quote(
            supplier_id="SUP-LYON-FR",
            source_language="fr",
            currency="EUR",
            incoterm="FOB",
            lead_time_value=3.0,
            lead_time_unit="weeks",
            price_tiers=[VolumePriceTier(min_qty=100, max_qty=499, unit_price=134.5)],
            tooling_fee=350.0,
            shipping_fee=None,
        )
        res = calculate_quote_tco(quote, TCOCalculationConfig())
        assert res.status == "COMPLETE"
        assert res.unit_price_usd == pytest.approx(145.9325, abs=1e-4)
        assert res.merchandise_subtotal_usd == pytest.approx(36483.13, abs=0.01)
        assert res.tooling_fee_usd == pytest.approx(379.75, abs=0.01)
        assert res.shipping_and_incoterm_usd == pytest.approx(2371.40, abs=0.01)
        assert res.lead_time_business_days == pytest.approx(15.0)
        assert res.delay_penalty_usd == pytest.approx(0.0)
        assert res.total_cost_of_ownership_usd == pytest.approx(39234.28, abs=0.02)

    def test_exw_with_explicit_shipping_and_range_delay(self) -> None:
        quote = _make_quote(
            currency="EUR",
            incoterm="EXW",
            lead_time_value=4.0,
            lead_time_max_value=6.0,
            lead_time_unit="weeks",
            price_tiers=[VolumePriceTier(min_qty=100, max_qty=499, unit_price=126.0)],
            tooling_fee=500.0,
            shipping_fee=1850.0,
        )
        res = calculate_quote_tco(quote, TCOCalculationConfig())
        assert res.status == "COMPLETE"
        assert res.lead_time_business_days == pytest.approx(25.0)
        assert res.delay_days == pytest.approx(10.0)
        assert res.delay_penalty_usd == pytest.approx(3500.0)
        assert res.total_cost_of_ownership_usd == pytest.approx(41594.35, abs=0.02)

    def test_fob_with_explicit_shipping_usd(self) -> None:
        quote = _make_quote(
            currency="USD",
            incoterm="FOB",
            lead_time_value=21.0,
            lead_time_unit="calendar_days",
            price_tiers=[VolumePriceTier(min_qty=100, max_qty=499, unit_price=154.0)],
            tooling_fee=0.0,
            shipping_fee=1150.0,
        )
        res = calculate_quote_tco(quote, TCOCalculationConfig())
        assert res.status == "COMPLETE"
        assert res.lead_time_business_days == pytest.approx(15.0)
        assert res.shipping_and_incoterm_usd == pytest.approx(2112.50, abs=0.01)
        assert res.total_cost_of_ownership_usd == pytest.approx(40612.50, abs=0.01)

    def test_negative_fees_and_unknown_currency_never_crash(self) -> None:
        quote = _make_quote(
            currency="  jpy ",
            incoterm=None,
            tooling_fee=-200.0,
            shipping_fee=-100.0,
            quote_valid_until=None,
            ambiguity_flags=[
                AmbiguityFlag(
                    field_name="currency",
                    raw_value="jpy",
                    reason="Unsupported currency",
                    severity="HIGH",
                )
            ],
        )
        res = calculate_quote_tco(quote, TCOCalculationConfig())
        assert res.status == "INCOMPLETE_DATA"
        assert res.total_cost_of_ownership_usd is None
        assert res.tooling_fee_usd == pytest.approx(0.0)
        assert len(res.calculation_warnings) >= 3

    def test_null_fields_produce_incomplete_status_without_crashing(self) -> None:
        quote = _make_quote(
            incoterm=None,
            lead_time_value=None,
            lead_time_unit=None,
            tooling_fee=None,
            shipping_fee=None,
        )
        res = calculate_quote_tco(quote, TCOCalculationConfig())
        assert res.status == "INCOMPLETE_DATA"
        assert res.total_cost_of_ownership_usd is None
        assert res.delivery_risk_score > 50.0
        assert len(res.calculation_warnings) >= 2

    def test_ranking_places_lowest_tco_first_and_incomplete_last(self) -> None:
        q_high = calculate_quote_tco(
            _make_quote(quote_id="Q-HIGH", supplier_id="SUP-HIGH", tooling_fee=5000.0),
            TCOCalculationConfig(),
        )
        q_low = calculate_quote_tco(
            _make_quote(quote_id="Q-LOW", supplier_id="SUP-LOW", tooling_fee=100.0),
            TCOCalculationConfig(),
        )
        q_inc = calculate_quote_tco(
            _make_quote(
                quote_id="Q-INC",
                supplier_id="SUP-INC",
                lead_time_value=None,
                lead_time_unit=None,
            ),
            TCOCalculationConfig(),
        )
        ranked = rank_quotes([q_high, q_inc, q_low])
        assert ranked[0].supplier_id == "SUP-LOW"
        assert ranked[0].rank == 1
        assert ranked[0].composite_score is not None
        assert ranked[1].supplier_id == "SUP-HIGH"
        assert ranked[1].rank == 2
        assert ranked[2].supplier_id == "SUP-INC"
        assert ranked[2].status == "INCOMPLETE_DATA"
        assert ranked[2].rank is None
        assert ranked[2].composite_score is None
        assert rank_quotes([]) == []
