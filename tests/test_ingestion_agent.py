"""Tests for the PydanticAI QuoteIngestionAgent across all 6 dirty quotes."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.ingestion_agent import (
    QuoteIngestionAgent,
    clean_numeric_string,
    clean_supplier_id,
)
from src.agents.model_factory import resolve_model_choice
from src.schemas import NormalizedQuote

QUOTES_DIR = Path("data/quotes")


class TestIngestionHelpers:
    def test_clean_numeric_string_en_fr_mxn(self) -> None:
        assert clean_numeric_string("$1,150.00") == pytest.approx(1150.0)
        assert clean_numeric_string("134,50 €") == pytest.approx(134.5)
        assert clean_numeric_string("1\u00a0850,00 €") == pytest.approx(1850.0)
        assert clean_numeric_string("MX$2,580.00") == pytest.approx(2580.0)
        assert clean_numeric_string("3 100,00 MXN") == pytest.approx(3100.0)
        assert clean_numeric_string("") is None
        assert clean_numeric_string(None) is None

    def test_clean_supplier_id(self) -> None:
        assert clean_supplier_id("  sup lyon fr ") == "SUP-LYON-FR"

    def test_model_factory_deterministic_fallback_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        label, is_det = resolve_model_choice("openai:gpt-4o-mini")
        assert is_det is True
        assert "deterministic" in label


@pytest.fixture(scope="module")
def agent() -> QuoteIngestionAgent:
    return QuoteIngestionAgent(model="deterministic")


class TestQuoteIngestionAgentAllQuotes:
    def test_quote_01_apex_deduplication_and_bom_cleanup(self, agent: QuoteIngestionAgent) -> None:
        q = agent.extract(QUOTES_DIR / "quote_01_apex_precision_en.csv")
        assert isinstance(q, NormalizedQuote)
        assert q.supplier_id == "SUP-APEX-US"
        assert q.source_language == "en"
        assert q.currency == "USD"
        assert q.incoterm == "DDP"
        assert q.lead_time_value == pytest.approx(12.0)
        assert q.lead_time_unit == "business_days"
        assert len(q.price_tiers) == 3
        assert q.tooling_fee == pytest.approx(450.0)
        assert q.shipping_fee == pytest.approx(0.0)
        assert q.quote_valid_until == "2026-12-31"
        assert any("Deduplicated" in a for a in q.master_data_hygiene_actions)
        assert any("BOM" in a for a in q.master_data_hygiene_actions)

    def test_quote_02_lyon_french_markdown_ocr(self, agent: QuoteIngestionAgent) -> None:
        q = agent.extract(QUOTES_DIR / "quote_02_atelier_lyon_fr.md")
        assert q.supplier_id == "SUP-LYON-FR"
        assert q.source_language == "fr"
        assert q.currency == "EUR"
        assert q.incoterm == "FOB"
        assert q.incoterm_location == "Port de Marseille"
        assert q.lead_time_value == pytest.approx(3.0)
        assert q.lead_time_unit == "weeks"
        assert len(q.price_tiers) == 3
        assert q.price_tiers[1].unit_price == pytest.approx(134.5)
        assert q.tooling_fee == pytest.approx(350.0)
        assert q.shipping_fee is None
        assert q.quote_valid_until == "2026-12-20"

    def test_quote_03_queretaro_zero_hallucination_missing_validity(
        self, agent: QuoteIngestionAgent
    ) -> None:
        q = agent.extract(QUOTES_DIR / "quote_03_industrias_queretaro_en.csv")
        assert q.supplier_id == "SUP-QRO-MX"
        assert q.source_language == "en"
        assert q.currency == "MXN"
        assert q.incoterm == "EXW"
        assert q.lead_time_value == pytest.approx(28.0)
        assert q.lead_time_unit == "calendar_days"
        assert q.tooling_fee == pytest.approx(6500.0)
        assert q.shipping_fee is None
        # Zero hallucination check: must stay None and be flagged
        assert q.quote_valid_until is None
        assert any(f.field_name == "quote_valid_until" for f in q.ambiguity_flags)

    def test_quote_04_toulouse_range_and_payment_ambiguity_flagged(
        self, agent: QuoteIngestionAgent
    ) -> None:
        q = agent.extract(QUOTES_DIR / "quote_04_meca_toulouse_fr.csv")
        assert q.supplier_id == "SUP-TLS-FR"
        assert q.source_language == "fr"
        assert q.currency == "EUR"
        assert q.incoterm == "EXW"
        assert q.lead_time_value == pytest.approx(4.0)
        assert q.lead_time_max_value == pytest.approx(6.0)
        assert q.lead_time_unit == "weeks"
        assert q.shipping_fee == pytest.approx(1850.0)
        assert q.tooling_fee == pytest.approx(500.0)
        assert q.payment_terms is not None and "convenir" in q.payment_terms.lower()
        flag_fields = {f.field_name for f in q.ambiguity_flags}
        assert "lead_time" in flag_fields
        assert "payment_terms" in flag_fields

    def test_quote_05_midwest_ocr_txt(self, agent: QuoteIngestionAgent) -> None:
        q = agent.extract(QUOTES_DIR / "quote_05_midwest_cnc_en.txt")
        assert q.supplier_id == "SUP-MWC-US"
        assert q.source_language == "en"
        assert q.currency == "USD"
        assert q.incoterm == "FOB"
        assert q.lead_time_value == pytest.approx(21.0)
        assert q.lead_time_unit == "calendar_days"
        assert q.shipping_fee == pytest.approx(1150.0)
        assert q.tooling_fee == pytest.approx(0.0)
        assert q.quote_valid_until == "2026-12-28"

    def test_quote_06_montreal_zero_hallucination_missing_incoterm_and_lead_time(
        self, agent: QuoteIngestionAgent
    ) -> None:
        q = agent.extract(QUOTES_DIR / "quote_06_usinage_montreal_fr.md")
        assert q.supplier_id == "SUP-MTL-CA"
        assert q.source_language == "fr"
        assert q.currency == "MXN"
        assert len(q.price_tiers) == 3
        assert q.price_tiers[1].unit_price == pytest.approx(2720.0)
        assert q.tooling_fee == pytest.approx(8000.0)
        # Zero hallucination check: both missing/TBD fields must remain None and be flagged
        assert q.incoterm is None
        assert q.lead_time_value is None
        assert q.lead_time_unit is None
        flag_fields = {f.field_name for f in q.ambiguity_flags}
        assert "incoterm" in flag_fields
        assert "lead_time" in flag_fields
