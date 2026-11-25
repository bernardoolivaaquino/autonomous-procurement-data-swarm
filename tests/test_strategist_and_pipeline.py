"""Tests for ProcurementStrategistAgent, EN/FR counteroffer fidelity, pipeline, and Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.agents.strategist_agent import ProcurementStrategistAgent
from src.cli import app
from src.pipeline import ProcurementSwarmPipeline
from src.schemas import ProcurementStrategyReport

runner = CliRunner()


class TestStrategistAndPipeline:
    def test_end_to_end_pipeline_outputs_and_language_fidelity(self, tmp_path: Path) -> None:
        pipeline = ProcurementSwarmPipeline(model="deterministic")
        result = pipeline.run(quotes_dir="data/quotes", output_dir=tmp_path)

        # Winner verification
        report: ProcurementStrategyReport = result.strategy_report
        assert report.recommended_supplier_id == "SUP-LYON-FR"
        assert report.recommended_tco_usd == pytest.approx(39234.28, abs=0.05)

        # Non-winners: 5 counteroffers (Quotes 1, 3, 4, 5, 6)
        assert len(report.counteroffers) == 5
        co_by_supplier = {c.supplier_id: c for c in report.counteroffers}
        assert "SUP-LYON-FR" not in co_by_supplier

        expected_langs = {
            "SUP-APEX-US": "en",
            "SUP-QRO-MX": "en",
            "SUP-MWC-US": "en",
            "SUP-TLS-FR": "fr",
            "SUP-MTL-CA": "fr",
        }
        for sup_id, expected_lang in expected_langs.items():
            draft = co_by_supplier[sup_id]
            assert draft.language == expected_lang
            if expected_lang == "fr":
                assert "Bonjour" in draft.email_body
                assert "Cordialement" in draft.email_body
                assert "Contre-offre" in draft.subject
            else:
                assert "Dear " in draft.email_body
                assert "Best regards" in draft.email_body
                assert "Counteroffer" in draft.subject

        # Incomplete quote counteroffer must request missing Incoterm & lead time in French
        mtl_body = co_by_supplier["SUP-MTL-CA"].email_body
        assert "Incoterm" in mtl_body
        assert "Délai de livraison" in mtl_body

        # Verify all expected files created in output dir
        expected_files = [
            tmp_path / "normalized_quotes.json",
            tmp_path / "master_data_cleaned.csv",
            tmp_path / "tco_comparison.json",
            tmp_path / "tco_comparison.csv",
            tmp_path / "executive_report.md",
            tmp_path / "pipeline_metrics.json",
        ]
        for path in expected_files:
            assert path.exists() and path.stat().st_size > 0

        counteroffer_files = list((tmp_path / "counteroffers").glob("*.md"))
        assert len(counteroffer_files) == 5

        metrics = json.loads((tmp_path / "pipeline_metrics.json").read_text(encoding="utf-8"))
        assert metrics["quotes_processed"] == 6
        assert metrics["complete_quotes"] == 5
        assert metrics["incomplete_quotes"] == 1
        assert metrics["recommended_supplier_id"] == "SUP-LYON-FR"

    def test_strategist_agent_direct_invocation(self) -> None:
        pipeline = ProcurementSwarmPipeline(model="deterministic")
        quotes = pipeline.ingest_quotes("data/quotes")
        breakdowns = pipeline.evaluate_tco(quotes)
        strategist = ProcurementStrategistAgent(model="deterministic")
        report = strategist.synthesize(quotes=quotes, tco_breakdowns=breakdowns)
        assert report.recommended_supplier_id == "SUP-LYON-FR"
        assert len(report.counteroffers) == 5


class TestTyperCLI:
    def test_cli_validate(self) -> None:
        res = runner.invoke(
            app,
            ["validate", "--quotes-dir", "data/quotes", "--model", "deterministic"],
            env={"COLUMNS": "200"},
        )
        assert res.exit_code == 0, res.output
        assert "SUP-LYON-FR" in res.output
        assert "SUP-MTL-CA" in res.output
        assert "MISSING" in res.output

    def test_cli_run_and_calculate(self, tmp_path: Path) -> None:
        run_res = runner.invoke(
            app,
            [
                "run",
                "--quotes-dir",
                "data/quotes",
                "--output-dir",
                str(tmp_path),
                "--model",
                "deterministic",
            ],
        )
        assert run_res.exit_code == 0, run_res.output
        assert "Atelier Mécanique de Lyon SAS" in run_res.output

        calc_out = tmp_path / "recalc"
        calc_res = runner.invoke(
            app,
            [
                "calculate",
                "--normalized-json",
                str(tmp_path / "normalized_quotes.json"),
                "--output-dir",
                str(calc_out),
            ],
        )
        assert calc_res.exit_code == 0, calc_res.output
        assert (calc_out / "tco_comparison.json").exists()
        assert (calc_out / "tco_comparison.csv").exists()
