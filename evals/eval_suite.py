"""End-to-end evaluation harness comparing pipeline outputs against evals/ground_truth.json."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.pipeline import ProcurementSwarmPipeline
from src.schemas import NormalizedQuote, QuoteTCOBreakdown, TCOCalculationConfig

GROUND_TRUTH_PATH = Path("evals/ground_truth.json")
EVAL_REPORT_PATH = Path("evals/eval_report.json")
TCO_TOLERANCE_USD = 0.05


def _check_language_content(body: str, subject: str, expected_lang: str) -> bool:
    """Verify counteroffer text actually matches the expected language."""
    combined = f"{subject}\n{body}"
    if expected_lang == "fr":
        return "Bonjour" in combined and "Cordialement" in combined and "Contre-offre" in combined
    if expected_lang == "en":
        return "Dear " in combined and "Best regards" in combined and "Counteroffer" in combined
    return False


def run_evaluation(
    quotes_dir: Path = Path("data/quotes"),
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    report_path: Path = EVAL_REPORT_PATH,
) -> dict[str, Any]:
    """Run the full procurement swarm and score every quality gate against ground truth."""
    gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    cfg = TCOCalculationConfig(**gt["evaluation_config"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = ProcurementSwarmPipeline(model="deterministic")
        result = pipeline.run(quotes_dir=quotes_dir, output_dir=tmp_dir, config=cfg)

    quotes_by_file = {Path(q.source_file).name: q for q in result.normalized_quotes}
    tco_by_sup = {b.supplier_id: b for b in result.tco_breakdowns}
    co_by_sup = {c.supplier_id: c for c in result.strategy_report.counteroffers}

    total_quotes = len(gt["quotes"])
    schema_compliant = 0
    hallucinations = 0
    field_checks_total = 0
    field_checks_passed = 0
    tco_checks_total = 0
    tco_checks_passed = 0
    counteroffer_checks_total = 0
    counteroffer_checks_passed = 0
    per_quote_details: list[dict[str, Any]] = []

    for expected in gt["quotes"]:
        fname = expected["source_file"]
        q = quotes_by_file.get(fname)
        detail: dict[str, Any] = {"source_file": fname, "supplier_id": expected["supplier_id"], "issues": []}

        # 1. Schema compliance gate
        if q is not None:
            NormalizedQuote.model_validate(q.model_dump())
            schema_compliant += 1
        else:
            detail["issues"].append("Missing normalized quote output.")
            per_quote_details.append(detail)
            continue

        # 2. Field extraction accuracy
        field_comparisons = [
            ("supplier_id", q.supplier_id == expected["supplier_id"]),
            ("supplier_name", q.supplier_name == expected["supplier_name"]),
            ("source_language", q.source_language == expected["source_language"]),
            ("currency", q.currency == expected["currency"]),
            ("incoterm", q.incoterm == expected["incoterm"]),
            ("lead_time_value", q.lead_time_value == expected["lead_time_value"]),
            ("lead_time_max_value", q.lead_time_max_value == expected["lead_time_max_value"]),
            ("lead_time_unit", q.lead_time_unit == expected["lead_time_unit"]),
            ("tier_count", len(q.price_tiers) == expected["expected_tier_count"]),
            ("quote_valid_until", q.quote_valid_until == expected["quote_valid_until"]),
            (
                "deduplication",
                (not expected["deduplication_expected"])
                or any("Deduplicated" in a for a in q.master_data_hygiene_actions),
            ),
        ]
        for field_name, ok in field_comparisons:
            field_checks_total += 1
            if ok:
                field_checks_passed += 1
            else:
                detail["issues"].append(f"Field mismatch on {field_name}")

        # 3. Zero-hallucination gate (missing fields stay None + ambiguity flags present)
        for null_field in expected["must_be_null_fields"]:
            if getattr(q, null_field) is not None:
                hallucinations += 1
                detail["issues"].append(f"Hallucinated non-null value for {null_field}")

        flag_fields = {f.field_name for f in q.ambiguity_flags}
        for flagged_field in expected["must_flag_ambiguity_fields"]:
            field_checks_total += 1
            if flagged_field in flag_fields:
                field_checks_passed += 1
            else:
                detail["issues"].append(f"Missing expected ambiguity flag for {flagged_field}")

        # 4. Deterministic TCO accuracy gate
        b = tco_by_sup.get(expected["supplier_id"])
        if b is None:
            tco_checks_total += 1
            detail["issues"].append("Missing TCO breakdown.")
        else:
            QuoteTCOBreakdown.model_validate(b.model_dump())
            tco_checks_total += 1
            status_ok = b.status == expected["expected_status"]
            rank_ok = b.rank == expected["expected_rank"]
            lt_ok = (
                b.lead_time_business_days is None and expected["lead_time_business_days"] is None
            ) or (
                b.lead_time_business_days is not None
                and expected["lead_time_business_days"] is not None
                and abs(b.lead_time_business_days - expected["lead_time_business_days"]) <= 0.05
            )
            tco_ok = (
                b.total_cost_of_ownership_usd is None and expected["total_cost_of_ownership_usd"] is None
            ) or (
                b.total_cost_of_ownership_usd is not None
                and expected["total_cost_of_ownership_usd"] is not None
                and abs(b.total_cost_of_ownership_usd - expected["total_cost_of_ownership_usd"]) <= TCO_TOLERANCE_USD
            )
            if status_ok and rank_ok and lt_ok and tco_ok:
                tco_checks_passed += 1
            else:
                detail["issues"].append(
                    f"TCO mismatch: status={b.status}, rank={b.rank}, tco={b.total_cost_of_ownership_usd}"
                )

        # 5. Counteroffer language fidelity gate
        expected_co_lang = expected["expected_counteroffer_language"]
        if expected_co_lang is not None:
            counteroffer_checks_total += 1
            draft = co_by_sup.get(expected["supplier_id"])
            if (
                draft is not None
                and draft.language == expected_co_lang
                and _check_language_content(draft.email_body, draft.subject, expected_co_lang)
            ):
                counteroffer_checks_passed += 1
            else:
                detail["issues"].append(f"Counteroffer language mismatch (expected {expected_co_lang})")

        per_quote_details.append(detail)

    winner_ok = result.strategy_report.recommended_supplier_id == gt["expected_winner_supplier_id"]
    actual_ranking = [
        b.supplier_id for b in result.tco_breakdowns if b.status == "COMPLETE"
    ]
    ranking_ok = actual_ranking == gt["expected_ranking_order"]

    schema_compliance_rate = round(schema_compliant / total_quotes, 4) if total_quotes else 0.0
    field_accuracy_rate = round(field_checks_passed / field_checks_total, 4) if field_checks_total else 0.0
    tco_accuracy_rate = round(tco_checks_passed / tco_checks_total, 4) if tco_checks_total else 0.0
    counteroffer_fidelity_rate = (
        round(counteroffer_checks_passed / counteroffer_checks_total, 4)
        if counteroffer_checks_total
        else 0.0
    )

    gates = {
        "schema_compliance_100pct": schema_compliance_rate == 1.0,
        "zero_hallucinations": hallucinations == 0,
        "field_extraction_100pct": field_accuracy_rate == 1.0,
        "tco_accuracy_100pct_within_5_cents": tco_accuracy_rate == 1.0,
        "counteroffer_language_fidelity_100pct": counteroffer_fidelity_rate == 1.0,
        "winner_and_ranking_exact_match": winner_ok and ranking_ok,
    }
    all_passed = all(gates.values())

    report = {
        "rfq_reference": gt["rfq_reference"],
        "all_gates_passed": all_passed,
        "metrics": {
            "quotes_evaluated": total_quotes,
            "schema_compliance_rate": schema_compliance_rate,
            "field_extraction_accuracy_rate": field_accuracy_rate,
            "field_checks_passed": field_checks_passed,
            "field_checks_total": field_checks_total,
            "hallucination_count": hallucinations,
            "tco_accuracy_rate": tco_accuracy_rate,
            "tco_checks_passed": tco_checks_passed,
            "tco_checks_total": tco_checks_total,
            "counteroffer_language_fidelity_rate": counteroffer_fidelity_rate,
            "counteroffer_checks_passed": counteroffer_checks_passed,
            "counteroffer_checks_total": counteroffer_checks_total,
            "recommended_supplier_id": result.strategy_report.recommended_supplier_id,
            "actual_ranking_order": actual_ranking,
        },
        "gates": gates,
        "per_quote_details": per_quote_details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    report = run_evaluation()
    print(json.dumps(report["metrics"], indent=2))
    print("GATES:", json.dumps(report["gates"], indent=2))
    if not report["all_gates_passed"]:
        print("EVALUATION FAILED:", json.dumps(report["per_quote_details"], indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
