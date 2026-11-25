"""Typer + Rich CLI for the Autonomous Procurement Data Swarm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.calculator import calculate_quote_tco, rank_quotes
from src.pipeline import ProcurementSwarmPipeline, write_tco_csv
from src.schemas import NormalizedQuote, QuoteTCOBreakdown, TCOCalculationConfig


app = typer.Typer(
    name="procurement-swarm",
    help="Autonomous multi-agent procurement data engineering and deterministic TCO swarm.",
    add_completion=False,
)
console = Console()


def _render_tco_table(breakdowns: list[QuoteTCOBreakdown]) -> Table:
    table = Table(title="Deterministic Total Cost of Ownership (TCO) Ranking", show_lines=False)
    table.add_column("Rank", justify="center", style="bold cyan")
    table.add_column("Supplier", style="white")
    table.add_column("Lang", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Unit USD", justify="right")
    table.add_column("Tooling USD", justify="right")
    table.add_column("Logistics USD", justify="right")
    table.add_column("Lead (bd)", justify="right")
    table.add_column("Penalty USD", justify="right")
    table.add_column("Total TCO USD", justify="right", style="bold green")
    table.add_column("Risk", justify="right")

    for b in breakdowns:
        status_style = "green" if b.status == "COMPLETE" else "yellow"
        table.add_row(
            f"#{b.rank}" if b.rank is not None else "N/A",
            f"{b.supplier_name} ({b.supplier_id})",
            b.source_language,
            f"[{status_style}]{b.status}[/{status_style}]",
            f"${b.unit_price_usd:,.2f}" if b.unit_price_usd is not None else "N/A",
            f"${b.tooling_fee_usd:,.2f}",
            f"${b.shipping_and_incoterm_usd:,.2f}",
            f"{b.lead_time_business_days:.1f}" if b.lead_time_business_days is not None else "TBD",
            f"${b.delay_penalty_usd:,.2f}",
            f"${b.total_cost_of_ownership_usd:,.2f}" if b.total_cost_of_ownership_usd is not None else "INCOMPLETE",
            f"{b.delivery_risk_score:.1f}",
        )
    return table


@app.command()
def run(
    quotes_dir: Path = typer.Option(Path("data/quotes"), "--quotes-dir", help="Directory containing raw supplier quotes."),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", help="Directory where output artifacts are written."),
    quantity: int = typer.Option(250, "--quantity", help="Target RFQ order quantity."),
    target_lead_time: float = typer.Option(15.0, "--target-lead-time", help="Target lead time in business days."),
    delay_penalty: float = typer.Option(350.0, "--delay-penalty", help="Delay penalty per excess business day in USD."),
    model: Optional[str] = typer.Option(None, "--model", help="PydanticAI model string or 'deterministic'."),
) -> None:
    """Run the full ingestion -> deterministic TCO -> strategist pipeline."""
    config = TCOCalculationConfig(
        target_quantity=quantity,
        target_lead_time_business_days=target_lead_time,
        delay_penalty_per_day_usd=delay_penalty,
    )
    pipeline = ProcurementSwarmPipeline(model=model)
    result = pipeline.run(quotes_dir=quotes_dir, output_dir=output_dir, config=config)

    console.print(
        Panel.fit(
            f"[bold green]Winner:[/bold green] {result.strategy_report.recommended_supplier_name} "
            f"(`{result.strategy_report.recommended_supplier_id}`)\n"
            f"[bold]Landed TCO:[/bold] ${result.strategy_report.recommended_tco_usd:,.2f} USD | "
            f"[bold]Model:[/bold] {result.metrics.model_used}\n"
            f"[bold]Quotes:[/bold] {result.metrics.quotes_processed} "
            f"({result.metrics.complete_quotes} COMPLETE, {result.metrics.incomplete_quotes} INCOMPLETE_DATA) | "
            f"[bold]Counteroffers:[/bold] {result.metrics.counteroffers_generated}",
            title=f"Procurement Swarm — {result.strategy_report.rfq_reference}",
        )
    )
    console.print(_render_tco_table(result.tco_breakdowns))
    console.print(f"[dim]Artifacts written to: {output_dir.resolve()}[/dim]")


@app.command()
def validate(
    quotes_dir: Path = typer.Option(Path("data/quotes"), "--quotes-dir", help="Directory containing raw supplier quotes."),
    model: Optional[str] = typer.Option(None, "--model", help="PydanticAI model string or 'deterministic'."),
) -> None:
    """Validate and normalize raw supplier quotes without writing strategy outputs."""
    pipeline = ProcurementSwarmPipeline(model=model)
    quotes = pipeline.ingest_quotes(quotes_dir)

    table = Table(title="Quote Ingestion & Master Data Hygiene Validation")
    table.add_column("Quote ID", style="cyan")
    table.add_column("Supplier")
    table.add_column("Lang")
    table.add_column("CCY")
    table.add_column("Incoterm")
    table.add_column("Lead Time")
    table.add_column("Tiers", justify="right")
    table.add_column("Flags", justify="right")
    table.add_column("Hygiene", justify="right")

    for q in quotes:
        lt_str = (
            f"{q.lead_time_value:g}-{q.lead_time_max_value:g} {q.lead_time_unit}"
            if q.lead_time_value is not None and q.lead_time_max_value is not None
            else (f"{q.lead_time_value:g} {q.lead_time_unit}" if q.lead_time_value is not None else "MISSING")
        )
        table.add_row(
            q.quote_id,
            f"{q.supplier_name} ({q.supplier_id})",
            q.source_language,
            q.currency,
            q.incoterm or "MISSING",
            lt_str,
            str(len(q.price_tiers)),
            str(len(q.ambiguity_flags)),
            str(len(q.master_data_hygiene_actions)),
        )

    console.print(table)


@app.command()
def calculate(
    normalized_json: Path = typer.Option(
        Path("output/normalized_quotes.json"),
        "--normalized-json",
        help="Path to normalized_quotes.json.",
    ),
    quotes_dir: Path = typer.Option(
        Path("data/quotes"),
        "--quotes-dir",
        help="Fallback raw quotes directory if normalized JSON does not exist.",
    ),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Optional directory to write TCO JSON/CSV."),
    quantity: int = typer.Option(250, "--quantity", help="Target RFQ order quantity."),
    target_lead_time: float = typer.Option(15.0, "--target-lead-time", help="Target lead time in business days."),
    delay_penalty: float = typer.Option(350.0, "--delay-penalty", help="Delay penalty per excess business day in USD."),
) -> None:
    """Run deterministic TCO calculations from normalized quotes (or raw quotes fallback)."""
    config = TCOCalculationConfig(
        target_quantity=quantity,
        target_lead_time_business_days=target_lead_time,
        delay_penalty_per_day_usd=delay_penalty,
    )
    if normalized_json.exists():
        raw_items = json.loads(normalized_json.read_text(encoding="utf-8"))
        quotes = [NormalizedQuote.model_validate(item) for item in raw_items]
    else:
        pipeline = ProcurementSwarmPipeline()
        quotes = pipeline.ingest_quotes(quotes_dir)

    ranked = rank_quotes([calculate_quote_tco(q, config) for q in quotes])
    console.print(_render_tco_table(ranked))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tco_comparison.json").write_text(
            json.dumps([b.model_dump() for b in ranked], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_tco_csv(ranked, output_dir / "tco_comparison.csv")
        console.print(f"[dim]TCO outputs written to: {output_dir.resolve()}[/dim]")


if __name__ == "__main__":
    app()
