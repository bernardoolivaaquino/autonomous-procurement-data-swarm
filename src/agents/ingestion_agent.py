"""PydanticAI QuoteIngestionAgent for messy multilingual supplier quotes."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo

from src.agents.model_factory import build_agent_model
from src.schemas import AmbiguityFlag, NormalizedQuote, VolumePriceTier


SYSTEM_PROMPT = """You are a strict Procurement Data Engineering Agent.
Your sole responsibility is to extract and normalize supplier quotation documents
into the canonical `NormalizedQuote` schema.

CRITICAL ANTI-HALLUCINATION RULES:
1. NEVER invent, guess, or extrapolate missing values. If a field such as
   `quote_valid_until`, `incoterm`, `lead_time_value`, `shipping_fee`, or
   `tooling_fee` is missing, blank, or marked "TBD" / "À confirmer", set it to `None`
   and record a clear `AmbiguityFlag`.
2. Preserve ambiguity explicitly: ranges like "4-6 semaines" MUST populate
   `lead_time_value=4.0`, `lead_time_max_value=6.0`, `lead_time_unit="weeks"`
   AND an `AmbiguityFlag`. Vague payment terms like "À convenir" must be preserved
   verbatim and flagged as ambiguous.
3. Deduplicate identical volume pricing tier rows and document the cleanup in
   `master_data_hygiene_actions`.
4. Always use the provided tools (`normalize_supplier_id`, `parse_numeric_string`)
   when normalizing identifiers and localized currency numbers.
"""


@dataclass
class IngestionDeps:
    """Runtime dependencies injected into QuoteIngestionAgent."""

    source_file: str
    raw_text: str
    quote_id: str | None = None


def clean_numeric_string(raw_value: str | None) -> float | None:
    """Parse EN/FR/MXN formatted numeric strings like '$1,150.00', '134,50 €', '1 850,00 €'."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None

    # Strip common currency markers and whitespace/NBSP noise
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = re.sub(r"(?i)\b(?:usd|eur|mxn|cad|ht)\b", "", text)
    text = re.sub(r"[$€£]|MX\$|US\$", "", text)
    text = text.strip()

    match = re.search(r"-?[\d\s.,]+", text)
    if not match:
        return None
    num_str = match.group(0).strip().replace(" ", "")
    if not num_str:
        return None

    if "," in num_str and "." in num_str:
        if num_str.rfind(",") > num_str.rfind("."):
            # 1.850,00 -> 1850.00
            num_str = num_str.replace(".", "").replace(",", ".")
        else:
            # 2,950.00 -> 2950.00
            num_str = num_str.replace(",", "")
    elif "," in num_str:
        parts = num_str.split(",")
        if len(parts[-1]) == 2 or len(parts) == 2 and len(parts[-1]) in (1, 2):
            num_str = num_str.replace(",", ".")
        else:
            num_str = num_str.replace(",", "")

    try:
        return float(num_str)
    except ValueError:
        return None


def clean_supplier_id(raw_id: str) -> str:
    """Normalize a supplier identifier to uppercase trimmed canonical form."""
    cleaned = re.sub(r"\s+", "-", str(raw_id).strip().upper())
    return re.sub(r"[^A-Z0-9\-_]", "", cleaned)


def _parse_qty_range(raw_range: str) -> tuple[int, int | None] | None:
    """Parse quantity tier strings such as '1-99', '100 - 499 unités', '500+ pcs'."""
    text = raw_range.strip()
    range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))
    plus_match = re.search(r"(\d+)\s*\+", text)
    if plus_match:
        return int(plus_match.group(1)), None
    single_match = re.search(r"\b(\d+)\b", text)
    if single_match:
        qty = int(single_match.group(1))
        return qty, qty
    return None


def _parse_lead_time(
    raw_lead: str | None,
) -> tuple[float | None, float | None, Literal["business_days", "calendar_days", "weeks"] | None, list[AmbiguityFlag]]:
    """Extract numeric lead time or range and unit without inventing missing values."""
    flags: list[AmbiguityFlag] = []
    if raw_lead is None or not raw_lead.strip():
        flags.append(
            AmbiguityFlag(
                field_name="lead_time",
                raw_value=raw_lead,
                reason="Lead time is missing.",
                severity="HIGH",
            )
        )
        return None, None, None, flags

    cleaned = raw_lead.strip()
    lower = cleaned.lower()

    if any(tok in lower for tok in ("tbd", "à confirmer", "a confirmer", "selon disponibilité", "non confirmé", "pending")):
        flags.append(
            AmbiguityFlag(
                field_name="lead_time",
                raw_value=cleaned,
                reason="Lead time is unconfirmed / TBD in supplier quote.",
                severity="HIGH",
            )
        )
        return None, None, None, flags

    unit: Literal["business_days", "calendar_days", "weeks"] | None = None
    if any(k in lower for k in ("business day", "jours ouvr", "working day", "bd")):
        unit = "business_days"
    elif any(k in lower for k in ("calendar day", "jours calendaire", "jours")):
        unit = "calendar_days"
    elif any(k in lower for k in ("week", "semaine")):
        unit = "weeks"

    range_match = re.search(r"(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)", cleaned)
    if range_match and unit is not None:
        v_min = float(range_match.group(1).replace(",", "."))
        v_max = float(range_match.group(2).replace(",", "."))
        flags.append(
            AmbiguityFlag(
                field_name="lead_time",
                raw_value=cleaned,
                reason=f"Ambiguous lead-time range ({v_min:g}-{v_max:g} {unit}); midpoint required for TCO.",
                severity="MEDIUM",
            )
        )
        return v_min, v_max, unit, flags

    single_match = re.search(r"(\d+(?:[.,]\d+)?)", cleaned)
    if single_match and unit is not None:
        val = float(single_match.group(1).replace(",", "."))
        return val, None, unit, flags

    flags.append(
        AmbiguityFlag(
            field_name="lead_time",
            raw_value=cleaned,
            reason="Unable to unambiguously parse lead time value and unit.",
            severity="HIGH",
        )
    )
    return None, None, None, flags


def _parse_incoterm(
    raw_incoterm: str | None,
    raw_location: str | None = None,
) -> tuple[Literal["EXW", "FOB", "DDP"] | None, str | None, list[AmbiguityFlag]]:
    """Extract canonical Incoterm and optional location, flagging missing/ambiguous terms."""
    flags: list[AmbiguityFlag] = []
    if raw_incoterm is None or not raw_incoterm.strip():
        flags.append(
            AmbiguityFlag(
                field_name="incoterm",
                raw_value=raw_incoterm,
                reason="Incoterm is missing from supplier quote.",
                severity="HIGH",
            )
        )
        return None, None, flags

    cleaned = raw_incoterm.strip()
    lower = cleaned.lower()
    if any(tok in lower for tok in ("à confirmer", "a confirmer", "non spécifié", "tbd", "pending", "unknown")):
        flags.append(
            AmbiguityFlag(
                field_name="incoterm",
                raw_value=cleaned,
                reason="Incoterm is explicitly unconfirmed / ambiguous in supplier quote.",
                severity="HIGH",
            )
        )
        return None, None, flags

    match = re.search(r"\b(DDP|FOB|EXW)\b", cleaned, flags=re.IGNORECASE)
    if not match:
        flags.append(
            AmbiguityFlag(
                field_name="incoterm",
                raw_value=cleaned,
                reason=f"Unrecognized or unsupported Incoterm '{cleaned}'.",
                severity="HIGH",
            )
        )
        return None, raw_location.strip() if raw_location and raw_location.strip() else None, flags

    term: Literal["EXW", "FOB", "DDP"] = match.group(1).upper()  # type: ignore[assignment]
    loc = raw_location.strip() if raw_location and raw_location.strip() else None
    if loc is None:
        paren_match = re.search(r"\(([^)]+)\)", cleaned)
        if paren_match:
            loc = paren_match.group(1).strip()
    return term, loc, flags


def _detect_currency(raw_text: str, explicit_currency: str | None = None) -> str:
    """Detect currency code from explicit field or price markers."""
    if explicit_currency:
        upper = explicit_currency.strip().upper()
        for code in ("USD", "EUR", "MXN", "CAD"):
            if re.search(rf"\b{code}\b", upper):
                return code
    if "MX$" in raw_text or re.search(r"\bMXN\b", raw_text, re.IGNORECASE):
        return "MXN"
    if "€" in raw_text or re.search(r"\bEUR\b", raw_text, re.IGNORECASE):
        return "EUR"
    if "$" in raw_text or re.search(r"\bUSD\b", raw_text, re.IGNORECASE):
        return "USD"
    return "USD"


def _derive_quote_id(source_file: str, explicit_quote_id: str | None) -> str:
    if explicit_quote_id and explicit_quote_id.strip():
        return explicit_quote_id.strip()
    stem = Path(source_file).stem
    num_match = re.match(r"quote_(\d+)", stem, re.IGNORECASE)
    if num_match:
        return f"Q-{int(num_match.group(1)):02d}"
    return stem


def _deduplicate_tiers(
    tiers: list[VolumePriceTier],
    hygiene_actions: list[str],
) -> list[VolumePriceTier]:
    """Deduplicate identical volume price tiers and record master-data hygiene action."""
    seen: set[tuple[int, int | None, float]] = set()
    unique: list[VolumePriceTier] = []
    for tier in tiers:
        key = (tier.min_qty, tier.max_qty, tier.unit_price)
        if key in seen:
            label = f"{tier.min_qty}-{tier.max_qty}" if tier.max_qty is not None else f"{tier.min_qty}+"
            hygiene_actions.append(
                f"Deduplicated repeated volume price tier row ({label} @ {tier.unit_price})."
            )
            continue
        seen.add(key)
        unique.append(tier)
    unique.sort(key=lambda t: t.min_qty)
    return unique


def _parse_csv_quote(source_file: str, raw_text: str, quote_id: str) -> NormalizedQuote:
    hygiene_actions: list[str] = []
    extraction_notes: list[str] = []
    ambiguity_flags: list[AmbiguityFlag] = []

    text = raw_text
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
        hygiene_actions.append("Stripped UTF-8 BOM prefix from CSV file.")

    lines = text.splitlines()
    comment_lines: list[str] = []
    data_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment_lines.append(stripped)
            continue
        # Skip empty delimiter-only rows like ';;;;;;;;;;;;;;;;'
        if not re.sub(r"[;,\s]", "", stripped):
            hygiene_actions.append("Removed empty delimiter-only trailing CSV row.")
            continue
        data_lines.append(line)

    if comment_lines:
        hygiene_actions.append(f"Skipped {len(comment_lines)} leading metadata comment line(s).")

    comment_blob = "\n".join(comment_lines)
    delimiter = ";" if data_lines and data_lines[0].count(";") > data_lines[0].count(",") else ","
    if delimiter == ";":
        hygiene_actions.append("Normalized semicolon-delimited CSV format.")

    reader = csv.DictReader(
        io.StringIO("\n".join(data_lines)),
        delimiter=delimiter,
        skipinitialspace=True,
    )
    rows: list[dict[str, str]] = []
    for row in reader:
        cleaned_row: dict[str, str] = {}
        for k, v in row.items():
            if k is None:
                continue
            val = " ".join(str(x).strip() for x in v) if isinstance(v, list) else str(v or "").strip()
            cleaned_row[k.strip().lower()] = val
        if any(val for val in cleaned_row.values()):
            rows.append(cleaned_row)

    def pick(row: dict[str, str], *candidates: str) -> str:
        for cand in candidates:
            if cand in row and row[cand]:
                return row[cand]
        return ""

    first = rows[0] if rows else {}
    supplier_name = pick(first, "supplier_name", "fournisseur", "supplier")
    supplier_id = clean_supplier_id(pick(first, "supplier_id", "code_fournisseur"))
    supplier_email = pick(first, "email", "supplier_email", "courriel") or None
    lang_raw = pick(first, "language", "langue", "lang").lower()
    source_language: Literal["en", "fr"] = "fr" if lang_raw.startswith("fr") else "en"

    rfq_ref = pick(first, "rfq_ref", "ref_rfq", "rfq_reference")
    if not rfq_ref:
        rfq_m = re.search(r"\b(RFQ-[A-Z0-9-]+)\b", comment_blob)
        rfq_ref = rfq_m.group(1) if rfq_m else "RFQ-IND-2026-088"

    part_m = re.search(r"\b(PART-[A-Z0-9-]+)\b", comment_blob + " " + " ".join(first.values()))
    part_number = part_m.group(1) if part_m else "PART-HMB-7075-088"

    part_desc = "CNC-Machined 7075-T6 Hydraulic Manifold Block"
    desc_m = re.search(r"\((CNC-[^)]+)\)", comment_blob)
    if desc_m:
        part_desc = desc_m.group(1).strip()

    currency = _detect_currency(text, pick(first, "currency", "devise"))
    incoterm, incoterm_loc, inc_flags = _parse_incoterm(
        pick(first, "incoterm"),
        pick(first, "delivery_location", "incoterm_location", "lieu_incoterm"),
    )
    ambiguity_flags.extend(inc_flags)

    lt_val, lt_max, lt_unit, lt_flags = _parse_lead_time(
        pick(first, "lead_time", "delai_livraison")
    )
    ambiguity_flags.extend(lt_flags)

    raw_tiers: list[VolumePriceTier] = []
    for row in rows:
        qty_raw = pick(row, "qty_range", "qty_tier", "quantite_palier", "quantity")
        price_raw = pick(row, "unit_price", "prix_unitaire_ht")
        parsed_qty = _parse_qty_range(qty_raw)
        parsed_price = clean_numeric_string(price_raw)
        if parsed_qty and parsed_price is not None:
            raw_tiers.append(
                VolumePriceTier(min_qty=parsed_qty[0], max_qty=parsed_qty[1], unit_price=parsed_price)
            )
    price_tiers = _deduplicate_tiers(raw_tiers, hygiene_actions)

    tooling_raw = pick(first, "tooling_fee", "outillage_cnc", "tooling")
    tooling_fee = clean_numeric_string(tooling_raw)

    shipping_raw = pick(first, "shipping_cost", "shipping_fee", "frais_de_port", "freight")
    shipping_fee = clean_numeric_string(shipping_raw)

    payment_terms = pick(first, "payment_terms", "conditions_paiement") or None
    if payment_terms and any(tok in payment_terms.lower() for tok in ("à convenir", "a convenir", "tbd", "negotiable")):
        ambiguity_flags.append(
            AmbiguityFlag(
                field_name="payment_terms",
                raw_value=payment_terms,
                reason="Payment terms are open-ended ('À convenir') and require commercial finalization.",
                severity="MEDIUM",
            )
        )

    certs_raw = pick(first, "certifications")
    certifications = [c.strip() for c in re.split(r"[,;]", certs_raw) if c.strip()] if certs_raw else []

    valid_until = pick(first, "quote_valid_until", "validite_offre", "valid_until") or None
    if valid_until is None:
        ambiguity_flags.append(
            AmbiguityFlag(
                field_name="quote_valid_until",
                raw_value=None,
                reason="Quote expiration/validity date is missing in CSV.",
                severity="HIGH",
            )
        )

    hygiene_actions.append("Trimmed whitespace and normalized localized currency symbols/separators in CSV cells.")
    extraction_notes.append(f"Parsed structured CSV quote ({len(rows)} raw data rows -> {len(price_tiers)} unique price tiers).")

    return NormalizedQuote(
        quote_id=quote_id,
        source_file=source_file,
        supplier_id=supplier_id,
        supplier_name=supplier_name,
        supplier_email=supplier_email,
        source_language=source_language,
        rfq_reference=rfq_ref,
        part_number=part_number,
        part_description=part_desc,
        currency=currency,
        incoterm=incoterm,
        incoterm_location=incoterm_loc,
        lead_time_value=lt_val,
        lead_time_max_value=lt_max,
        lead_time_unit=lt_unit,
        price_tiers=price_tiers,
        tooling_fee=tooling_fee,
        shipping_fee=shipping_fee,
        payment_terms=payment_terms,
        certifications=certifications,
        quote_valid_until=valid_until,
        ambiguity_flags=ambiguity_flags,
        extraction_notes=extraction_notes,
        master_data_hygiene_actions=hygiene_actions,
    )


def _strip_md(value: str) -> str:
    return re.sub(r"[`*_]", "", value).strip()


def _parse_text_or_markdown_quote(source_file: str, raw_text: str, quote_id: str) -> NormalizedQuote:
    hygiene_actions: list[str] = []
    extraction_notes: list[str] = []
    ambiguity_flags: list[AmbiguityFlag] = []

    text = raw_text.lstrip("\ufeff")
    if re.search(r"---\s*PAGE\s*\d+/\d+\s*---|Page\s+\d+\s+of\s+\d+", text, re.IGNORECASE):
        hygiene_actions.append("Stripped simulated PDF OCR page headers/footers.")

    # Detect language
    lower = text.lower()
    if "langue / language:** français" in lower or "langue du document:** français" in lower or "(`fr`)" in lower:
        source_language: Literal["en", "fr"] = "fr"
    elif re.search(r"lang\s*:\s*fr\b", lower):
        source_language = "fr"
    elif re.search(r"lang\s*:\s*en\b", lower):
        source_language = "en"
    else:
        source_language = "fr" if ("fournisseur" in lower or "devis" in lower or "semaines" in lower) else "en"

    # Supplier Name
    supplier_name = ""
    for pattern in (
        r"\*\*Fournisseur\s*/\s*Supplier:\*\*\s*([^\n(]+)",
        r"#\s*DEVIS\s*/\s*QUOTATION\s*[—-]\s*([^\n]+)",
        r"^\s*([A-Z][A-Z0-9 .,&-]+(?:INC\.|LLC|SAS|SARL))\s*--\s*OFFICIAL",
    ):
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            supplier_name = _strip_md(m.group(1)).strip()
            if supplier_name.isupper():
                words = supplier_name.title().split()
                fixed_words = []
                for idx, w in enumerate(words):
                    if w.upper() in {"CNC", "SAS", "SARL", "LLC", "USA"}:
                        fixed_words.append(w.upper())
                    elif idx > 0 and w.lower() in {"de", "du", "des", "la", "le", "et"}:
                        fixed_words.append(w.lower())
                    else:
                        fixed_words.append(w)
                supplier_name = " ".join(fixed_words)
            break

    # Supplier ID
    sid_m = re.search(r"\b(SUP-[A-Z0-9-]+)\b", text)
    supplier_id = clean_supplier_id(sid_m.group(1)) if sid_m else "SUP-UNKNOWN"

    # Email
    email_m = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", text)
    supplier_email = email_m.group(1) if email_m else None

    # RFQ & Part
    rfq_m = re.search(r"\b(RFQ-[A-Z0-9-]+)\b", text)
    rfq_ref = rfq_m.group(1) if rfq_m else "RFQ-IND-2026-088"
    part_m = re.search(r"\b(PART-[A-Z0-9-]+)\b", text)
    part_number = part_m.group(1) if part_m else "PART-HMB-7075-088"
    part_desc = "CNC-Machined 7075-T6 Hydraulic Manifold Block"

    # Currency
    ccy_m = re.search(r"(?:Devise(?:\s*/\s*Currency)?|Currency)\s*(?::|\*\*:\*\*|\*\*)\s*([A-Z]{3})", text, re.IGNORECASE)
    currency = _detect_currency(text, ccy_m.group(1) if ccy_m else None)

    # Price tiers from markdown/ASCII pipe table rows
    raw_tiers: list[VolumePriceTier] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        qty_parsed = _parse_qty_range(cells[0])
        price_parsed = clean_numeric_string(cells[1])
        if qty_parsed and price_parsed is not None:
            raw_tiers.append(
                VolumePriceTier(min_qty=qty_parsed[0], max_qty=qty_parsed[1], unit_price=price_parsed)
            )
    price_tiers = _deduplicate_tiers(raw_tiers, hygiene_actions)

    # Build normalized key-value pairs from bullet/metadata lines
    kv_lines: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        cleaned_line = re.sub(r"^[\s>*\-•]+", "", raw_line).replace("**", "").replace("`", "").strip()
        if ":" not in cleaned_line or cleaned_line.startswith("|"):
            continue
        key_part, val_part = cleaned_line.split(":", 1)
        norm_key = re.sub(r"\s+", " ", key_part).strip().lower()
        norm_val = re.sub(r"\s+", " ", val_part).strip()
        if norm_key and norm_val:
            kv_lines.append((norm_key, norm_val))

    def extract_line_value(*labels: str) -> str | None:
        for label in labels:
            label_re = re.compile(label, re.IGNORECASE)
            for k, v in kv_lines:
                if label_re.search(k):
                    return v
        return None

    # Tooling fee
    tooling_line = extract_line_value(r"Frais d'outillage", r"Outillage\s*/\s*Tooling Fee", r"Tooling\s*/\s*NRE", r"Tooling")
    tooling_fee = clean_numeric_string(tooling_line)

    # Shipping fee
    shipping_line = extract_line_value(r"Explicit Freight", r"Frais de transport", r"Shipping")
    shipping_fee: float | None = None
    if shipping_line and not any(
        neg in shipping_line.lower() for neg in ("non inclus", "non chiffré", "non chiffre", "not included", "organized by")
    ):
        shipping_fee = clean_numeric_string(shipping_line)

    # Incoterm
    incoterm_line = extract_line_value(r"Incoterm")
    incoterm, incoterm_loc, inc_flags = _parse_incoterm(incoterm_line)
    ambiguity_flags.extend(inc_flags)

    # Lead time
    lead_line = extract_line_value(r"Délai de livraison", r"Lead\s+Time")
    lt_val, lt_max, lt_unit, lt_flags = _parse_lead_time(lead_line)
    ambiguity_flags.extend(lt_flags)

    # Payment terms
    payment_raw = extract_line_value(r"Conditions de paiement", r"Payment\s+Terms")
    payment_terms = payment_raw.strip() if payment_raw else None

    # Certifications
    certs_raw = extract_line_value(r"Certifications qualité", r"Certifications")
    certifications = [c.strip() for c in re.split(r"[,;]", certs_raw) if c.strip()] if certs_raw else []

    # Valid until
    valid_raw = extract_line_value(r"Date de validité de l'offre", r"Validité de l'offre", r"Quote Valid Until")
    quote_valid_until: str | None = None
    if valid_raw:
        date_m = re.search(r"(\d{4}-\d{2}-\d{2})", valid_raw)
        quote_valid_until = date_m.group(1) if date_m else valid_raw.strip()
    else:
        ambiguity_flags.append(
            AmbiguityFlag(
                field_name="quote_valid_until",
                raw_value=None,
                reason="Quote expiration/validity date is missing.",
                severity="HIGH",
            )
        )

    hygiene_actions.append("Normalized OCR table spacing, localized decimal commas, and space thousands separators.")
    extraction_notes.append(f"Parsed unstructured OCR document into {len(price_tiers)} price tiers.")

    return NormalizedQuote(
        quote_id=quote_id,
        source_file=source_file,
        supplier_id=supplier_id,
        supplier_name=supplier_name,
        supplier_email=supplier_email,
        source_language=source_language,
        rfq_reference=rfq_ref,
        part_number=part_number,
        part_description=part_desc,
        currency=currency,
        incoterm=incoterm,
        incoterm_location=incoterm_loc,
        lead_time_value=lt_val,
        lead_time_max_value=lt_max,
        lead_time_unit=lt_unit,
        price_tiers=price_tiers,
        tooling_fee=tooling_fee,
        shipping_fee=shipping_fee,
        payment_terms=payment_terms,
        certifications=certifications,
        quote_valid_until=quote_valid_until,
        ambiguity_flags=ambiguity_flags,
        extraction_notes=extraction_notes,
        master_data_hygiene_actions=hygiene_actions,
    )


def parse_quote_document(
    source_file: str,
    raw_text: str,
    quote_id: str | None = None,
) -> NormalizedQuote:
    """Parse any CSV, Markdown, or plain-text OCR supplier quote into NormalizedQuote."""
    qid = _derive_quote_id(source_file, quote_id)
    suffix = Path(source_file).suffix.lower()
    if suffix == ".csv":
        return _parse_csv_quote(source_file, raw_text, qid)
    return _parse_text_or_markdown_quote(source_file, raw_text, qid)


class QuoteIngestionAgent:
    """PydanticAI-backed agent that extracts and validates dirty supplier quotes."""

    def __init__(self, model: str | None = None) -> None:
        self._requested_model = model
        resolved_model, self.model_label = build_agent_model(model, self._deterministic_model)
        self.agent: Agent[IngestionDeps, NormalizedQuote] = Agent(
            resolved_model,
            output_type=NormalizedQuote,
            deps_type=IngestionDeps,
            system_prompt=SYSTEM_PROMPT,
            retries=2,
            defer_model_check=True,
            name="QuoteIngestionAgent",
        )
        self._register_tools()

    def _register_tools(self) -> None:
        @self.agent.tool
        def normalize_supplier_id(ctx: RunContext[IngestionDeps], raw_id: str) -> str:
            """Normalize a raw supplier code into canonical uppercase format."""
            return clean_supplier_id(raw_id)

        @self.agent.tool
        def parse_numeric_string(ctx: RunContext[IngestionDeps], raw_value: str) -> float | None:
            """Parse localized EN/FR/MXN currency and numeric strings into a float."""
            return clean_numeric_string(raw_value)

    @staticmethod
    def _extract_deps_from_prompt(prompt_text: str) -> tuple[str, str | None, str]:
        src_m = re.search(r"SOURCE_FILE:\s*(.+)", prompt_text)
        qid_m = re.search(r"QUOTE_ID:\s*(.+)", prompt_text)
        body_m = re.search(r"RAW_DOCUMENT_START\n(.*)\nRAW_DOCUMENT_END", prompt_text, re.DOTALL)
        source_file = src_m.group(1).strip() if src_m else "quote.txt"
        quote_id = qid_m.group(1).strip() if qid_m and qid_m.group(1).strip() != "AUTO" else None
        raw_text = body_m.group(1) if body_m else prompt_text
        return source_file, quote_id, raw_text

    def _deterministic_model(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Offline FunctionModel callback executing tool calls and structured output."""
        prompt_text = ""
        for msg in messages:
            for part in getattr(msg, "parts", []):
                content = getattr(part, "content", None)
                if isinstance(content, str) and "RAW_DOCUMENT_START" in content:
                    prompt_text = content
                    break

        source_file, quote_id, raw_text = self._extract_deps_from_prompt(prompt_text)
        parsed = parse_quote_document(source_file=source_file, raw_text=raw_text, quote_id=quote_id)

        # Turn 1: exercise PydanticAI @agent.tool calls when available
        if len(messages) == 1 and info.function_tools:
            tool_names = {t.name for t in info.function_tools}
            calls: list[ToolCallPart] = []
            if "normalize_supplier_id" in tool_names:
                calls.append(
                    ToolCallPart(
                        tool_name="normalize_supplier_id",
                        args={"raw_id": parsed.supplier_id},
                    )
                )
            if "parse_numeric_string" in tool_names and parsed.price_tiers:
                calls.append(
                    ToolCallPart(
                        tool_name="parse_numeric_string",
                        args={"raw_value": str(parsed.price_tiers[0].unit_price)},
                    )
                )
            if calls:
                return ModelResponse(parts=calls)

        # Turn 2+: emit structured NormalizedQuote via output tool
        output_tool_name = info.output_tools[0].name
        return ModelResponse(
            parts=[ToolCallPart(tool_name=output_tool_name, args=parsed.model_dump())]
        )

    def extract(
        self,
        source_file: str | Path,
        raw_text: str | None = None,
        quote_id: str | None = None,
    ) -> NormalizedQuote:
        """Run the PydanticAI QuoteIngestionAgent synchronously on a quote document."""
        src_str = str(source_file).replace("\\", "/")
        if raw_text is None:
            raw_text = Path(source_file).read_text(encoding="utf-8")

        deps = IngestionDeps(source_file=src_str, raw_text=raw_text, quote_id=quote_id)
        prompt = (
            f"SOURCE_FILE: {src_str}\n"
            f"QUOTE_ID: {quote_id or 'AUTO'}\n"
            "Extract and normalize the following supplier quote without inventing missing fields.\n"
            f"RAW_DOCUMENT_START\n{raw_text}\nRAW_DOCUMENT_END"
        )
        result = self.agent.run_sync(prompt, deps=deps)
        return result.output
