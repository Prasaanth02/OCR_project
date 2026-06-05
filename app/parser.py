import re
import spacy

# en_ner_bc5cdr_md is pretrained on biomedical literature
# Labels: CHEMICAL (drugs, compounds) and DISEASE
# Install: pip install scispacy
#          pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_ner_bc5cdr_md")
    return _nlp


# Things NER wrongly labels as CHEMICAL — units, diluents, admin abbreviations
_NOT_DRUG = re.compile(
    r"^(dob|d\.o\.b|nhs|cga|date.*|time.*|batch|ward|weight|name|no\.|number"
    r"|glu|nacl|d5w|d10w|glucose|dextrose|sodium chloride|water|wfi|saline"
    r"|micrograms?.*|nanograms?.*|units?|milliunits?|mmol|meq|mcg|mg|ml|kg|hr|min"
    r"|[\d\s\.\-/]+)$",
    re.IGNORECASE,
)


def _extract_drug_names(text: str) -> list[tuple[int, int, str]]:
    """Return list of (start_char, end_char, drug_name) from NER, noise filtered."""
    nlp = _get_nlp()
    doc = nlp(text[:100000])
    results = []
    for ent in doc.ents:
        if ent.label_ != "CHEMICAL":
            continue
        name = ent.text.strip()
        # skip short tokens, noise patterns, newline-spanning spans, unit-heavy strings
        if len(name) < 4:
            continue
        if _NOT_DRUG.match(name):
            continue
        if "\n" in name:
            continue
        if sum(c.isalpha() for c in name) < len(name) * 0.5:
            continue
        results.append((ent.start_char, ent.end_char, name))
    return results


# --- Clinical field patterns (broad, not chart-specific) ---
_PATTERNS = {
    "concentration": re.compile(
        r"\d+\.?\d*\s*(mg|mcg|micrograms?|nanograms?|units?|milliunits?|mmol|mEq)"
        r"\s*/\s*(ml|L)",
        re.IGNORECASE,
    ),
    "dose_range": re.compile(
        r"\d+\.?\d*\s*[-–]\s*\d+\.?\d*\s*"
        r"(mg|mcg|micrograms?|nanograms?|units?|milliunits?|mmol|mEq|ml)"
        r"(/kg)?(/hr|/min|/dose)?",
        re.IGNORECASE,
    ),
    "single_dose": re.compile(
        r"\d+\.?\d*\s*(mg|mcg|micrograms?|nanograms?|units?|milliunits?|mmol|mEq|ml)"
        r"(/kg)?(/hr|/min|/dose)?",
        re.IGNORECASE,
    ),
    "max_dose": re.compile(
        r"\bmax(imum)?\b[\s:]*\d+\.?\d*\s*(mg|mcg|micrograms?|ml|units?|mmol)",
        re.IGNORECASE,
    ),
    "min_dose": re.compile(
        r"\bmin(imum)?\b[\s:]*\d+\.?\d*\s*(mg|mcg|micrograms?|ml|units?)",
        re.IGNORECASE,
    ),
    "infusion_rate": re.compile(
        r"\d+\.?\d*\s*ml\s*/\s*(kg\s*/\s*)?(hr|hour|min|minute)",
        re.IGNORECASE,
    ),
    "route": re.compile(
        r"\b(intravenous|IV|IM|intramuscular|subcutaneous|SC|oral|PO|PR|rectal"
        r"|intranasal|IN|intraosseous|IO|peripheral|central|NEAT)\b",
        re.IGNORECASE,
    ),
    "age_range": re.compile(
        r"(under\s*\d+\s*(kg|years?|months?)"
        r"|\d+\s*(kg|years?|months?)\s*(\+|and (over|above))?"
        r"|\d+\s*months?\s*(to|-)\s*\d+\s*(years?|months?))",
        re.IGNORECASE,
    ),
    "diluent": re.compile(
        r"(sodium chloride|NaCl|normal saline|glucose\s*\d+%"
        r"|dextrose\s*\d*%?|Glu\s*\d+%|D5W|D10W|hartmann|ringer"
        r"|water for injection|WFI)",
        re.IGNORECASE,
    ),
    "page_marker": re.compile(r"^---\s*page_\d+\.png\s*---$"),
    "category_header": re.compile(r"^[A-Z][A-Z\s/]{4,}$"),
    # Patient metadata lines to skip
    "skip_line": re.compile(
        r"\b(date of birth|dob|d\.o\.b|patient name|name:|nhs|hospital no"
        r"|ward:|consultant:|weight:|allergies|signature|signed|print name"
        r"|prescribed by|checked by|administered by|date:|time:)\b",
        re.IGNORECASE,
    ),
}


def _extract_fields(line: str) -> dict:
    fields = {}

    m = _PATTERNS["concentration"].search(line)
    if m:
        fields["concentration"] = m.group(0)

    dr = _PATTERNS["dose_range"].search(line)
    if dr:
        fields["dose_range"] = dr.group(0)
    else:
        sd = _PATTERNS["single_dose"].search(line)
        if sd:
            fields["dose"] = sd.group(0)

    mx = _PATTERNS["max_dose"].search(line)
    if mx:
        fields["max_dose"] = mx.group(0)

    mn = _PATTERNS["min_dose"].search(line)
    if mn:
        fields["min_dose"] = mn.group(0)

    ir = _PATTERNS["infusion_rate"].search(line)
    if ir:
        fields["infusion_rate"] = ir.group(0)

    rt = _PATTERNS["route"].search(line)
    if rt:
        fields["route"] = rt.group(0)

    ag = _PATTERNS["age_range"].search(line)
    if ag:
        fields["age_range"] = ag.group(0)

    dl = _PATTERNS["diluent"].search(line)
    if dl:
        fields["diluent"] = dl.group(0)

    return fields


def parse_ocr_text(text: str) -> list[dict]:
    # Step 1 — run NER on the full text once to find all drug name spans
    drug_spans = _extract_drug_names(text)

    lines = text.splitlines(keepends=True)
    results = []
    current_category = "GENERAL"
    current_page = None
    char_offset = 0

    for raw_line in lines:
        line = raw_line.strip()
        line_start = char_offset
        line_end = char_offset + len(raw_line)
        char_offset = line_end

        if not line:
            continue

        if _PATTERNS["page_marker"].match(line):
            current_page = re.search(r"page_\d+", line).group(0)
            continue

        # Skip patient metadata / admin lines
        if _PATTERNS["skip_line"].search(line):
            continue

        # Only treat as category header if NER found no drug on this line
        line_has_drug = any(s >= line_start and e <= line_end for s, e, _ in drug_spans)
        if _PATTERNS["category_header"].match(line) and not line_has_drug:
            current_category = line
            continue

        # Assign drug name if any NER span falls within this line
        drug_name = None
        for start, end, name in drug_spans:
            if start >= line_start and end <= line_end:
                drug_name = name.title()
                break

        fields = _extract_fields(line)

        # Only record if there is at least one clinical field alongside the drug name
        if drug_name and fields:
            fields["drug_name"] = drug_name
            fields["category"] = current_category
            fields["page"] = current_page
            results.append(fields)

    return _merge_entries(results)


def _merge_entries(entries: list[dict]) -> list[dict]:
    merged = []
    current = None
    for entry in entries:
        if "drug_name" in entry:
            if current:
                merged.append(current)
            current = entry.copy()
        elif current:
            for key, val in entry.items():
                if key not in ("category", "page", "drug_name") and key not in current:
                    current[key] = val
    if current:
        merged.append(current)
    return merged
