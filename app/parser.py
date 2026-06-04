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


def _extract_drug_names(text: str) -> list[tuple[int, int, str]]:
    """Return list of (start_char, end_char, drug_name) from NER on the full text block."""
    nlp = _get_nlp()
    doc = nlp(text[:100000])  # spaCy limit guard
    return [
        (ent.start_char, ent.end_char, ent.text.strip())
        for ent in doc.ents
        if ent.label_ == "CHEMICAL"
    ]


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
    drug_char_set = set()
    for start, end, _ in drug_spans:
        drug_char_set.update(range(start, end))

    lines = text.splitlines()
    results = []
    current_category = "GENERAL"
    current_page = None
    char_offset = 0

    for raw_line in lines:
        line = raw_line.strip()
        line_start = char_offset
        char_offset += len(raw_line) + 1  # +1 for newline

        if not line:
            continue

        if _PATTERNS["page_marker"].match(line):
            current_page = re.search(r"page_\d+", line).group(0)
            continue

        if _PATTERNS["category_header"].match(line):
            current_category = line
            continue

        # Check if any NER drug span overlaps this line
        line_end = line_start + len(line)
        drug_name = None
        for start, end, name in drug_spans:
            if start >= line_start and end <= line_start + len(raw_line):
                drug_name = name.title()
                break

        fields = _extract_fields(line)

        if drug_name:
            fields["drug_name"] = drug_name

        if fields:
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
        else:
            merged.append(entry)
    if current:
        merged.append(current)
    return merged
