import re
import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_ner_bc5cdr_md")
    return _nlp


# Noise filter — things NER wrongly labels CHEMICAL
_NOT_DRUG = re.compile(
    r"^(dob|d\.o\.b|nhs|cga|date.*|time.*|batch|ward|weight|name|no\.|number"
    r"|glu|nacl|d5w|d10w|glucose|dextrose|sodium chloride|water|wfi|saline"
    r"|micrograms?.*|nanograms?.*|units?|milliunits?|mmol|meq|mcg|mg|ml|kg|hr|min"
    r"|neat|iv|im|sc|po|io|[\d\s\.\-/]+)$",
    re.IGNORECASE,
)

# Known category keywords — tighter than just ALL-CAPS
_CATEGORY_KEYWORDS = re.compile(
    r"\b(resuscitat|sedati|paralyt|analges|antibioti|antifungal|antiviral"
    r"|anticoagul|cardiovascular|cardiac|fluid|electrolyte|respiratory"
    r"|neurolog|anticonvuls|antiepilept|miscellaneous|inotrope|vasopressor"
    r"|palliative|antidot|muscle relaxant|anaesth|opiate|opioid)\b",
    re.IGNORECASE,
)

# Structural markers
_PAGE_MARKER = re.compile(r"^---\s*(page_\d+)\.png\s*---$")

# Known section delimiters — column headers, patient info rows to skip
_SKIP_LINE = re.compile(
    r"\b(date of birth|dob|d\.o\.b|patient name|name:|nhs|hospital no"
    r"|ward:|consultant:|weight:|allergies|signature|signed|print name"
    r"|prescribed by|checked by|administered by|date:|time:|concentration"
    r"|amount in syringe|dose range|embrace no|follow odn|cardiology advice"
    r"|under \d+kg|kg\+|age\s*$|weight\s*\(kg\)|ett size)\b",
    re.IGNORECASE,
)

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
}


def _extract_drug_names(text: str) -> list[tuple[int, int, str]]:
    nlp = _get_nlp()
    doc = nlp(text[:100000])
    results = []
    for ent in doc.ents:
        if ent.label_ != "CHEMICAL":
            continue
        name = ent.text.strip()
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


def _is_category_header(line: str) -> bool:
    """
    A line is a category header if:
    - It is mostly uppercase AND
    - It contains a known medical section keyword OR is short enough to be a title
    - It does NOT look like a dosage or column header
    """
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 80:
        return False
    # Must be mostly uppercase letters
    alpha_chars = [c for c in stripped if c.isalpha()]
    if not alpha_chars:
        return False
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    if upper_ratio < 0.75:
        return False
    # Must match a known category keyword
    if _CATEGORY_KEYWORDS.search(stripped):
        return True
    # OR be a short all-caps title with no digits (e.g. "FLUIDS", "ELECTROLYTES")
    if upper_ratio == 1.0 and not any(c.isdigit() for c in stripped) and len(stripped.split()) <= 4:
        return True
    return False


def parse_ocr_text(text: str) -> list[dict]:
    # NER pass over full text
    drug_spans = _extract_drug_names(text)

    # Build line → drug name map via char offsets
    lines = text.splitlines(keepends=True)
    line_drug_map: dict[int, str] = {}
    char_offset = 0
    for idx, raw_line in enumerate(lines):
        line_start = char_offset
        line_end = char_offset + len(raw_line)
        char_offset = line_end
        for start, end, name in drug_spans:
            if start >= line_start and end <= line_end:
                line_drug_map[idx] = name.title()
                break

    results = []
    current_category = "GENERAL"
    current_page = None
    current_entry: dict | None = None

    # Track the last N category headers seen — use the most recent one
    # that appeared on the SAME PAGE as the drug, not carried over from prev page
    page_categories: dict[str, str] = {}  # page → last category seen on that page

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        # Page marker
        pm = _PAGE_MARKER.match(line)
        if pm:
            current_page = pm.group(1)
            # Reset category per page — don't bleed page1 categories into page2
            current_category = page_categories.get(current_page, "GENERAL")
            continue

        if _SKIP_LINE.search(line):
            continue

        # Category header detection (tightened)
        if _is_category_header(line) and idx not in line_drug_map:
            if current_entry:
                results.append(current_entry)
                current_entry = None
            current_category = line.strip().title()
            if current_page:
                page_categories[current_page] = current_category
            continue

        if idx in line_drug_map:
            if current_entry:
                results.append(current_entry)
            # Assign the category that was most recently seen on THIS page
            cat = page_categories.get(current_page, current_category) if current_page else current_category
            current_entry = {
                "drug_name": line_drug_map[idx],
                "category": cat,
                "page": current_page,
            }
            fields = _extract_fields(line)
            for k, v in fields.items():
                if k not in current_entry:
                    current_entry[k] = v
        else:
            fields = _extract_fields(line)
            if fields and current_entry is not None:
                for key, val in fields.items():
                    if key not in current_entry:
                        current_entry[key] = val

    if current_entry:
        results.append(current_entry)

    clinical_keys = {"concentration", "dose", "dose_range", "max_dose",
                     "min_dose", "infusion_rate", "route", "diluent", "age_range"}
    return [e for e in results if clinical_keys & e.keys()]
