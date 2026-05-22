import re
from collections import defaultdict

PATTERNS = {
    "concentration": re.compile(
        r"(\d+\.?\d*)\s*(mg|micrograms|nanograms|units|milliunits)/ml", re.IGNORECASE
    ),
    "dose_range": re.compile(
        r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*(mg|micrograms|nanograms|units|milliunits|mmol|ml)"
        r"(/kg)?(/hr|/min)?",
        re.IGNORECASE,
    ),
    "single_dose": re.compile(
        r"(\d+\.?\d*)\s*(mg|micrograms|nanograms|units|milliunits|mmol|ml)"
        r"(/kg)?(/hr|/min)?",
        re.IGNORECASE,
    ),
    "max_dose": re.compile(r"MAX\s*(\d+\.?\d*)\s*(mg|micrograms|ml|units|mmol)", re.IGNORECASE),
    "min_dose": re.compile(r"min\s*(\d+\.?\d*)\s*(mg|micrograms|ml|units)", re.IGNORECASE),
    "infusion_rate": re.compile(r"(\d+\.?\d*)\s*ml/kg/hr|(\d+\.?\d*)\s*ml/hr", re.IGNORECASE),
    "route": re.compile(r"\b(IV|IM|PERIPHERAL|CENTRAL|NEAT|SC|PO|PR|IN|IO)\b"),
    "category_header": re.compile(r"^[A-Z][A-Z\s/]{4,}$"),
    "page_marker": re.compile(r"^---\s*page_\d+\.png\s*---$"),
    "age_range": re.compile(
        r"(Under\s*\d+\s*kg|\d+\s*kg\+|\d+\s*years?\+?|Under\s*\d+\s*years?|"
        r"\d+\s*month[s]?\s*(to\s*\d+\s*(years?|months?))?)",
        re.IGNORECASE,
    ),
    "diluent": re.compile(
        r"(Glu\s*\d+%|NaCl\s*\d+\.?\d*%|Glucose\s*\d+%|dextrose)", re.IGNORECASE
    ),
}

KNOWN_DRUGS = [
    "Ketamine", "Rocuronium", "Fentanyl", "Fentany", "Atropine", "Propofol",
    "Thiopentone", "Naloxone", "Sugammadex", "Neostigmine", "Glycopyrolate",
    "Adrenaline", "Calcium Gluconate", "Sodium Bicarbonate", "Glucose",
    "Lorazepam", "Mannitol", "Amiodarone", "Morphine", "Midazolam",
    "Dinoprostone", "Dobutamine", "Noradrenaline", "Milrinone",
    "Vasopressin", "Argipressin", "Metaraminol", "Heparin", "Atracurium",
    "Paracetamol",
]
DRUG_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(d) for d in KNOWN_DRUGS) + r")\b", re.IGNORECASE
)


def extract_fields(line: str) -> dict:
    fields = {}

    drug_match = DRUG_PATTERN.search(line)
    if drug_match:
        fields["drug_name"] = drug_match.group(1).title()

    conc = PATTERNS["concentration"].search(line)
    if conc:
        fields["concentration"] = conc.group(0)

    dose_range = PATTERNS["dose_range"].search(line)
    if dose_range:
        fields["dose_range"] = dose_range.group(0)
    else:
        single = PATTERNS["single_dose"].search(line)
        if single:
            fields["dose"] = single.group(0)

    max_d = PATTERNS["max_dose"].search(line)
    if max_d:
        fields["max_dose"] = max_d.group(0)

    min_d = PATTERNS["min_dose"].search(line)
    if min_d:
        fields["min_dose"] = min_d.group(0)

    rate = PATTERNS["infusion_rate"].search(line)
    if rate:
        fields["infusion_rate"] = rate.group(0)

    route = PATTERNS["route"].search(line)
    if route:
        fields["route"] = route.group(1)

    age = PATTERNS["age_range"].search(line)
    if age:
        fields["age_range"] = age.group(0)

    diluent = PATTERNS["diluent"].search(line)
    if diluent:
        fields["diluent"] = diluent.group(0)

    return fields


def parse_ocr_text(text: str) -> list[dict]:
    results = []
    current_category = "GENERAL"
    current_page = None

    for line in [l.strip() for l in text.splitlines()]:
        if not line:
            continue
        if PATTERNS["page_marker"].match(line):
            current_page = re.search(r"page_\d+", line).group(0)
            continue
        if PATTERNS["category_header"].match(line) and not DRUG_PATTERN.search(line):
            current_category = line.strip()
            continue
        fields = extract_fields(line)
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
