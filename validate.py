import json
from collections import defaultdict

PADDLE_FILE = "structured_drugs_paddle.json"
TESS_FILE = "structured_drugs_tesseract.json"
REPORT_FILE = "validation_report.txt"
GROUND_TRUTH_FILE = "Paediatric-Drug-Chart.pdf"

GROUND_TRUTH_DRUGS = [
    "adrenaline", "amiodarone", "argipressin", "atracurium", "atropine",
    "calcium gluconate", "dinoprostone", "dobutamine", "fentanyl", "glucose",
    "glycopyrolate", "heparin", "ketamine", "lorazepam", "mannitol",
    "metaraminol", "midazolam", "milrinone", "morphine", "naloxone",
    "neostigmine", "noradrenaline", "paracetamol", "propofol", "rocuronium",
    "sodium bicarbonate", "sugammadex", "thiopentone", "vasopressin"
]

COMPARE_FIELDS = ["concentration", "dose", "dose_range", "max_dose", "min_dose",
                  "infusion_rate", "route", "diluent", "age_range"]


def load(filepath):
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def group_by_drug(entries):
    grouped = defaultdict(list)
    for e in entries:
        grouped[e["drug_name"].strip().lower()].append(e)
    return grouped


def field_richness(entry):
    return sum(1 for f in COMPARE_FIELDS if f in entry)


def coverage_report(paddle_grouped, tess_grouped, all_drugs):
    lines = ["=" * 60, "1. DRUG COVERAGE vs GROUND TRUTH (29 drugs)", "=" * 60]
    gt = set(GROUND_TRUTH_DRUGS)
    paddle_found = set(paddle_grouped)
    tess_found = set(tess_grouped)

    p_hit = paddle_found & gt
    t_hit = tess_found & gt
    p_missed = gt - paddle_found
    t_missed = gt - tess_found
    p_extras = paddle_found - gt   # OCR typos / non-GT entries
    t_extras = tess_found - gt
    only_paddle = p_hit - tess_found
    only_tess = t_hit - paddle_found
    both = p_hit & t_hit

    lines.append(f"{'Metric':<45} {'Paddle':>6} {'Tesseract':>10}")
    lines.append("-" * 63)
    lines.append(f"{'Ground truth drugs':<45} {'29':>6} {'29':>10}")
    lines.append(f"{'Total drugs extracted (incl. OCR typos)':<45} {len(paddle_found):>6} {len(tess_found):>10}")
    lines.append(f"{'Matched with ground truth':<45} {len(p_hit):>6} {len(t_hit):>10}")
    lines.append(f"{'Missed from ground truth':<45} {len(p_missed):>6} {len(t_missed):>10}")
    lines.append(f"{'OCR typos / extras not in GT':<45} {len(p_extras):>6} {len(t_extras):>10}")
    lines.append(f"{'Coverage % (matched/29)':<45} {len(p_hit)/len(gt)*100:>5.1f}% {len(t_hit)/len(gt)*100:>9.1f}%")
    lines.append("")
    lines.append(f"Matched by both engines : {sorted(both)}")
    lines.append(f"Paddle only (GT match)  : {sorted(only_paddle) or 'None'}")
    lines.append(f"Tesseract only (GT match): {sorted(only_tess) or 'None'}")
    lines.append(f"Paddle missed (GT)      : {sorted(p_missed) or 'None'}")
    lines.append(f"Tesseract missed (GT)   : {sorted(t_missed) or 'None'}")
    lines.append(f"Paddle OCR typos        : {sorted(p_extras) or 'None'}")
    lines.append(f"Tesseract OCR typos     : {sorted(t_extras) or 'None'}")
    return lines


def richness_report(paddle_grouped, tess_grouped, all_drugs):
    lines = ["", "=" * 60, "2. FIELD RICHNESS (avg fields populated per drug)", "=" * 60]
    lines.append(f"{'Drug':<25} {'Paddle':>10} {'Tesseract':>12} {'Winner':>10}")
    lines.append("-" * 60)

    for drug in sorted(all_drugs):
        p_entries = paddle_grouped.get(drug, [])
        t_entries = tess_grouped.get(drug, [])

        p_avg = round(sum(field_richness(e) for e in p_entries) / len(p_entries), 1) if p_entries else 0
        t_avg = round(sum(field_richness(e) for e in t_entries) / len(t_entries), 1) if t_entries else 0

        # if one engine didn't find the drug at all, the other wins outright
        if p_entries and not t_entries:
            winner = "Paddle"
        elif t_entries and not p_entries:
            winner = "Tesseract"
        elif p_avg > t_avg:
            winner = "Paddle"
        elif t_avg > p_avg:
            winner = "Tesseract"
        else:
            # both found drug but 0 fields each — check ground truth coverage
            winner = "Tie (no fields)"
        lines.append(f"{drug:<25} {p_avg:>10} {t_avg:>12} {winner:>10}")

    return lines


def mismatch_report(paddle_grouped, tess_grouped, all_drugs):
    lines = ["", "=" * 60, "3. VALUE MISMATCHES (fields present in both but differ)", "=" * 60]

    mismatches_found = False
    for drug in sorted(all_drugs):
        p_entries = paddle_grouped.get(drug, [])
        t_entries = tess_grouped.get(drug, [])
        if not p_entries or not t_entries:
            continue

        # compare first entry of each for shared fields
        p = p_entries[0]
        t = t_entries[0]

        drug_mismatches = []
        for field in COMPARE_FIELDS:
            if field in p and field in t and p[field] != t[field]:
                drug_mismatches.append(f"  {field}: Paddle='{p[field]}' | Tesseract='{t[field]}'")

        if drug_mismatches:
            mismatches_found = True
            lines.append(f"\n{drug.title()}:")
            lines.extend(drug_mismatches)

    if not mismatches_found:
        lines.append("No mismatches found for shared fields.")
    return lines


def empty_fields_report(paddle_grouped, tess_grouped, all_drugs):
    lines = ["", "=" * 60, "4. DRUGS WITH NO FIELDS EXTRACTED (drug name only)", "=" * 60]

    paddle_empty = [d for d in paddle_grouped if all(field_richness(e) == 0 for e in paddle_grouped[d])]
    tess_empty = [d for d in tess_grouped if all(field_richness(e) == 0 for e in tess_grouped[d])]

    lines.append(f"Paddle  ({len(paddle_empty)}): {sorted(paddle_empty) or 'None'}")
    lines.append(f"Tesseract ({len(tess_empty)}): {sorted(tess_empty) or 'None'}")
    return lines


def summary(paddle_grouped, tess_grouped):
    gt = set(GROUND_TRUTH_DRUGS)
    paddle_found = set(paddle_grouped)
    tess_found = set(tess_grouped)
    p_hit = paddle_found & gt
    t_hit = tess_found & gt
    p_missed = gt - paddle_found
    t_missed = gt - tess_found
    p_extras = paddle_found - gt
    t_extras = tess_found - gt
    p_total_fields = sum(field_richness(e) for entries in paddle_grouped.values() for e in entries)
    t_total_fields = sum(field_richness(e) for entries in tess_grouped.values() for e in entries)
    p_paddle_wins = sum(
        1 for d in gt
        if (paddle_grouped.get(d) and not tess_grouped.get(d)) or
           (paddle_grouped.get(d) and tess_grouped.get(d) and
            sum(field_richness(e) for e in paddle_grouped[d]) / len(paddle_grouped[d]) >
            sum(field_richness(e) for e in tess_grouped[d]) / len(tess_grouped[d]))
    )

    lines = ["", "=" * 60, "5. OVERALL SUMMARY", "=" * 60]
    lines.append(f"{'Metric':<45} {'Paddle':>6} {'Tesseract':>10}")
    lines.append("-" * 63)
    lines.append(f"{'Ground truth drugs':<45} {'29':>6} {'29':>10}")
    lines.append(f"{'Total drugs extracted (incl. OCR typos)':<45} {len(paddle_found):>6} {len(tess_found):>10}")
    lines.append(f"{'Matched with ground truth':<45} {len(p_hit):>6} {len(t_hit):>10}")
    lines.append(f"{'Missed from ground truth':<45} {len(p_missed):>6} {len(t_missed):>10}")
    lines.append(f"{'OCR typos / extras not in GT':<45} {len(p_extras):>6} {len(t_extras):>10}")
    lines.append(f"{'Coverage % (matched/29)':<45} {len(p_hit)/len(gt)*100:>5.1f}% {len(t_hit)/len(gt)*100:>9.1f}%")
    lines.append(f"{'Total entries extracted':<45} {sum(len(v) for v in paddle_grouped.values()):>6} {sum(len(v) for v in tess_grouped.values()):>10}")
    lines.append(f"{'Total fields extracted':<45} {p_total_fields:>6} {t_total_fields:>10}")
    lines.append(f"{'Drugs where Paddle wins (GT)':<45} {p_paddle_wins:>6}")
    lines.append(f"{'Recommended engine':<45} {'Paddle' if p_total_fields >= t_total_fields else 'Tesseract':>6}")
    return lines


if __name__ == "__main__":
    paddle_data = load(PADDLE_FILE)
    tess_data = load(TESS_FILE)

    paddle_grouped = group_by_drug(paddle_data)
    tess_grouped = group_by_drug(tess_data)
    all_drugs = set(paddle_grouped) | set(tess_grouped)

    report_lines = (
        coverage_report(paddle_grouped, tess_grouped, all_drugs)
        + richness_report(paddle_grouped, tess_grouped, all_drugs)
        + mismatch_report(paddle_grouped, tess_grouped, all_drugs)
        + empty_fields_report(paddle_grouped, tess_grouped, all_drugs)
        + summary(paddle_grouped, tess_grouped)
    )

    report = "\n".join(report_lines)
    print(report)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nValidation report saved -> {REPORT_FILE}")
