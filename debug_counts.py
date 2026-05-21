import json

GROUND_TRUTH = set([
    "adrenaline", "amiodarone", "argipressin", "atracurium", "atropine",
    "calcium gluconate", "dinoprostone", "dobutamine", "fentanyl", "glucose",
    "glycopyrolate", "heparin", "ketamine", "lorazepam", "mannitol",
    "metaraminol", "midazolam", "milrinone", "morphine", "naloxone",
    "neostigmine", "noradrenaline", "paracetamol", "propofol", "rocuronium",
    "sodium bicarbonate", "sugammadex", "thiopentone", "vasopressin"
])

paddle = json.load(open("structured_drugs.json"))
tess = json.load(open("structured_drugs_tesseract.json"))

paddle_found = set(e["drug_name"].strip().lower() for e in paddle)
tess_found = set(e["drug_name"].strip().lower() for e in tess)

print("GT count:", len(GROUND_TRUTH))
print("Paddle unique drugs found:", len(paddle_found), sorted(paddle_found))
print("Tess unique drugs found:", len(tess_found), sorted(tess_found))
print()
print("GT missed by Paddle:", sorted(GROUND_TRUTH - paddle_found))
print("Paddle extras (not in GT):", sorted(paddle_found - GROUND_TRUTH))
print()
print("GT missed by Tess:", sorted(GROUND_TRUTH - tess_found))
print("Tess extras (not in GT):", sorted(tess_found - GROUND_TRUTH))
