import fitz
from parse_drug_chart import DRUG_PATTERN

doc = fitz.open("Paediatric-Drug-Chart.pdf")
all_drugs_found = set()

for i in range(len(doc)):
    page = doc[i]
    text = page.get_text()
    print(f"--- PAGE {i+1} FULL TEXT ---")
    print(text.encode('ascii', errors='replace').decode('ascii'))
    matches = DRUG_PATTERN.findall(text)
    unique = set(m.lower() for m in matches)
    all_drugs_found.update(unique)
    print(f"\nDrugs found on page {i+1}: {sorted(unique)}\n")

print("=" * 50)
print(f"TOTAL UNIQUE DRUGS IN PDF (ground truth): {len(all_drugs_found)}")
print(sorted(all_drugs_found))
