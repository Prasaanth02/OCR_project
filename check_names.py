files = {
    "parse_drug_chart.py": ["ocr_output_medical.txt", "structured_drugs_paddle.json", "ocr_output_tesseract.txt", "structured_drugs_tesseract.json"],
    "validate.py": ["structured_drugs_paddle.json", "structured_drugs_tesseract.json"],
    "debug_counts.py": ["structured_drugs_paddle.json", "structured_drugs_tesseract.json"],
    "main_paddle.py": ["ocr_output_medical.txt", "accuracy_report_paddle.txt"],
    "main_tesseract.py": ["ocr_output_tesseract.txt", "accuracy_report_tesseract.txt"],
    "ground_truth_check.py": ["Paediatric-Drug-Chart.pdf"],
}

all_ok = True
for fname, expected in files.items():
    content = open(fname).read()
    for name in expected:
        if name in content:
            print(f"OK      {fname} -> {name}")
        else:
            print(f"MISSING {fname} -> {name}")
            all_ok = False

print()
print("All filenames consistent!" if all_ok else "ISSUES FOUND - check above")
