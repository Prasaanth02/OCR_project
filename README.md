# Paediatric Drug Chart OCR Extractor

A pipeline to extract structured drug information from paediatric drug chart PDFs using OCR (PaddleOCR and Tesseract), with accuracy benchmarking and a FastAPI web interface.

---

## Project Structure

```
OCR/
├── app/
│   ├── templates/index.html      # Web UI (drag-and-drop PDF upload)
│   ├── ocr.py                    # PaddleOCR extraction module (used by FastAPI)
│   └── parser.py                 # Regex-based drug field parser (used by FastAPI)
├── main.py                       # FastAPI web application entry point
├── main_paddle.py                # Standalone PaddleOCR pipeline with accuracy metrics
├── main_tesseract.py             # Standalone Tesseract pipeline with accuracy metrics
├── parse_drug_chart.py           # Parses saved OCR output .txt files → structured JSON
├── validate.py                   # Compares PaddleOCR vs Tesseract JSON outputs
├── requirements.txt              # Python dependencies
├── Paediatric-Drug-Chart.pdf     # Input PDF
├── ocr_output_medical.txt        # PaddleOCR raw text output
├── ocr_output_tesseract.txt      # Tesseract raw text output
├── structured_drugs_paddle.json  # Structured drug data from PaddleOCR
├── structured_drugs_tesseract.json
├── accuracy_report_medical.txt   # WER/CER report for PaddleOCR
├── accuracy_report_tesseract.txt # WER/CER report for Tesseract
└── validation_report.txt         # Head-to-head comparison report
```

---

## Pipeline Overview

### Pipeline A — Standalone Benchmarking (PaddleOCR vs Tesseract)

```
PDF
 │
 ├─ fitz (PyMuPDF) → renders each page as PNG (1.5x scale)
 │
 ├─ Image Enhancement (OpenCV)
 │    └─ Grayscale → FastNlMeansDenoising → AdaptiveThreshold (Gaussian, 31×31, C=10)
 │
 ├─ OCR Extraction
 │    ├─ PaddleOCR  (use_angle_cls=True, lang=en)  → main_paddle.py
 │    └─ Tesseract  (--oem 3 --psm 3)              → main_tesseract.py
 │
 ├─ Accuracy Evaluation (jiwer)
 │    ├─ Ground truth = PyMuPDF embedded text (page.get_text())
 │    └─ Reports WER, CER, Word Accuracy, Char Accuracy per page + overall
 │         → accuracy_report_medical.txt
 │         → accuracy_report_tesseract.txt
 │
 ├─ Raw text saved
 │    ├─ ocr_output_medical.txt
 │    └─ ocr_output_tesseract.txt
 │
 ├─ Structured Parsing  (parse_drug_chart.py)
 │    ├─ Detects page markers, category headers, drug names (30-drug dictionary)
 │    ├─ Regex extraction: concentration, dose/dose_range, max_dose, min_dose,
 │    │                     infusion_rate, route, age_range, diluent
 │    └─ Merges multi-line entries belonging to the same drug
 │         → structured_drugs_paddle.json
 │         → structured_drugs_tesseract.json
 │
 └─ Validation  (validate.py)
      ├─ Drug coverage vs 29 ground-truth drugs
      ├─ Field richness per drug (avg fields populated)
      ├─ Value mismatch report (fields present in both but differ)
      ├─ Drugs with name only (no fields extracted)
      └─ Overall summary + recommended engine
           → validation_report.txt
```

### Pipeline B — FastAPI Web Application

```
Browser (drag-and-drop PDF + engine toggle: PaddleOCR | Tesseract)
 │
 └─ POST /upload  (main.py)  — form fields: file, engine
      │
      ├─ Validate engine ∈ { "paddle", "tesseract" }
      ├─ Save PDF to uploads/ (UUID filename, auto-deleted after processing)
      │
      ├─ app/ocr.py  — extract_text_from_pdf(pdf_path, engine)
      │    ├─ fitz renders pages → uploads/pages_temp/
      │    ├─ OpenCV enhancement (same as Pipeline A)
      │    ├─ engine == "paddle"     → PaddleOCR (lazy-loaded on first use)
      │    ├─ engine == "tesseract"  → pytesseract (--oem 3 --psm 3)
      │    └─ pages_temp/ cleaned up after extraction
      │
      ├─ app/parser.py  — parse_ocr_text()
      │    ├─ Same regex patterns as parse_drug_chart.py
      │    └─ Returns list of structured drug dicts
      │
      └─ JSON response
           ├─ summary: { total_drugs, unique_drug_names, categories, pages, engine }
           └─ drugs: [ { drug_name, category, page, concentration, dose,
                          dose_range, max_dose, min_dose, infusion_rate,
                          route, diluent, age_range }, ... ]
```

### Web UI Engine Selector

The upload card contains a **PaddleOCR / Tesseract toggle** (pill-style radio buttons):

- Switching engines updates a hint below the toggle describing trade-offs
- Selected engine is sent as a form field (`engine`) alongside the PDF
- After results load, a coloured badge next to the results heading shows the engine used
  - Blue badge → PaddleOCR
  - Yellow badge → Tesseract

---

## Extracted Fields

| Field | Example |
|---|---|
| `drug_name` | `Morphine` |
| `category` | `RESUSCITATION DRUGS` |
| `page` | `page_1` |
| `concentration` | `1mg/ml` |
| `dose` | `0.1mg/kg` |
| `dose_range` | `0.05–0.1mg/kg` |
| `max_dose` | `MAX 10mg` |
| `min_dose` | `min 2.5mg` |
| `infusion_rate` | `0.5ml/kg/hr` |
| `route` | `IV` |
| `diluent` | `NaCl 0.9%` |
| `age_range` | `Under 10 kg` |

---

## Installation

```bash
pip install -r requirements.txt
```

> Tesseract binary must be installed separately and available on PATH:
> - Windows: https://github.com/UB-Mannheim/tesseract/wiki
> - Linux: `sudo apt install tesseract-ocr`

---

## Usage

### Run PaddleOCR benchmark pipeline

```bash
python main_paddle.py
```

Outputs: `ocr_output_medical.txt`, `accuracy_report_medical.txt`

### Run Tesseract benchmark pipeline

```bash
python main_tesseract.py
```

Outputs: `ocr_output_tesseract.txt`, `accuracy_report_tesseract.txt`

### Parse OCR output to structured JSON

```bash
python parse_drug_chart.py
```

Outputs: `structured_drugs_paddle.json`, `structured_drugs_tesseract.json`

### Validate and compare both engines

```bash
python validate.py
```

Outputs: `validation_report.txt`

### Run the web application

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser, upload a PDF, and get structured drug data instantly.

---

## Dependencies

| Package | Purpose |
|---|---|
| `paddleocr` + `paddlepaddle` | Primary OCR engine |
| `pytesseract` | Secondary OCR engine (benchmarking) |
| `pymupdf` (fitz) | PDF → image conversion, ground truth text |
| `opencv-python` | Image enhancement (denoise + threshold) |
| `numpy` | Array operations for image processing |
| `jiwer` | WER / CER accuracy metrics |
| `fastapi` + `uvicorn` | Web API server |
| `python-multipart` | File upload handling |
| `jinja2` | HTML templating |
