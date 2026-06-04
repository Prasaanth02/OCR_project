# Paediatric Drug Chart OCR Extractor

A pipeline to extract structured drug information from any drug chart PDF using OCR (PaddleOCR or Tesseract) combined with biomedical NER (scispaCy `en_ner_bc5cdr_md`) for drug name detection — no hardcoded drug lists.

---

## Project Structure

```
OCR/
├── app/
│   ├── templates/index.html      # Web UI (drag-and-drop PDF upload + engine toggle)
│   ├── ocr.py                    # PDF → image → OCR text (PaddleOCR or Tesseract)
│   └── parser.py                 # scispaCy NER + regex clinical field extraction
├── main.py                       # FastAPI web application entry point
├── requirements.txt              # Python dependencies
└── Paediatric-Drug-Chart.pdf     # Example input PDF
```

---

## Pipeline

```
Browser (drag-and-drop PDF + engine toggle: PaddleOCR | Tesseract)
 │
 └─ POST /upload  (main.py)  — form fields: file, engine
      │
      ├─ Validate engine ∈ { "paddle", "tesseract" }
      ├─ Save PDF to uploads/ (UUID filename, auto-deleted after processing)
      │
      ├─ app/ocr.py  — extract_text_from_pdf(pdf_path, engine)
      │    ├─ fitz (PyMuPDF) renders each page as PNG (1.5x scale)
      │    ├─ OpenCV enhancement per page
      │    │    └─ Grayscale → FastNlMeansDenoising → AdaptiveThreshold (Gaussian, 31×31, C=10)
      │    ├─ engine == "paddle"    → PaddleOCR (lazy-loaded, use_angle_cls=True, lang=en)
      │    ├─ engine == "tesseract" → pytesseract (--oem 3 --psm 3, confidence > 0 filter)
      │    └─ uploads/pages_temp/ cleaned up after extraction
      │
      ├─ app/parser.py  — parse_ocr_text(text)
      │    │
      │    ├─ NER pass — scispaCy en_ner_bc5cdr_md over full OCR text
      │    │    └─ Extracts CHEMICAL entities → drug names
      │    │         No hardcoded drug list — works on any drug chart
      │    │
      │    ├─ Line-by-line pass
      │    │    ├─ Page markers  → track current page
      │    │    ├─ ALL-CAPS lines → track current category
      │    │    ├─ NER span overlap → assign drug_name to line
      │    │    └─ Regex extraction of clinical fields (broad patterns, not chart-specific)
      │    │         concentration  — e.g. 1mg/ml, 500mcg/ml, 10units/L
      │    │         dose_range     — e.g. 0.05–0.1mg/kg
      │    │         dose           — e.g. 0.1mg/kg/hr
      │    │         max_dose       — e.g. MAX 10mg, maximum 500mcg
      │    │         min_dose       — e.g. min 2.5mg
      │    │         infusion_rate  — e.g. 0.5ml/kg/hr, 10ml/hr
      │    │         route          — IV, IM, SC, oral, intranasal, intraosseous, etc.
      │    │         diluent        — NaCl, glucose, D5W, D10W, Hartmann's, WFI, etc.
      │    │         age_range      — Under 10kg, 1 month to 2 years, 12 years+, etc.
      │    │
      │    └─ Merge consecutive lines belonging to the same drug entry
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

## Why scispaCy instead of a fixed drug list

| Approach | Generalisation | Maintenance |
|---|---|---|
| Hardcoded `KNOWN_DRUGS` list | Only works for those exact 30 drugs | Must be updated for every new chart |
| scispaCy `en_ner_bc5cdr_md` | Recognises any drug/chemical name it was trained on (BC5CDR corpus — thousands of biomedical entities) | No maintenance, works across charts |

The model is pretrained on the **BC5CDR biomedical corpus** and recognises `CHEMICAL` (drugs, compounds) and `DISEASE` entities. Only `CHEMICAL` entities are used as drug names here.

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

# Install the scispaCy biomedical NER model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

> Tesseract binary must be installed separately and available on PATH:
> - Windows: https://github.com/UB-Mannheim/tesseract/wiki
> - Linux: `sudo apt install tesseract-ocr`

---

## Usage

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000`, upload any drug chart PDF, select PaddleOCR or Tesseract, and get structured drug data instantly.

---

## Dependencies

| Package | Purpose |
|---|---|
| `paddleocr` + `paddlepaddle` | Primary OCR engine |
| `pytesseract` | Secondary OCR engine |
| `pymupdf` (fitz) | PDF → image conversion |
| `opencv-python` | Image enhancement (denoise + threshold) |
| `numpy` | Array operations for image processing |
| `scispacy` + `en_ner_bc5cdr_md` | Biomedical NER — drug name extraction |
| `fastapi` + `uvicorn` | Web API server |
| `python-multipart` | File upload handling |
| `jinja2` | HTML templating |
