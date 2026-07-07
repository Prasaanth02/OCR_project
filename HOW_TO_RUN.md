# How to Run — Paediatric Drug Chart OCR Extractor

## Requirements

- Python 3.10 (all dependencies are installed here)
- Tesseract binary (only needed if using the Tesseract engine)

---

## Step 1 — Navigate to the project folder

```bash
cd C:\Users\prasa\Desktop\OCR
```

---

## Step 2 — Install dependencies (first time only)

```bash
py -3.10 -m pip install -r requirements.txt
```

Install the scispaCy NER model (first time only):

```bash
py -3.10 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

Install Tesseract binary (only if you want to use the Tesseract engine):
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- After installing, make sure it is added to PATH

---

## Step 3 — Kill any existing server on port 8000

```bash
for /f "tokens=5" %a in ('netstat -ano ^| findstr :8000') do taskkill /F /PID %a
```

---

## Step 4 — Start the server

```bash
py -3.10 -m uvicorn main:app --reload
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

---

## Step 5 — Open the web app

Open your browser and go to:

```
http://localhost:8000
```

---

## Step 6 — Upload a PDF

1. Click **Choose PDF File** or drag and drop a drug chart PDF
2. Select the OCR engine — **PaddleOCR** (more accurate) or **Tesseract** (faster)
3. Click **Extract Drug Information**
4. Wait for processing (PaddleOCR takes ~30 seconds on first run as it loads models)
5. Results appear in a table with drug name, category, dose, route, diluent etc.
6. Use the **Filter by Category** dropdown or **Search Drug** box to filter results
7. Click **Download JSON** to save the extracted data

---

## Stopping the server

Press `CTRL + C` in the terminal.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Address already in use` on port 8000 | Run the kill command in Step 3 |
| `Can't find model 'en_ner_bc5cdr_md'` | Run the scispaCy model install in Step 2 |
| `ModuleNotFoundError` | Run `py -3.10 -m pip install -r requirements.txt` |
| Tesseract not found | Install Tesseract binary and add to PATH |
| First upload is slow | Normal — PaddleOCR downloads and loads models on first use |
