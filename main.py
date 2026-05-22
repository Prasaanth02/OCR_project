import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from app.ocr import extract_text_from_pdf
from app.parser import parse_ocr_text

app = FastAPI(title="Paediatric Drug Chart Extractor")
templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # save uploaded file temporarily
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.pdf")
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        # run OCR
        ocr_text = extract_text_from_pdf(temp_path)

        # run parser
        drugs = parse_ocr_text(ocr_text)

        # summary stats
        categories = list({d.get("category", "GENERAL") for d in drugs})
        summary = {
            "total_drugs": len(drugs),
            "unique_drug_names": len({d.get("drug_name", "") for d in drugs if "drug_name" in d}),
            "categories": categories,
            "pages": len({d.get("page") for d in drugs if d.get("page")}),
        }

        return JSONResponse({"summary": summary, "drugs": drugs})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
