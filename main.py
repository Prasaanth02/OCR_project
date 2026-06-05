import os
import uuid
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from app.ocr import extract_text_from_pdf
from app.parser import parse_ocr_text

_executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="Paediatric Drug Chart Extractor")
templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), engine: str = Form("paddle")):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if engine not in ("paddle", "tesseract"):
        raise HTTPException(status_code=400, detail="engine must be 'paddle' or 'tesseract'.")

    temp_path = None
    try:
        contents = await file.read()
        temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.pdf")
        with open(temp_path, "wb") as f:
            f.write(contents)

        loop = asyncio.get_event_loop()
        ocr_text = await loop.run_in_executor(_executor, extract_text_from_pdf, temp_path, engine)
        drugs = await loop.run_in_executor(_executor, parse_ocr_text, ocr_text)
        drugs = [d for d in drugs if isinstance(d, dict)]

        categories = list({d.get("category") or "GENERAL" for d in drugs})
        summary = {
            "total_drugs": len(drugs),
            "unique_drug_names": len({d.get("drug_name", "") for d in drugs if d.get("drug_name")}),
            "categories": categories,
            "pages": len({d.get("page") for d in drugs if d.get("page")}),
            "engine": engine,
        }

        return JSONResponse({"summary": summary, "drugs": drugs})

    except HTTPException:
        raise
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse(status_code=500, content={"detail": tb})

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
