import fitz
import cv2
import numpy as np
import os
import shutil
from paddleocr import PaddleOCR
import pytesseract
import logging
logging.getLogger("ppocr").setLevel(logging.WARNING)

_paddle_ocr = None


def _get_paddle():
    global _paddle_ocr
    if _paddle_ocr is None:
        _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _paddle_ocr


def enhance_image(img_path: str):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    enhanced = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    cv2.imwrite(img_path, enhanced)


def _render_pages(pdf_path: str, pages_dir: str):
    if os.path.exists(pages_dir):
        shutil.rmtree(pages_dir)
    os.makedirs(pages_dir)
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # 2.0x scale gives sharper text — better for tables with small fonts
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_path = f"{pages_dir}/page_{page_num + 1}.png"
        pix.save(img_path)
        enhance_image(img_path)
        image_paths.append((page_num + 1, img_path))
    return image_paths


def extract_text_from_pdf(pdf_path: str, engine: str = "paddle") -> str:
    pages_dir = "uploads/pages_temp"
    image_paths = _render_pages(pdf_path, pages_dir)
    all_text = []

    if engine == "tesseract":
        # PSM 6 = assume a single uniform block of text — preserves line structure
        # better than PSM 3 for dense clinical tables
        tess_config = r"--oem 3 --psm 6"
        for page_num, img_path in image_paths:
            img = cv2.imread(img_path)
            # image_to_string preserves newlines / line grouping (unlike image_to_data
            # which returns one word per row, destroying table row context)
            raw = pytesseract.image_to_string(img, config=tess_config)
            # Keep non-empty lines only
            page_lines = [l for l in raw.splitlines() if l.strip()]
            all_text.append(f"--- page_{page_num}.png ---\n" + "\n".join(page_lines))
    else:
        ocr = _get_paddle()
        for page_num, img_path in image_paths:
            result = ocr.ocr(img_path, cls=True)
            page_text = []
            if result and result[0]:
                for line in result[0]:
                    page_text.append(line[1][0])
            all_text.append(f"--- page_{page_num}.png ---\n" + "\n".join(page_text))

    shutil.rmtree(pages_dir)
    return "\n\n".join(all_text)
