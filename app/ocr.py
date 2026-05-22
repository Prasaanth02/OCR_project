import fitz
import cv2
import numpy as np
import os
import shutil
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="en")


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


def extract_text_from_pdf(pdf_path: str) -> str:
    pages_dir = "uploads/pages_temp"
    if os.path.exists(pages_dir):
        shutil.rmtree(pages_dir)
    os.makedirs(pages_dir)

    doc = fitz.open(pdf_path)
    all_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_path = f"{pages_dir}/page_{page_num + 1}.png"
        pix.save(img_path)

        enhance_image(img_path)
        result = ocr.ocr(img_path, cls=True)

        page_text = []
        if result and result[0]:
            for line in result[0]:
                page_text.append(line[1][0])

        all_text.append(f"--- page_{page_num + 1}.png ---\n" + "\n".join(page_text))

    shutil.rmtree(pages_dir)
    return "\n\n".join(all_text)
