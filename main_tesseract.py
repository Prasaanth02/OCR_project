import fitz  # pymupdf
import os
import cv2
import numpy as np
import pytesseract
from jiwer import wer, cer


# PDF PATH

pdf_path = "Paediatric-Drug-Chart.pdf"


# CREATE IMAGE FOLDER (separate folder to avoid conflict with paddle images)

import shutil
if os.path.exists("pages_tesseract"):
    shutil.rmtree("pages_tesseract")
os.makedirs("pages_tesseract")


# CONVERT PDF TO IMAGES

doc = fitz.open(pdf_path)

ground_truth_pages = {}

for page_num in range(len(doc)):

    page = doc.load_page(page_num)

    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))

    image_path = f"pages_tesseract/page_{page_num+1}.png"

    pix.save(image_path)

    ground_truth_pages[f"page_{page_num+1}.png"] = page.get_text().strip()

print("PDF converted into images.")


# IMAGE ENHANCEMENT (same pipeline as PaddleOCR for fair comparison)

def enhance_image(img_path):

    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    enhanced = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    cv2.imwrite(img_path, enhanced)


# TESSERACT CONFIG
# PSM 3 = fully automatic page segmentation 
# OEM 3 = default OCR engine 

TESS_CONFIG = r"--oem 3 --psm 3"


# OCR EXTRACTION

all_text = []
accuracy_report = []

image_files = sorted(
    os.listdir("pages_tesseract"),
    key=lambda x: int(x.split('_')[1].split('.')[0])
)

for img_file in image_files:

    img_path = os.path.join("pages_tesseract", img_file)

    print(f"\nProcessing: {img_file}")

    enhance_image(img_path)

    # Get text + confidence data via image_to_data
    img = cv2.imread(img_path)

    data = pytesseract.image_to_data(
        img, config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )

    page_text = []
    confidences = []

    for i, conf in enumerate(data["conf"]):
        # conf == -1 means non-text element; filter low-confidence noise
        if int(conf) > 0:
            word = data["text"][i].strip()
            if word:
                page_text.append(word)
                confidences.append(int(conf))

    if page_text:
        avg_conf = sum(confidences) / len(confidences)
        ocr_text = " ".join(page_text).strip()
        print(f"Extracted {len(page_text)} words | Avg Confidence: {avg_conf:.2f}%")
    else:
        ocr_text = ""
        print("No text detected on this page (blank or image-only page)")

    gt_text = ground_truth_pages.get(img_file, "").strip()

    if ocr_text and gt_text:

        page_wer = wer(gt_text, ocr_text)
        page_cer = cer(gt_text, ocr_text)

        print(f"WER: {page_wer:.2%} | CER: {page_cer:.2%}")

        accuracy_report.append({
            "page": img_file,
            "wer": page_wer,
            "cer": page_cer
        })

    else:
        print("Skipping WER/CER (no OCR text or no ground truth for this page)")

    all_text.append(f"--- {img_file} ---\n" + "\n".join(page_text))


# SAVE OUTPUT

final_text = "\n\n".join(all_text)

with open("ocr_output_tesseract.txt", "w", encoding="utf-8") as f:
    f.write(final_text)

print("\nOCR COMPLETED")
print("Output saved as ocr_output_tesseract.txt")


# ACCURACY REPORT

if accuracy_report:

    avg_wer = sum(p["wer"] for p in accuracy_report) / len(accuracy_report)
    avg_cer = sum(p["cer"] for p in accuracy_report) / len(accuracy_report)

    print(f"\nOverall WER : {avg_wer:.2%}")
    print(f"Overall CER : {avg_cer:.2%}")
    print(f"Word Accuracy  : {(1 - avg_wer):.2%}")
    print(f"Char Accuracy  : {(1 - avg_cer):.2%}")

    with open("accuracy_report_tesseract.txt", "w", encoding="utf-8") as f:

        f.write(f"{'Page':<20} {'WER':>10} {'CER':>10} {'Word Acc':>12} {'Char Acc':>12}\n")
        f.write("-" * 66 + "\n")

        for p in accuracy_report:
            f.write(
                f"{p['page']:<20} {p['wer']:>10.2%} {p['cer']:>10.2%} "
                f"{(1-p['wer']):>12.2%} {(1-p['cer']):>12.2%}\n"
            )

        f.write("-" * 66 + "\n")
        f.write(
            f"{'OVERALL':<20} {avg_wer:>10.2%} {avg_cer:>10.2%} "
            f"{(1-avg_wer):>12.2%} {(1-avg_cer):>12.2%}\n"
        )

    print("Accuracy report saved as accuracy_report_tesseract.txt")
