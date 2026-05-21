import fitz #importing name for pymupdf 
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from jiwer import wer, cer #internally uses levenshtein distance algorithm 


# PDF PATH


pdf_path = "Paediatric-Drug-Chart.pdf"


# CREATE IMAGE FOLDER


import shutil
if os.path.exists("pages"):
    shutil.rmtree("pages")
os.makedirs("pages")


# CONVERT PDF TO IMAGES


doc = fitz.open(pdf_path)

ground_truth_pages = {}

for page_num in range(len(doc)):

    page = doc.load_page(page_num)

    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))

    image_path = f"pages/page_{page_num+1}.png"

    pix.save(image_path)

    ground_truth_pages[f"page_{page_num+1}.png"] = page.get_text().strip()

print("PDF converted into images.")


# INITIALIZE OCR


ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)


# IMAGE ENHANCEMENT


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


# OCR EXTRACTION


all_text = []
accuracy_report = []

image_files = sorted(os.listdir("pages"), key=lambda x: int(x.split('_')[1].split('.')[0]))

for img_file in image_files:

    img_path = os.path.join("pages", img_file)

    print(f"\nProcessing: {img_file}")

    enhance_image(img_path)

    result = ocr.ocr(img_path, cls=True)

    page_text = []

    if result and result[0]:

        confidences = []

        for line in result[0]:

            text = line[1][0]
            confidence = line[1][1]

            page_text.append(text)
            confidences.append(confidence)

            print(f"{text} | Confidence: {confidence:.2f}")

        avg_conf = sum(confidences) / len(confidences)
        print(f"Page Confidence (avg): {avg_conf:.2%}")

    else:
        print("No text detected on this page (blank or image-only page)")

    ocr_text = " ".join(page_text).strip()
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

with open("ocr_output_medical.txt", "w", encoding="utf-8") as f:

    f.write(final_text)

print("\nOCR COMPLETED")
print("Output saved as ocr_output_medical.txt")


# ACCURACY REPORT


if accuracy_report:

    avg_wer = sum(p["wer"] for p in accuracy_report) / len(accuracy_report)
    avg_cer = sum(p["cer"] for p in accuracy_report) / len(accuracy_report)

    print(f"\nOverall WER : {avg_wer:.2%}")
    print(f"Overall CER : {avg_cer:.2%}")
    print(f"Word Accuracy  : {(1 - avg_wer):.2%}")
    print(f"Char Accuracy  : {(1 - avg_cer):.2%}")

    with open("accuracy_report_paddle.txt", "w", encoding="utf-8") as f:

        f.write(f"{'Page':<20} {'WER':>10} {'CER':>10} {'Word Acc':>12} {'Char Acc':>12}\n")
        f.write("-" * 66 + "\n")

        for p in accuracy_report:
            f.write(f"{p['page']:<20} {p['wer']:>10.2%} {p['cer']:>10.2%} {(1-p['wer']):>12.2%} {(1-p['cer']):>12.2%}\n")

        f.write("-" * 66 + "\n")
        f.write(f"{'OVERALL':<20} {avg_wer:>10.2%} {avg_cer:>10.2%} {(1-avg_wer):>12.2%} {(1-avg_cer):>12.2%}\n")

    print("Accuracy report saved as accuracy_report_paddle.txt")