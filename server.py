import json
import io
import csv
import zipfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from classifier import load_model, classify_uploaded_images

app = FastAPI()

CATEGORIES_FILE = Path(__file__).parent / "categories.json"
STATIC_DIR = Path(__file__).parent / "static"

# 모델은 서버 시작 시 로드
print("🔄 CLIP 모델 로딩 중... (첫 실행 시 ~340MB 다운로드)")
model, preprocess, tokenizer, device = load_model()
print(f"✅ 모델 로드 완료 (device: {device})")


def get_categories() -> list[dict]:
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        return json.load(f)["categories"]


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/categories")
async def api_categories():
    return get_categories()


@app.post("/api/classify")
async def api_classify(files: list[UploadFile] = File(...)):
    categories = get_categories()
    results, _ = classify_uploaded_images(
        files, categories, model, preprocess, tokenizer, device
    )
    return results


@app.post("/api/download-zip")
async def api_download_zip(files: list[UploadFile] = File(...)):
    categories = get_categories()
    results, images_dict = classify_uploaded_images(
        files, categories, model, preprocess, tokenizer, device
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            folder = r["category_label"]
            filename = r["filename"]
            if filename in images_dict:
                img_bytes = io.BytesIO()
                images_dict[filename].save(img_bytes, format="JPEG")
                zf.writestr(f"{folder}/{filename}", img_bytes.getvalue())

        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=["filename", "category_label", "confidence"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename": r["filename"],
                "category_label": r["category_label"],
                "confidence": f"{r['confidence']:.2%}",
            })
        zf.writestr("분류결과.csv", csv_buf.getvalue().encode("utf-8-sig"))

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=classified_photos.zip"},
    )
