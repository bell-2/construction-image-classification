import json
import io
import csv
import zipfile
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image
from classifier import load_model, classify_uploaded_images

CATEGORIES_FILE = Path(__file__).parent / "categories.json"

# 모델 로드
print("🔄 CLIP 모델 로딩 중...")
model, preprocess, tokenizer, device = load_model()
print(f"✅ 모델 로드 완료 (device: {device})")


def get_categories() -> list[dict]:
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        return json.load(f)["categories"]


def classify(files):
    """이미지 분류 실행."""
    if not files:
        return None, "⚠️ 사진을 업로드해주세요.", None

    categories = get_categories()

    # Gradio는 파일 경로를 전달하므로 PIL로 열기
    pil_images = []
    filenames = []
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            fname = Path(f).name
            pil_images.append(img)
            filenames.append(fname)
        except Exception:
            continue

    if not pil_images:
        return None, "⚠️ 유효한 이미지가 없습니다.", None

    # 분류 실행
    import torch

    prompts = [cat["prompt"] for cat in categories]
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = []
    images_dict = {}
    batch_size = 16

    for i in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[i:i + batch_size]
        batch_names = filenames[i:i + batch_size]
        tensors = []

        for img, fname in zip(batch_imgs, batch_names):
            tensors.append(preprocess(img).unsqueeze(0))
            images_dict[fname] = img

        image_batch = torch.cat(tensors).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).softmax(dim=-1)

        for j, fname in enumerate(batch_names):
            scores = similarities[j].cpu().tolist()
            top_idx = scores.index(max(scores))
            results.append({
                "filename": fname,
                "category_id": categories[top_idx]["id"],
                "category_label": categories[top_idx]["label"],
                "confidence": scores[top_idx],
                "all_scores": {cat["label"]: round(s, 4) for cat, s in zip(categories, scores)},
            })

    # 카테고리별 갤러리 구성
    category_counts = {}
    for r in results:
        label = r["category_label"]
        category_counts[label] = category_counts.get(label, 0) + 1

    # 요약 텍스트
    summary_parts = []
    for label, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        summary_parts.append(f"**{label}**: {count}장")
    summary = f"### 📊 총 {len(results)}장 분류 완료\n\n" + " · ".join(summary_parts)

    # 갤러리 데이터: (이미지, 캡션) 튜플
    gallery_data = []
    for r in sorted(results, key=lambda x: x["category_label"]):
        img = images_dict[r["filename"]]
        caption = f"[{r['category_label']}] {r['filename']} ({r['confidence']:.0%})"
        gallery_data.append((img, caption))

    # ZIP 생성
    zip_path = _create_zip(results, images_dict)

    return gallery_data, summary, zip_path


def _create_zip(results, images_dict):
    """카테고리별 폴더로 나뉜 ZIP 파일 생성."""
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            folder = r["category_label"]
            fname = r["filename"]
            if fname in images_dict:
                img_bytes = io.BytesIO()
                images_dict[fname].save(img_bytes, format="JPEG")
                zf.writestr(f"{folder}/{fname}", img_bytes.getvalue())

        # CSV 포함
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

    return tmp.name


# ── Gradio UI ──
css = """
.gradio-container {
    max-width: 1200px !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}
.gr-button-primary {
    background: #f59e0b !important;
    border: none !important;
    color: #000 !important;
    font-weight: 700 !important;
}
.gr-button-primary:hover {
    background: #fbbf24 !important;
}
"""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Noto Sans KR"), "sans-serif"],
).set(
    body_background_fill="#0a0a0a",
    body_background_fill_dark="#0a0a0a",
    body_text_color="#fafafa",
    body_text_color_dark="#fafafa",
    block_background_fill="#1a1a1a",
    block_background_fill_dark="#1a1a1a",
    block_border_color="#2a2a2a",
    block_border_color_dark="#2a2a2a",
    input_background_fill="#141414",
    input_background_fill_dark="#141414",
    button_primary_background_fill="#f59e0b",
    button_primary_background_fill_dark="#f59e0b",
    button_primary_text_color="#000",
    button_primary_text_color_dark="#000",
)

with gr.Blocks(title="Construction View — 건설 현장 사진 분류", theme=theme, css=css) as demo:
    gr.Markdown(
        """
        # 🏗️ CONSTRUCTION VIEW
        **건설 현장 사진 AI 분류기** — CLIP ViT-B/32 모델 (로컬 실행, 무료)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📤 사진 업로드 (여러 장 가능)",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
            classify_btn = gr.Button("🔍 분류 시작", variant="primary", size="lg")

            summary_output = gr.Markdown(value="사진을 업로드하고 분류를 시작하세요.")
            zip_output = gr.File(label="📦 카테고리별 ZIP 다운로드", visible=False)

        with gr.Column(scale=2):
            gallery_output = gr.Gallery(
                label="분류 결과",
                columns=4,
                height="auto",
                object_fit="cover",
                show_label=True,
            )

    def run_classify(files):
        if not files:
            return None, "⚠️ 사진을 업로드해주세요.", gr.update(visible=False)
        gallery, summary, zip_path = classify(files)
        return gallery, summary, gr.update(value=zip_path, visible=True)

    classify_btn.click(
        fn=run_classify,
        inputs=[file_input],
        outputs=[gallery_output, summary_output, zip_output],
    )

if __name__ == "__main__":
    demo.launch(theme=theme, css=css)
