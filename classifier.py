import open_clip
import torch
from PIL import Image
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def load_model():
    """CLIP 모델을 로드합니다 (ViT-B-32, ~340MB 첫 다운로드)."""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer, device


def scan_folder(folder_path: str, recursive: bool = False) -> list[Path]:
    """폴더에서 이미지 파일을 검색합니다."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    files = []
    for f in folder.glob(pattern):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(f)
    return sorted(files)


def classify_images(
    image_paths: list[Path],
    categories: list[dict],
    model,
    preprocess,
    tokenizer,
    device: str,
    batch_size: int = 16,
    progress_callback=None,
) -> list[dict]:
    """이미지들을 카테고리별로 분류합니다."""
    prompts = [cat["prompt"] for cat in categories]
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = []
    total = len(image_paths)

    for i in range(0, total, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        valid_paths = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(preprocess(img).unsqueeze(0))
                valid_paths.append(path)
            except Exception:
                continue

        if not images:
            continue

        image_batch = torch.cat(images).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).softmax(dim=-1)

        for j, path in enumerate(valid_paths):
            scores = similarities[j].cpu().tolist()
            top_idx = scores.index(max(scores))
            top_score = scores[top_idx]

            results.append({
                "path": str(path),
                "filename": path.name,
                "category_id": categories[top_idx]["id"],
                "category_label": categories[top_idx]["label"],
                "confidence": top_score,
                "all_scores": {cat["label"]: round(s, 4) for cat, s in zip(categories, scores)},
            })

        if progress_callback:
            progress_callback(min(i + batch_size, total), total)

    return results


def classify_uploaded_images(
    uploaded_files,
    categories: list[dict],
    model,
    preprocess,
    tokenizer,
    device: str,
    batch_size: int = 16,
    progress_callback=None,
) -> tuple[list[dict], dict]:
    """업로드된 파일(Streamlit UploadedFile)을 분류합니다. (이미지 dict도 함께 반환)"""
    prompts = [cat["prompt"] for cat in categories]
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = []
    images_dict = {}
    total = len(uploaded_files)

    for i in range(0, total, batch_size):
        batch_files = uploaded_files[i : i + batch_size]
        images = []
        valid_files = []

        for f in batch_files:
            try:
                # FastAPI UploadFile은 .file, Streamlit UploadedFile은 직접 사용
                file_obj = f.file if hasattr(f, 'file') else f
                img = Image.open(file_obj).convert("RGB")
                fname = getattr(f, 'filename', None) or getattr(f, 'name', 'unknown')
                images.append(preprocess(img).unsqueeze(0))
                images_dict[fname] = img
                valid_files.append((f, fname))
            except Exception:
                continue

        if not images:
            continue

        image_batch = torch.cat(images).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).softmax(dim=-1)

        for j, (f, fname) in enumerate(valid_files):
            scores = similarities[j].cpu().tolist()
            top_idx = scores.index(max(scores))
            top_score = scores[top_idx]

            results.append({
                "path": "",
                "filename": fname,
                "category_id": categories[top_idx]["id"],
                "category_label": categories[top_idx]["label"],
                "confidence": top_score,
                "all_scores": {cat["label"]: round(s, 4) for cat, s in zip(categories, scores)},
            })

        if progress_callback:
            progress_callback(min(i + batch_size, total), total)

    return results, images_dict
