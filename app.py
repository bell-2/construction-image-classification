import json
import csv
import io
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image
from classifier import load_model, scan_folder, classify_images, classify_uploaded_images

CATEGORIES_FILE = Path(__file__).parent / "categories.json"


def load_categories() -> list[dict]:
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        return json.load(f)["categories"]


def save_categories(categories: list[dict]):
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump({"categories": categories}, f, ensure_ascii=False, indent=2)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


def results_to_zip(results: list[dict], images_dict: dict) -> bytes:
    """분류 결과를 카테고리별 폴더로 나눈 ZIP 파일 생성."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            folder = r["category_label"]
            filename = r["filename"]
            img_bytes = io.BytesIO()

            if filename in images_dict:
                images_dict[filename].save(img_bytes, format="JPEG")
            elif r.get("path") and Path(r["path"]).exists():
                img_bytes.write(Path(r["path"]).read_bytes())
            else:
                continue

            zf.writestr(f"{folder}/{filename}", img_bytes.getvalue())

        # CSV도 포함
        csv_content = results_to_csv(results)
        zf.writestr("분류결과.csv", csv_content.encode("utf-8-sig"))

    return buf.getvalue()


def results_to_csv(results: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["filename", "category_label", "confidence"])
    writer.writeheader()
    for r in results:
        writer.writerow({
            "filename": r["filename"],
            "category_label": r["category_label"],
            "confidence": f"{r['confidence']:.2%}",
        })
    return output.getvalue()


def main():
    st.set_page_config(page_title="건설 현장 사진 분류기", page_icon="🏗️", layout="wide")
    st.title("🏗️ 건설 현장 사진 분류기")
    st.caption("건설 현장 사진을 업로드하면 자동으로 분류합니다 (CLIP 모델, 무료)")

    # --- 카테고리 초기화 ---
    if "categories" not in st.session_state:
        st.session_state.categories = load_categories()
    if "results" not in st.session_state:
        st.session_state.results = None
    if "result_images" not in st.session_state:
        st.session_state.result_images = {}

    # --- 사이드바 ---
    with st.sidebar:
        st.header("⚙️ 설정")

        input_mode = st.radio("입력 방식", ["📤 파일 업로드", "📁 로컬 폴더 경로"], horizontal=True)

        if input_mode == "📤 파일 업로드":
            uploaded_files = st.file_uploader(
                "사진 업로드 (여러 장 가능)",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True,
            )
        else:
            folder_path = st.text_input("사진 폴더 경로", placeholder="/Users/username/photos")
            recursive = st.checkbox("하위 폴더 포함")
            uploaded_files = None

        classify_btn = st.button("🔍 분류 시작", type="primary", use_container_width=True)

        st.divider()

        # 카테고리 관리
        with st.expander("📋 카테고리 관리"):
            cats = st.session_state.categories

            for i, cat in enumerate(cats):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"{cat['label']} ({cat['id']})")
                with col2:
                    if cat["id"] != "other" and st.button("✕", key=f"del_{i}"):
                        cats.pop(i)
                        save_categories(cats)
                        st.rerun()

            st.markdown("**새 카테고리 추가**")
            new_label = st.text_input("한글 이름", placeholder="예: 용접 작업")
            new_prompt = st.text_input("영문 설명 (CLIP용)", placeholder="예: welding work at construction site")
            if st.button("➕ 추가") and new_label and new_prompt:
                new_id = new_label.replace(" ", "_")
                other_idx = next((i for i, c in enumerate(cats) if c["id"] == "other"), len(cats))
                cats.insert(other_idx, {"id": new_id, "label": new_label, "prompt": new_prompt})
                save_categories(cats)
                st.rerun()

            if st.button("🔄 기본값 복원"):
                st.session_state.categories = load_categories()
                st.rerun()

    # --- 분류 실행 ---
    if classify_btn:
        if input_mode == "📤 파일 업로드":
            if not uploaded_files:
                st.error("사진을 업로드해주세요.")
                return

            st.info(f"📸 {len(uploaded_files)}개 이미지 분류 중...")

            with st.spinner("모델 로딩 중... (첫 실행 시 ~340MB 다운로드)"):
                model, preprocess, tokenizer, device = get_model()

            progress_bar = st.progress(0, text="분류 진행 중...")

            def update_progress(current, total):
                progress_bar.progress(current / total, text=f"분류 진행 중... ({current}/{total})")

            results, images_dict = classify_uploaded_images(
                uploaded_files,
                st.session_state.categories,
                model, preprocess, tokenizer, device,
                progress_callback=update_progress,
            )

            st.session_state.results = results
            st.session_state.result_images = images_dict
            progress_bar.empty()
            st.success(f"✅ {len(results)}개 이미지 분류 완료!")

        else:
            if not folder_path:
                st.error("폴더 경로를 입력해주세요.")
                return

            images = scan_folder(folder_path, recursive)
            if not images:
                st.error(f"'{folder_path}'에서 이미지를 찾을 수 없습니다.")
                return

            st.info(f"📸 {len(images)}개 이미지 발견. 분류 중...")

            with st.spinner("모델 로딩 중... (첫 실행 시 ~340MB 다운로드)"):
                model, preprocess, tokenizer, device = get_model()

            progress_bar = st.progress(0, text="분류 진행 중...")

            def update_progress(current, total):
                progress_bar.progress(current / total, text=f"분류 진행 중... ({current}/{total})")

            results = classify_images(
                images,
                st.session_state.categories,
                model, preprocess, tokenizer, device,
                progress_callback=update_progress,
            )

            st.session_state.results = results
            st.session_state.result_images = {}
            progress_bar.empty()
            st.success(f"✅ {len(results)}개 이미지 분류 완료!")

    # --- 결과 표시 ---
    results = st.session_state.results
    if not results:
        return

    # 요약 통계
    category_counts = {}
    for r in results:
        label = r["category_label"]
        category_counts[label] = category_counts.get(label, 0) + 1

    st.subheader("📊 분류 결과 요약")
    cols = st.columns(min(len(category_counts), 5))
    for i, (label, count) in enumerate(sorted(category_counts.items(), key=lambda x: -x[1])):
        with cols[i % len(cols)]:
            st.metric(label, f"{count}장")

    # 다운로드 버튼
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "📦 카테고리별 ZIP 다운로드",
            results_to_zip(results, st.session_state.result_images),
            "분류결과.zip",
            "application/zip",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            "📥 CSV만 다운로드",
            results_to_csv(results),
            "분류결과.csv",
            "text/csv",
            use_container_width=True,
        )

    # 카테고리별 탭
    active_categories = [cat for cat in st.session_state.categories if cat["label"] in category_counts]
    if not active_categories:
        return

    tabs = st.tabs([f"{cat['label']} ({category_counts.get(cat['label'], 0)})" for cat in active_categories])
    images_dict = st.session_state.result_images

    for tab, cat in zip(tabs, active_categories):
        with tab:
            cat_results = [r for r in results if r["category_id"] == cat["id"]]
            cat_results.sort(key=lambda x: -x["confidence"])

            grid_cols = st.columns(4)
            for idx, r in enumerate(cat_results):
                with grid_cols[idx % 4]:
                    # 업로드된 이미지는 메모리에서, 로컬은 경로에서
                    if r["filename"] in images_dict:
                        st.image(images_dict[r["filename"]], use_container_width=True)
                    elif r.get("path"):
                        st.image(r["path"], use_container_width=True)
                    st.caption(f"**{r['filename']}**\n신뢰도: {r['confidence']:.0%}")


if __name__ == "__main__":
    main()
