---
title: Construction View
emoji: 🏗️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: "6.10.0"
app_file: gradio_app.py
pinned: false
---

# 🏗️ Construction View — 건설 현장 사진 분류기

AI(CLIP ViT-B/32)를 사용해 건설 현장 사진을 자동으로 분류합니다.

## 기본 카테고리

- 안전 교육 / 안전모 착용 / 해체 작업 / 철근 작업
- 콘크리트 타설 / 장비 운용 / 현장 점검 / 비계 작업 / 굴착 작업

## 로컬 실행

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Gradio UI
python gradio_app.py

# 또는 FastAPI UI
uvicorn server:app --port 8000

# 또는 Streamlit UI
streamlit run app.py
```
