# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

건설 현장 사진 자동 분류 웹 앱. OpenAI CLIP 모델(로컬 실행, 무료)을 사용해 건설 사진을 카테고리별로 분류한다. 파일 업로드와 로컬 폴더 경로 두 가지 입력 방식을 지원.

## Commands

```bash
# 가상환경 활성화
source venv/bin/activate

# 앱 실행
streamlit run app.py

# 의존성 설치
pip install -r requirements.txt
```

## Architecture

- **app.py** — Streamlit 메인 UI. 파일 업로드 또는 로컬 폴더 경로로 이미지 입력. 분류 결과를 카테고리별 탭+썸네일 그리드로 표시. 사이드바에서 카테고리 추가/삭제 가능. CSV 다운로드 지원.
- **classifier.py** — CLIP 기반 분류 엔진. `open-clip-torch`의 ViT-B-32 모델 사용. 이미지를 배치(16개씩) 처리하고, 각 이미지와 카테고리 영문 프롬프트 간 코사인 유사도(softmax)로 분류. `classify_images()`는 로컬 파일 경로용, `classify_uploaded_images()`는 Streamlit UploadedFile용.
- **categories.json** — 분류 카테고리 정의. 각 항목에 `id`, `label`(한글), `prompt`(영문 CLIP 프롬프트) 포함. 앱에서 편집 시 이 파일에 직접 반영됨.

## Key Design Decisions

- CLIP 프롬프트는 **영문**으로 작성해야 정확도가 높음 (CLIP이 영어 텍스트로 학습됨)
- 디바이스 자동 감지: MPS (Apple Silicon) > CUDA > CPU
- 첫 실행 시 모델 ~340MB 자동 다운로드, 이후 캐시됨
- `@st.cache_resource`로 모델을 한 번만 로드
