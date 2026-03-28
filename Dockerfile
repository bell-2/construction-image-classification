FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch \
    open-clip-torch \
    Pillow \
    fastapi \
    uvicorn \
    python-multipart

COPY classifier.py categories.json server.py ./
COPY static/ static/

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
