FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    open-clip-torch \
    Pillow \
    fastapi \
    uvicorn \
    python-multipart

COPY classifier.py categories.json server.py ./
COPY static/ static/

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
