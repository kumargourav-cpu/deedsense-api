FROM python:3.12-slim

# System deps:
# - tesseract-ocr: OCR engine
# - poppler-utils: needed for pdf2image (pdftoppm)
# - libgl1, libglib2.0-0: common PIL/OpenCV deps (safe to include)
RUN apt-get update && apt-get install -y \
  tesseract-ocr \
  tesseract-ocr-eng \
  tesseract-ocr-ara \
  poppler-utils \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=10000
EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
