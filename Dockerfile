FROM python:3.12-slim

# 1) System deps:
# - tesseract-ocr: OCR engine
# - poppler-utils: pdftoppm used by pdf2image for scanned PDF OCR
# - libgl1/libglib2.0-0: helps Pillow handle some image formats safely
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# 2) App setup
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# 3) Render uses PORT (usually 10000). Support both.
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# 4) Start FastAPI
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
