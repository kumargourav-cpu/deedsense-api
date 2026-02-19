FROM python:3.12-slim

# System deps for OCR + PDF text extraction
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Render provides PORT, default to 10000
ENV PORT=10000

# Start
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
