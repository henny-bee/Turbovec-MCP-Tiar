FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Dependencies first so the layer is cached across source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application packages. Data files (memory.db, index.tvim, metadata.json) are
# written to the working directory at runtime, not baked into the image.
COPY main.py ontology.json ./
COPY core/ ./core/
COPY storage/ ./storage/
COPY graph/ ./graph/
COPY memory/ ./memory/
COPY search/ ./search/
COPY extraction/ ./extraction/
COPY embeddings/ ./embeddings/
COPY reranking/ ./reranking/
COPY telemetry/ ./telemetry/
COPY librarian/ ./librarian/
COPY tools/ ./tools/
COPY vector_db.py ./

ENTRYPOINT ["python", "main.py"]
