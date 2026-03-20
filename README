# Orchestrated AI for Legal Documents

Pipeline-driven OCR + NLP backend for structured legal document processing.

**Stages**: S0 materialisation → S1 PDF classification → S2 language/script detection → S3 Paddle parsing (DLA/OCR) → S4 machine translation.

---

## Prerequisites: system-wide software

These must be installed at the OS level:

### PostgreSQL
more info at: https://www.postgresql.org/download/linux/ubuntu/

```bash
sudo apt install postgresql
sudo systemctl enable --now postgresql

# Create database and user
sudo -u postgres psql <<'SQL'
CREATE USER admin WITH PASSWORD 'secret';
CREATE DATABASE cm3070_ai_orchestration OWNER admin;
SQL
```

### Redis
more info at: https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/apt/
```bash
sudo apt install redis-server
sudo systemctl enable --now redis-server
# Verify
redis-cli ping   # → PONG
```

### Tesseract OCR
more info at: https://tesseract-ocr.github.io/tessdoc/Installation.html
```bash
sudo apt install tesseract-ocr
# Add language packs as needed, e.g.:
sudo apt install tesseract-ocr-eng tesseract-ocr-fra \
                tesseract-ocr-nld tesseract-ocr-rus \
                tesseract-ocr-ukr tesseract-ocr-ita \
                tesseract-ocr-spa
```

### CUDA toolkit (GPU only)

Install the CUDA version that matches your GPU driver before installing any GPU Python packages. The project was tested with CUDA 12.6.

---

## Python environment

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Core dependencies (CPU-only packages)

```bash

pip install "fastapi[standard]" "uvicorn[standard]" \
    "sqlalchemy[asyncio]" asyncpg "psycopg[binary]" \
    redis rq \
    pydantic pydantic-settings python-dotenv \
    alembic aiofiles httpie \
    opencv-python PyMuPDF \
    pytesseract lingua-language-detector \
    ipython numpy pytest
```

### 3. PaddlePaddle + PaddleOCR (GPU build)
more info at: https://www.paddlepaddle.org.cn/en/install
```bash
# Install the GPU wheel first — the index URL selects the CUDA 12.6 build
pip install paddlepaddle-gpu==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

pip install paddleocr
```

### 4. PaddleX (required for PP-StructureV3)

Check the installed PaddleX version first, then install the matching extras:

```bash
pip show paddlex | sed -n 's/^Version: //p'
# e.g. 3.4.2

pip install "paddlex[ocr]==3.4.2"   # substitute your actual version
```
more info at: https://www.paddleocr.ai/main/en/version3.x/paddlex/quick_start.html

### 5. CTranslate2 (CT2 translation backend)

```bash
pip install ctranslate2
```

### 6. TranslateGemma / quantization support

```bash
pip install torch transformers accelerate bitsandbytes
```

---

## Model setup

### CTranslate2 OPUS models

Convert HuggingFace OPUS models once and place them under `./models/`. These
commands require `transformers` and `torch` (see note above) but do not need to
run on the production machine.

```bash
ct2-transformers-converter \
    --model Helsinki-NLP/opus-mt-fr-en \
    --output_dir ./models/opus_fr_en_ct2 \
    --copy_files source.spm target.spm tokenizer_config.json vocab.json \
    --quantization int8 --force

ct2-transformers-converter \
    --model Helsinki-NLP/opus-mt-nl-en \
    --output_dir ./models/opus_nl_en_ct2 \
    --copy_files source.spm target.spm tokenizer_config.json vocab.json \
    --quantization int8 --force

ct2-transformers-converter \
    --model Helsinki-NLP/opus-mt-ru-en \
    --output_dir ./models/opus_ru_en_ct2 \
    --copy_files source.spm target.spm tokenizer_config.json vocab.json \
    --quantization int8 --force

ct2-transformers-converter \
    --model Helsinki-NLP/opus-mt-uk-en \
    --output_dir ./models/opus_uk_en_ct2 \
    --copy_files source.spm target.spm tokenizer_config.json vocab.json \
    --quantization int8 --force
```

### TranslateGemma

These models require accepting Google's license on HuggingFace before downloading.

```bash
# 1. Authenticate
hf auth login

# 2. Accept the license at:
#    https://huggingface.co/google/translategemma-4b-it

# 3. Download (the model is cached locally for offline use)
hf download google/translategemma-4b-it
```

The backend uses `local_files_only=True`, so the model must be downloaded before
running the worker. The `MT_GEMMA_MODEL` setting in `.env` controls which variant
is loaded (`translategemma-4b-it`, `translategemma-12b-it`, or `translategemma-27b-it`).

---

## Configuration

Copy the example environment files and edit as needed:

```
.env                  → sets APP_ENV (development | staging | production)
.env.development      → local overrides (DB URLs, Redis, model paths, etc.)
.env.staging
.env.production
```

---

## Database migrations

### First-time setup

If the `alembic/` directory does not yet exist in the project root, initialise it once:

```bash
alembic init alembic
```

Then edit `alembic.ini` and `alembic/env.py` to point at the database (the sync URL from `DB_SYNC_URL`).
This only needs to be done once per fresh clone.

### Normal workflow

```bash

# Generate first migration
`alembic revision --autogenerate -m "init schema"`

# Apply all pending migrations
alembic upgrade head

# Generate a new migration after changing models.py
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

---

## Running the application

Open three terminals (all with the venv activated).

### Terminal 1 — FastAPI web server

```bash
uvicorn app.main:app --reload
```

Interactive API docs available at `http://localhost:8000/docs`.

### Terminal 2 — RQ worker

```bash
python -m app.workers
```

### Terminal 3 — API calls

#### example: Ingesting a document

```bash
# Ingest a PDF (mode=1: dedup-aware default)
curl -X POST http://localhost:8000/ingest/file \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/document.pdf" \
     -F "mode=1"

# Always create a new document row, even if the file was already ingested
curl -X POST http://localhost:8000/ingest/file \
     -F "file=@/path/to/document.pdf" \
     -F "mode=0"
```

HTTP ingestion modes (currently implemented):

| Mode | Behaviour |
|---|---|
| `0` | Always create a new Document row and run a full pipeline |
| `1` | Dedup-aware default: skip re-processing true duplicates |

---

## Project layout

```
alembic/
app/
├── main.py                        FastAPI app + route registration
├── db.py                          Async + sync SQLAlchemy engines
├── models.py                      ORM models
├── workers.py                     RQ worker entry point
├── job_enqueue.py                 Web-safe enqueue helpers
├── config/
│   ├── config.py                  Pydantic Settings (env-driven)
│   ├── mt_config.py               MT language maps + token limits
│   └── PP-StructureV3.yaml        PaddlePaddle-StructureV3 Config
├── ingestion/
│   ├── ingest_core.py             Transport-agnostic ingestion logic
│   └── ingest_file.py             FastAPI UploadFile adapter
├── deletion/
│   └── delete.py                  Hard-delete + filesystem cleanup
├── pipeline/
│   ├── modes.py                   PipelineMode enum
│   ├── orchestrator_std.py        Worker pipeline orchestrator (S0–S5)
│   ├── s0_materialisation/        Page image rendering
│   ├── s1_classification/         PDF source-type classification
│   ├── s2_detection/              Language + script detection
│   ├── s3_parsing/                DLA/OCR (Paddle PP-StructureV3)
│   ├── s4_nlp/                    Machine translation
│   └── s5_embeddings/             Vector embeddings (not yet implemented)
├── routes/
│   ├── ingestion.py
│   ├── documents.py
│   ├── search.py
│   └── ui.py
└── utils/
    ├── logger.py
    ├── storage.py                 Filesystem layout + URI helpers
    ├── job_tracking.py            Job event logging
    └── mem_probe.py               RAM + GPU memory diagnostics
tests/
venv/
env.

```



