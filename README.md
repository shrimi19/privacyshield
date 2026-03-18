# PrivacyShield

> **Redact sensitive PDFs locally, share safely, restore anytime with your personal key.**

PrivacyShield is a local-first PDF redaction system that automatically detects and blacks out personal information (PII) in PDF documents — with no data ever leaving your device. Every redaction is reversible: a unique encryption key lets the document owner restore original content at any time.

---

## Features

- **Automatic PII Detection** — names, SSNs, IBANs, phone numbers, emails, addresses, IDs, medical conditions, financial amounts, and more
- **Multilingual** — English, German, French, Italian, Spanish
- **Text + Scanned PDFs** — pdfplumber for text-based pages, pypdfium2 + PaddleOCR for scanned/image pages
- **Reversible Redaction** — encrypted `.privacyshield` key file lets you restore original values
- **100% Local** — no cloud, no API calls, no data leaves your machine
- **Simple Web UI** — drag-and-drop PDF upload, one-click download
- **REST API** — FastAPI backend with `/redact` and `/unredact` endpoints for programmatic access
- **Swiss PII Support** — AHV/AVS numbers, IBAN validation, RF creditor references

---

## How It Works
```
PDF Input
    ↓
Analyzer      →  classify each page: text / scanned / mixed
    ↓
Text pages    →  Extractor → NER Engine → Redactor → PDF Rebuilder
Scanned pages →  pypdfium2 → PaddleOCR → Image Redactor
Mixed pages   →  both pipelines run and results are merged
    ↓
Key Manager   →  encrypt token map → .privacyshield file (Fernet AES-128)
    ↓
Output: Redacted PDF  +  Encryption Key (shown once to user)
```

To restore: upload redacted PDF + paste your key → original values decrypted and restored.

---

## Detected PII Types

| Category | Examples |
|---|---|
| Person names | Full names, partial names |
| Contact info | Email, phone (international formats) |
| National IDs | SSN, passport, driver's license |
| Swiss-specific | AHV/AVS number (`756.XXXX.XXXX.XX`) |
| Financial | IBAN (with mod-97 validation), SWIFT/BIC, salary amounts |
| Document IDs | Policy numbers, invoice numbers, claim numbers, UUIDs, TAX IDs, RF references |
| Medical | Diagnosis labels, condition names |
| Location | Addresses |

---

## Project Structure
```
privacyshield/
├── app.py                          ← Flask web application
├── streamlit_app.py                ← Streamlit UI (alternative)
├── requirements.txt
├── privacyshield/
│   ├── analyzer/
│   │   └── pdf_analyzer.py         ← Classify pages: text/scanned/mixed
│   ├── text_pipeline/
│   │   ├── extractor.py            ← Extract text + char coordinates
│   │   ├── ner_engine.py           ← PII detection (Presidio + spaCy)
│   │   ├── redactor.py             ← Token replacement
│   │   └── pdf_rebuilder.py        ← Draw black boxes (PyMuPDF)
│   ├── image_pipeline/
│   │   ├── pdf_to_image.py         ← Convert PDF page → PIL Image (pypdfium2)
│   │   ├── ocr_engine.py           ← PaddleOCR text + coordinates
│   │   ├── image_classifier.py     ← Classify image regions (photo/scanned/id_card)
│   │   └── image_redactor.py       ← Draw boxes on image layer with token labels
│   ├── key_manager/
│   │   ├── encryptor.py            ← Fernet encryption
│   │   └── decryptor.py            ← Fernet decryption
│   ├── reconstructor/
│   │   └── pdf_merger.py           ← Merge text + image redactions into final PDF
│   ├── templates/
│   │   └── index.html              ← Single-page web UI
│   └── pipeline.py                 ← Orchestrates full pipeline
├── api/
│   ├── main.py                     ← FastAPI app
│   ├── routes/
│   │   ├── redact.py               ← POST /redact endpoint
│   │   ├── unredact.py             ← POST /unredact endpoint
│   │   └── health.py               ← GET /health endpoint
│   └── models/
│       └── schemas.py              ← Pydantic request/response models
└── testing/
    └── GSF/                        ← 100 synthetic test documents
```

---

## Prerequisites

- Python 3.11
- pip
- No system dependencies required — pypdfium2 bundles its own PDF renderer and works on Windows, macOS, and Linux without poppler

> **Note:** First run will download spaCy language models (~2GB total) and PaddleOCR models (~200MB) automatically.

---

## Installation and Run

### 1. Clone the repository
```bash
git clone https://github.com/DebDDash/privacyshield.git
cd privacyshield
```

### 2. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run this once first:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> Total install size is approximately 2.5–3GB due to PaddleOCR and spaCy models.

### 4. Run the Flask web app
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

### 6. Optional — Run FastAPI backend instead
```bash
uvicorn api.main:app --reload --port 8000
```

Then open the interactive API docs at:
```
http://127.0.0.1:8000/docs
```

---

## Usage

### Redact a PDF

1. Open `http://127.0.0.1:5000`
2. Drag and drop your PDF onto the upload area (max 50MB)
3. Click **Upload & Process**
4. Wait for processing (30–120 seconds depending on PDF size)
5. **Copy and save your Recovery Key** — shown once, never stored
6. Click **Download Redacted PDF**

### Restore Original Values

1. Click the **Decrypt PDF** tab
2. Upload the redacted PDF
3. Paste your Recovery Key
4. Click **Restore Original PDF**
5. Download the restored document

---

## Security Model
```
Original PDF  ──→  Redaction  ──→  Redacted PDF (safe to share)
                       │
                       └──→  Token Map  ──→  Fernet Encrypt  ──→  .privacyshield
                                                   │
                                             Recovery Key
                                          (shown once to user,
                                           never stored on disk)
```

- The `.privacyshield` file contains the encrypted mapping of tokens to original values
- The Recovery Key uses Fernet (AES-128-CBC + HMAC-SHA256)
- Neither the key nor the original values are ever stored by the application
- The redacted PDF alone reveals nothing — you need both the redacted PDF and the key to restore

---

## Supported Languages

| Language | spaCy Model | PII Detection |
|---|---|---|
| English | `en_core_web_lg` | Full |
| German | `de_core_news_lg` | Full + AHV/AVS |
| French | `fr_core_news_lg` | Full + AVS |
| Italian | `it_core_news_lg` | Full |
| Spanish | `es_core_news_lg` | Full |

Language is auto-detected per page using `langdetect`.

---

## Troubleshooting

**Port 5000 already in use:**
```bash
# macOS/Linux
lsof -i :5000
kill -9 <PID>
python app.py
```

**spaCy model not found:**
```bash
python -m spacy download en_core_web_lg
python -m spacy download de_core_news_lg
python -m spacy download fr_core_news_lg
python -m spacy download it_core_news_lg
python -m spacy download es_core_news_lg
```

**PaddleOCR slow on first run:**
First run downloads OCR models (~200MB). Subsequent runs use cached models and are significantly faster.

**numpy ABI error on startup:**
```
RuntimeError: module compiled against ABI version 0x1000009
```
Fix with:
```bash
pip install "numpy<2.0"
```

**PDF processing fails:**
- Ensure the PDF is not password-protected
- Check the terminal for detailed error messages

---

## Tech Stack

| Component | Library |
|---|---|
| Web framework | Flask |
| REST API | FastAPI + Uvicorn |
| NER / PII detection | Microsoft Presidio + spaCy |
| PDF text extraction | pdfplumber |
| PDF rendering / redaction | PyMuPDF (fitz) |
| PDF → image conversion | pypdfium2 |
| OCR (scanned pages) | PaddleOCR |
| Image redaction | Pillow (PIL) |
| Encryption | cryptography (Fernet) |
| Language detection | langdetect |

---

## Team

Built for the **GenAI Zürich Hackathon 2026** — GoCalma Privacy Redaction Track.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
