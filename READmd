ai-teaching-assistant/
├─ app.py                 # FastAPI app (upload + chat)
├──ingest_service.py
├── ingest_teacher_data.py      # earlier version: inserting sample teacher notes into Chroma
├── ingest_teacher_file.py      # new version: reads teacher .txt file and ingests into Chroma
├── query_chroma.py             # test querying Chroma for relevant notes
├── rag_server.py
├── venv/                       # your Python virtual environment
└── (later we’ll add FastAPI server files)
├─ ingest_text.py         # one-off/CLI ingestion helper (txt only)
├─ chroma_db/            # will be created automatically
    ├─ A.bin
    ├─ b.bin
    ├─ c.bin
├─ data/                  # teacher drops .txt files here (optional)
│   └─ thermodynamics.txt
└─ templates/
    ├─ base.html
    ├─ teacher.html       # upload + (re)ingest
    └─ student.html       # chat UI


=============================================================================================================================
ai-teaching-assistant/
│
├─ app.py                      # FastAPI app (main entry)
│
├─ services/
│   ├─ ingest_service.py        # ingestion logic (from ingest_teacher_file.py)
│   └─ query_service.py         # query logic (from query_chroma.py)
│
├─ scripts/
│   ├─ ingest_teacher_file.py   # CLI ingestion script (calls ingest_service)
│   └─ query_chroma.py          # CLI query tester (calls query_service)
│
├─ data/                       # teacher uploads .txt notes here
│   └─ thermodynamics.txt
│
├─ chroma_db/                  # vector DB auto storage
│
├─ templates/                  # HTML templates for FastAPI UI
│   ├─ base.html
│   ├─ teacher.html
│   └─ student.html
│
└─ venv/                       # local env (ignore in git)
└─requirements.txt
=============================================================================================================================================
Nice — let’s clean this up and move forward properly.

I’ll deliver, in order:

1. High-level design (HLD) — components and runtime flow.
2. Low-level design (LLD) — data shapes, APIs, sequence steps, error cases.
3. Concrete project layout (clean structure).
4. Ready-to-run code for the APIs (FastAPI) and lightweight UI templates.
5. How to run & test (curl/Postman + simple UI), and a note about wiring n8n later.

I kept this actionable: you can copy-paste the code and run it immediately.

# HLD

**Goal:** Teacher uploads `.txt` → store semantic chunks in ChromaDB → Student submits question → retrieve top-k chunks from Chroma → pass context+question to local Ollama model → return an answer.

Components:

* **FastAPI (app.py)** — host two main endpoints:

  * `/ingest` (teacher): accepts `.txt` file + subject/topic, stores into Chroma.
  * `/ask` (student): accepts subject/topic/question, returns an LLM answer.
* **ChromaDB** — local vector DB persisted to `./chroma_db`.
* **Embedding engine** — `sentence-transformers` (`all-MiniLM-L6-v2`) used via `chromadb.utils.embedding_functions`.
* **Ollama** — local LLM runner, called from /ask to synthesize retrieved context + question.
* **Templates** — teacher.html and student.html for upload and ask.
* **n8n (optional automation later)** — will POST to `/ingest` webhook for teacher uploads and send notifications after ingestion.

Operational flow:

1. Teacher uploads file → `app.py:/ingest` → service parses file into chunks → embeddings created → chunks saved in Chroma collection named `<teacher or subject>_<topic>`.
2. Student asks question → `app.py:/ask` → retrieve top-k related chunks → build prompt → call Ollama → return response (and optionally the retrieved chunks).

# LLD

## Collections & metadata

* Collection name: `"{subject}_{topic}".replace(" ", "_").lower()`
* Each stored item:

  * id: string (unique, e.g. `<filename>_<i>`)
  * document text: chunk text
  * metadata: `{ "subject": subject, "topic": topic, "filename": filename, "source": "teacher_upload" }`

## Chunking strategy

* Split input text on blank lines and/or sentences.
* Recommended chunk size: \~300–800 chars (or sentence boundaries). Overlap optional 50–120 chars.

## APIs

### POST /ingest

Request: multipart/form-data

* fields: `subject` (str), `topic` (str), `file` (.txt only)
  Behavior:
* validate file extension `.txt`
* read file bytes => text
* chunk text into pieces
* for each chunk: add to Chroma collection with id and metadata
  Response:

```json
{ "status":"ok", "collection":"physics_thermodynamics", "chunks": 5 }
```

Errors: 400 for validation, 500 for ingestion errors.

### POST /ask

Request: JSON or form-data

* `subject` (str), `topic` (str), `question` (str), optional `k` (int, default 3)
  Behavior:
* connect to Chroma, collection name as above
* if collection not found => return helpful error
* query collection to get top-k documents
* build prompt template:

  ```
  You are an AI tutor. Use the following context to answer the question.
  CONTEXT:
  <doc1>
  <doc2>
  QUESTION: <question>
  ANSWER:
  ```
* call Ollama (local) with prompt; return the model output and the retrieved docs (optional).
  Response:

```json
{ "answer": "...", "sources": [ ... ] }
```

## Ollama calling details

Two ways:

* Use `ollama serve` (HTTP API at `http://localhost:11434`) — if you run `ollama serve`. Then call its HTTP endpoint.
* Use `ollama run <model>` via subprocess and pass prompt via stdin — the script below uses subprocess for simplicity (works with your local `ollama run model`).

## Error handling

* Missing collection: return `{"error":"collection not found; ingest first"}` with 404.
* Ollama not available: return `{"error":"llm unavailable"}` with 502.

# Clean Project Layout (final)

```
ai-teaching-assistant/
├─ app.py                          # FastAPI app (routes -> services)
├─ services/
│   ├─ ingest_service.py           # ingestion logic (file -> chunks -> chroma)
│   └─ query_service.py            # retrieval + call to Ollama
├─ scripts/
│   ├─ ingest_teacher_file.py      # CLI helper to ingest local .txt files (calls ingest_service)
│   └─ query_chroma.py             # CLI retrieval for debugging (calls query_service)
├─ data/                           # teacher files (optional)
│   └─ thermodynamics.txt
├─ chroma_db/                      # created by chromadb
├─ templates/
│   ├─ base.html
│   ├─ teacher.html
│   └─ student.html
├─ requirements.txt
└─ venv/
```

# Code — copy these files into your project

> Note: I use `chromadb` with `SentenceTransformer` embeddings (fast and local). Ollama is called with `subprocess` `ollama run <model>` and prompt via stdin. Adjust model name to what you've pulled (you showed `llama3.1:8b` and `gemma2:2b`).

---

## services/ingest\_service.py

```python
# services/ingest_service.py
import os
import uuid
from typing import List
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# Use sentence-transformers via chromadb helper
EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    # simple split by blank line or sentences fallback
    parts = []
    # split on double newline first
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            parts.append(para)
        else:
            # fallback: split into sentences naive by period
            sentences = [s.strip() for s in para.split(". ") if s.strip()]
            cur = ""
            for s in sentences:
                if len(cur) + len(s) + 2 <= max_chars:
                    cur = (cur + " " + s).strip()
                else:
                    if cur:
                        parts.append(cur)
                    cur = s
            if cur:
                parts.append(cur)
    return parts

def ingest_text_file(file_path: str, subject: str, topic: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection_name = f"{subject}_{topic}".replace(" ", "_").lower()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=EMBEDDING_FN
    )

    ids = []
    docs = []
    metadatas = []
    for i, c in enumerate(chunks):
        doc_id = f"{os.path.basename(file_path)}_{i}_{uuid.uuid4().hex[:6]}"
        ids.append(doc_id)
        docs.append(c)
        metadatas.append({
            "subject": subject,
            "topic": topic,
            "filename": os.path.basename(file_path),
            "source": "teacher_upload"
        })

    collection.add(documents=docs, ids=ids, metadatas=metadatas)
    return {"collection": collection_name, "chunks": len(docs)}
```

---

## services/query\_service.py

```python
# services/query_service.py
import os
import subprocess
from typing import Tuple, List
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # change if needed

def retrieve_context(subject: str, topic: str, question: str, k: int = 3):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_name = f"{subject}_{topic}".replace(" ", "_").lower()
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return None
    res = collection.query(query_texts=[question], n_results=k, include=["documents", "metadatas", "distances"])
    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    sources = [{"text": d, "meta": m, "score": s} for d,m,s in zip(documents, metadatas, distances)]
    return sources

def call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout_sec: int = 60) -> str:
    """
    Calls local ollama model using subprocess. It runs `ollama run <model>` and passes prompt via stdin.
    Requires `ollama` binary in PATH.
    """
    proc = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        out, err = proc.communicate(input=prompt, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        return "Error: Ollama timed out."
    if err and not out:
        # Ollama may print on stderr; return error for debugging
        return f"Ollama error: {err.strip()}"
    return out.strip()

def answer_question(subject: str, topic: str, question: str, k: int = 3) -> dict:
    sources = retrieve_context(subject, topic, question, k)
    if sources is None:
        return {"error": "collection_not_found"}
    # build prompt
    context_text = "\n\n---\n\n".join([s["text"] for s in sources])
    prompt = f"""You are an expert AI tutor. Use the context below to answer the question concisely and clearly. Cite short source lines where helpful.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""
    llm_output = call_ollama(prompt)
    return {"answer": llm_output, "sources": sources}
```

---

## app.py (FastAPI)

```python
# app.py
import os
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from services.ingest_service import ingest_text_file
from services.query_service import answer_question

app = FastAPI()
templates = Jinja2Templates(directory="templates")
os.makedirs("data", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse("/teacher")

@app.get("/teacher", response_class=HTMLResponse)
def teacher_page(request: Request):
    return templates.TemplateResponse("teacher.html", {"request": request})

@app.post("/ingest")
async def ingest(subject: str = Form(...), topic: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt allowed")
    save_path = os.path.join("data", file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    res = ingest_text_file(save_path, subject, topic)
    return JSONResponse(res)

@app.get("/student", response_class=HTMLResponse)
def student_page(request: Request):
    return templates.TemplateResponse("student.html", {"request": request})

@app.post("/ask")
async def ask(subject: str = Form(...), topic: str = Form(...), question: str = Form(...), k: int = Form(3)):
    res = answer_question(subject, topic, question, k=int(k))
    if "error" in res and res["error"] == "collection_not_found":
        raise HTTPException(status_code=404, detail="Collection not found. Did you ingest data?")
    return templates.TemplateResponse("student.html", {
        "request": Request,
        "answer": res.get("answer", ""),
        "sources": res.get("sources", []),
        "subject": subject,
        "topic": topic,
        "question": question
    })
```

> Note: The POST `/ask` returns the student.html template for a built-in simple UI experience. You can adapt to JSON API if you want frontend JS.

---

## templates/teacher.html (small changes)

```html
{% extends "base.html" %}
{% block content %}
<h2>Teacher Upload</h2>
<form action="/ingest" method="post" enctype="multipart/form-data">
  <label>Subject</label>
  <input name="subject" type="text" required />
  <label>Topic</label>
  <input name="topic" type="text" required />
  <label>Text file</label>
  <input type="file" name="file" accept=".txt" required />
  <button type="submit">Upload & Ingest</button>
</form>
{% endblock %}
```

## templates/student.html

```html
{% extends "base.html" %}
{% block content %}
<h2>Student Chat</h2>
<form action="/ask" method="post">
  <input name="subject" placeholder="Physics" required/>
  <input name="topic" placeholder="Thermodynamics" required/>
  <textarea name="question" placeholder="Explain the first law" required></textarea>
  <button type="submit">Ask</button>
</form>

{% if answer %}
  <h3>Answer</h3>
  <pre>{{answer}}</pre>

  <h4>Sources</h4>
  <ul>
  {% for s in sources %}
    <li>{{ s.meta.filename }} — {{ s.text[:140] }}...</li>
  {% endfor %}
  </ul>
{% endif %}
{% endblock %}
```

---

## scripts/ingest\_teacher\_file.py (CLI wrapper)

```python
# scripts/ingest_teacher_file.py
import sys
from services.ingest_service import ingest_text_file

if len(sys.argv) < 4:
    print("Usage: python ingest_teacher_file.py path/to/file.txt Subject Topic")
    sys.exit(1)

path = sys.argv[1]
subject = sys.argv[2]
topic = sys.argv[3]
res = ingest_text_file(path, subject, topic)
print("Ingested:", res)
```

---

## requirements.txt (suggested)

```
fastapi
uvicorn[standard]
chromadb
sentence-transformers
jinja2
```

Add `ollama` installed separately (OS-level).

# How to run & test

1. Install dependencies in your venv:

```bash
pip install -r requirements.txt
```

2. Start FastAPI:

```bash
uvicorn app:app --reload --port 9000
```

3. Teacher ingestion (UI):

* Open `http://127.0.0.1:9000/teacher` → upload thermodynamics.txt with Subject `Physics` and Topic `Thermodynamics`.

Or CLI:

```bash
python scripts/ingest_teacher_file.py data/thermodynamics.txt Physics Thermodynamics
```

4. Test retrieval via UI:

* Open `http://127.0.0.1:9000/student`, ask “What is the first law of thermodynamics?” — app will retrieve context from Chroma and call Ollama.

5. Test retrieval via curl (JSON API style):

```bash
curl -X POST "http://127.0.0.1:9000/ingest" -F "subject=Physics" -F "topic=Thermodynamics" -F "file=@data/thermodynamics.txt"
curl -X POST "http://127.0.0.1:9000/ask" -F "subject=Physics" -F "topic=Thermodynamics" -F "question=Explain the first law of thermodynamics" -F "k=3"
```

# Notes & troubleshooting

* If `ollama` is not available in PATH or `ollama run <model>` blocks differently on your machine, switch to `ollama serve` mode and call its HTTP API or adjust `call_ollama()` to use `requests.post` to `http://localhost:11434/api/generate` (I can provide that variant if you prefer).
* If Chromadb raises a `collection not found` error on `get_collection`, catch safely — current code does that.
* On Windows, huggingface caching may warn about symlinks — harmless.
* Keep `chroma_db/` in your `.gitignore`.

# Next steps (pick one)

* I can adapt `call_ollama()` to use **ollama HTTP API** if you prefer `ollama serve`.
* I can produce an **n8n workflow JSON** (importable) that posts teacher uploads to `/ingest` and sends email notifications.
* I can add **authentication** for teacher endpoints (token) if needed.

Which next step do you want me to implement now? (HTTP Ollama adaptation, n8n flow, or add auth + teacher isolation?)



Perfect — thanks for pasting the full `app.py`.
I’ll rewrite it so that:

* **Ingest** works with `class_name + subject + topic` (so your Chroma collections are consistent).
* **Records API** is available (`/records`) to fetch back stored docs for testing.
* Your `/ask` stays integrated with `query_notes`.

Here’s the updated **`app.py`**:

```python
import os
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from services.ingest_service import ingest_text_file
from services.query_service import query_notes as answer_question
import chromadb

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ✅ Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to frontend origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

# Global chroma client for /records endpoint
chroma_client = chromadb.PersistentClient(path="chroma_db")


@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse("/teacher")


@app.get("/teacher", response_class=HTMLResponse)
def teacher_page(request: Request):
    return templates.TemplateResponse("teacher.html", {"request": request})


# ✅ Ingest notes into Chroma
@app.post("/ingest")
async def ingest(
    class_name: str = Form(...),
    subject: str = Form(...),
    topic: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt allowed")

    save_path = os.path.join("data", file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    res = ingest_text_file(save_path, class_name, subject, topic)
    return JSONResponse(res)


# ✅ View stored records for testing
@app.get("/records")
async def get_records(
    class_name: str,
    subject: str,
    topic: str
):
    try:
        collection_name = f"{class_name}_{subject}_{topic}".lower().replace(" ", "_")
        collection = chroma_client.get_collection(name=collection_name)
        results = collection.peek()
        return {"collection": collection_name, "records": results}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {e}")


@app.get("/student", response_class=HTMLResponse)
def student_page(request: Request):
    return templates.TemplateResponse("student.html", {"request": request})


# ✅ Ask questions
@app.post("/ask")
async def ask(
    request: Request,
    class_name: str = Form(...),
    subject: str = Form(...),
    topic: str = Form(...),
    question: str = Form(...),
    k: int = Form(3)
):
    try:
        res = answer_question(question, class_name, subject, topic, k=int(k))

        if "error" in res:
            raise HTTPException(status_code=400, detail=res["error"])

        # JSON response for API clients
        if request.headers.get("accept") == "application/json":
            return JSONResponse(content=res)

        # HTML template for browser
        return templates.TemplateResponse("student.html", {
            "request": request,
            "answer": res.get("answer", ""),
            "sources": res.get("sources", []),
            "subject": subject,
            "topic": topic,
            "question": question
        })

    except Exception as e:
        import traceback
        print("❌ ERROR in /ask:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
```

---

### ✅ Flow to Test

1. **Start FastAPI**

```bash
uvicorn app:app --reload
```

2. **POST /ingest** (upload notes)

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -F "class_name=Class10" \
  -F "subject=Physics" \
  -F "topic=Thermodynamics" \
  -F "file=@notes.txt"
```

3. **GET /records** (verify docs saved)

```bash
curl "http://127.0.0.1:8000/records?class_name=Class10&subject=Physics&topic=Thermodynamics"
```

4. **POST /ask** (ask a question)

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "accept: application/json" \
  -F "class_name=Class10" \
  -F "subject=Physics" \
  -F "topic=Thermodynamics" \
  -F "question=What is thermodynamics?" \
  -F "k=3"
```

---

Perfect 👍 let’s make the exact **Postman requests** you’ll need.
Your FastAPI expects `Form(...)` values, so in Postman you must use **Body → form-data**.

---

## 🔹 1. Ingest Notes (upload file)

**Request**

* Method: `POST`
* URL: `http://127.0.0.1:8000/ingest`
* Body → form-data:

| KEY          | VALUE                   | TYPE |
| ------------ | ----------------------- | ---- |
| `class_name` | Class10                 | Text |
| `subject`    | Physics                 | Text |
| `topic`      | Thermodynamics          | Text |
| `file`       | (choose notes.txt file) | File |

---

## 🔹 2. Get Records (verify saved notes)

**Request**

* Method: `GET`
* URL: `http://127.0.0.1:8000/records`
* Params:

| KEY          | VALUE          |
| ------------ | -------------- |
| `class_name` | Class10        |
| `subject`    | Physics        |
| `topic`      | Thermodynamics |

---

## 🔹 3. Ask a Question (query LLM using stored notes)

**Request**

* Method: `POST`
* URL: `http://127.0.0.1:8000/ask`
* Body → form-data:

| KEY          | VALUE                                    | TYPE |
| ------------ | ---------------------------------------- | ---- |
| `class_name` | Class10                                  | Text |
| `subject`    | Physics                                  | Text |
| `topic`      | Thermodynamics                           | Text |
| `question`   | What is the first law of thermodynamics? | Text |
| `k`          | 3                                        | Text |

---

✅ Response will return JSON with `answer` and `sources`.
If you open it in browser (`/student`), you’ll see the rendered template instead.

---

Would you like me to also prepare a **Postman Collection JSON** file you can import directly (so you don’t need to build these manually)?
