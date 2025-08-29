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
# @app.get("/records")
# async def get_records(
#     class_name: str,
#     subject: str,
#     topic: str
# ):
#     try:
#         collection_name = f"{class_name}_{subject}_{topic}".lower().replace(" ", "_")
#         collection = chroma_client.get_collection(name=collection_name)
#         results = collection.peek()
#         return {"collection": collection_name, "records": results}
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=f"Collection not found: {e}")

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

        # peek returns dict with numpy arrays and objects -> not JSON serializable
        results = collection.peek()

        # ✅ Extract only clean parts
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        return {
            "status": "success",
            "collection": collection_name,
            "count": len(ids),
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas
        }

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
