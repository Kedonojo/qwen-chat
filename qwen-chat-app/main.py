from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
import json
from docx import Document
import ollama
import chromadb
import io
import os
import re
import asyncio
import nltk
import numpy as np

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

app = FastAPI(title="RAG Assistant", version="4.0")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
LLM_MODEL   = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"
COLLECTION  = "documents"
MAX_CONTEXT_CHARS = 40000   # ~10k tokens — keeps broad queries within num_ctx
TTS_VOICE   = "af_heart"   # kokoro voice: af_heart, af_bella, am_adam, bm_lewis

# ─────────────────────────────────────────
# KOKORO TTS  (lazy-loaded on first /speak)
# ─────────────────────────────────────────
_kokoro_pipeline = None
_kokoro_lock     = asyncio.Lock()
_kokoro_ok       = False

def _try_import_kokoro():
    """Return True if kokoro is importable."""
    try:
        import kokoro  # noqa: F401
        return True
    except ImportError:
        return False

_kokoro_ok = _try_import_kokoro()
if _kokoro_ok:
    print("✅ Kokoro TTS found — /speak endpoint will be available.")
else:
    print("⚠️  Kokoro not installed. Install with:  pip install kokoro soundfile")


async def _get_kokoro():
    global _kokoro_pipeline
    if not _kokoro_ok:
        return None
    if _kokoro_pipeline is not None:
        return _kokoro_pipeline
    async with _kokoro_lock:
        if _kokoro_pipeline is None:
            try:
                from kokoro import KPipeline
                print(f"  [TTS] Loading Kokoro pipeline (voice={TTS_VOICE})…")
                _kokoro_pipeline = KPipeline(lang_code="a")   # 'a' = American English
                print("  [TTS] Kokoro ready ✅")
            except Exception as e:
                print(f"  [TTS] Failed to load Kokoro: {e}")
                return None
    return _kokoro_pipeline


def _strip_markdown(text: str) -> str:
    """Remove markdown so TTS doesn't read symbols aloud."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',     r'\1', text)
    text = re.sub(r'`([^`]*)`',     r'\1', text)
    text = re.sub(r'#{1,6}\s*',     '',    text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[•›\-]\s+', ', ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────
# CHROMADB
# ─────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

chat_history: dict = {}


# ─────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────
_LIGATURE_MAP = {
    '\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl',
    '\ufb03': 'ffi', '\ufb04': 'ffl', '\ufb05': 'st', '\ufb06': 'st',
    '\uf000': '', '\uf001': 'fi', '\uf002': 'fl',
}

def _collapse_spaced_letters(text: str) -> str:
    pattern = re.compile(
        r'(?<![A-Za-z\d])([A-Za-z])(?:[ \t]([A-Za-z])){2,}(?![A-Za-z\d])'
    )
    def _joiner(m): return m.group(0).replace(' ', '').replace('\t', '')
    for _ in range(3):
        new = pattern.sub(_joiner, text)
        if new == text: break
        text = new
    return text

def _fix_broken_words(text: str) -> str:
    _A = re.compile(r'(\b[A-Za-z]{2,})\s+([a-z]{1,3})\s+([A-Za-z]+)\b')
    _B = re.compile(r'(\b[A-Za-z]{3,})\s+([a-z])\s+([A-Za-z]{2,})\b')
    _C = re.compile(r'(\b[A-Za-z]{4,})\s+([a-z])\b(?!\s*[a-z])')
    _D = re.compile(r'\b([A-Z])\s+([a-z])\s+([A-Za-z]{3,})\b')
    _E = re.compile(r'(?<=[A-Za-z]{2})\s+([a-z])\s+([a-z]{1,3})(?=[^A-Za-z]|$)')
    for _ in range(6):
        prev = text
        text = _A.sub(r'\1\2\3', text)
        text = _B.sub(r'\1\2\3', text)
        text = _C.sub(r'\1\2',   text)
        text = _D.sub(r'\1\2\3', text)
        text = _E.sub(r'\1\2',   text)
        if text == prev: break
    return text

def _fix_punctuation(text: str) -> str:
    # NOTE: pattern `(\w)\.\s+([a-z]{2,})` removed — it stripped spaces after
    # periods before lowercase words, corrupting normal sentence structure.
    text = re.sub(r'([.!?])([A-Z])',        r'\1 \2', text)
    text = re.sub(r'([,:;])([A-Za-z])',     r'\1 \2', text)
    text = re.sub(r'\s*[·•∙]\s*',          ' · ',    text)
    text = re.sub(r'(\w)-\n(\w)',           r'\1\2',  text)
    text = re.sub(r'\s*--+\s*',            ' — ',    text)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    return text

def clean_extracted_text(text: str) -> str:
    if not text: return ""
    for lig, rep in _LIGATURE_MAP.items():
        text = text.replace(lig, rep)
    text = _collapse_spaced_letters(text)
    # _fix_broken_words is intentionally skipped — it was designed for PDF OCR
    # artifacts and corrupts normal Word document text by joining words like
    # "works in the department" → "worksinthedepartment"
    text = _fix_punctuation(text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +\n', '\n', text)
    return text.strip()


# ─────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────
def _looks_like_name(text: str) -> bool:
    t = text.strip()
    if not t: return False
    words = t.split()
    if not (1 <= len(words) <= 6): return False
    if ':' in t or '\u203a' in t or '/' in t: return False
    if t.isupper(): return False
    if not t[0].isupper(): return False
    if not any(w[0].isupper() for w in words if w.isalpha()): return False
    return True

def _looks_like_job_title(text: str) -> bool:
    t = text.strip()
    if not t: return False
    words = t.split()
    if not (1 <= len(words) <= 6): return False
    if ':' in t or '\u203a' in t: return False
    if not t[0].isupper(): return False
    role_keywords = {
        'developer', 'engineer', 'analyst', 'specialist', 'designer',
        'architect', 'manager', 'consultant', 'lead', 'director',
        'scientist', 'researcher', 'intern', 'associate', 'full', 'stack',
        'backend', 'frontend', 'cloud', 'data', 'machine', 'web', 'python',
        'technical', 'digital', 'mobile', 'devops', 'qa', 'test'
    }
    return bool({w.lower().strip('&,') for w in words} & role_keywords)

def _split_by_sections(paragraphs: list[str]) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_heading, current_body = "", []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
        if para.isupper() and 1 <= len(para.split()) <= 8:
            if current_body or current_heading:
                sections.append((current_heading, current_body))
            current_heading, current_body = para, []
            i += 1
        elif _looks_like_name(para) and _looks_like_job_title(next_para):
            if current_body or current_heading:
                sections.append((current_heading, current_body))
            current_heading, current_body = para + " \u2014 " + next_para, []
            i += 2
        else:
            current_body.append(para)
            i += 1
    if current_body or current_heading:
        sections.append((current_heading, current_body))
    return sections

def split_text_smart(text: str, chunk_size: int = 1400, overlap: int = 200) -> list[str]:
    if not text: return []
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    sections   = _split_by_sections(paragraphs)
    chunks: list[str] = []
    for heading, body in sections:
        section_text = (heading + "\n\n" if heading else "") + "\n\n".join(body)
        section_text = section_text.strip()
        if not section_text: continue
        if len(section_text) <= chunk_size:
            chunks.append(section_text)
        else:
            prefix    = (heading + "\n\n") if heading else ""
            sentences = sent_tokenize(section_text)
            temp      = prefix
            for sent in sentences:
                if len(temp) + len(sent) + 1 <= chunk_size:
                    temp += (" " if temp else "") + sent
                else:
                    if temp.strip(): chunks.append(temp.strip())
                    temp = prefix + sent
            if temp.strip(): chunks.append(temp.strip())
    if len(chunks) > 1 and overlap > 0:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            overlapped.append(chunks[i - 1][-overlap:] + "\n\n" + chunks[i])
        return overlapped
    return chunks


# ─────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────
def _extract_docx(content: bytes) -> str:
    doc   = Document(io.BytesIO(content))
    parts = []
    for block in doc.element.body:
        tag = block.tag.split('}')[-1]
        if tag == 'p':
            text = ''.join(n.text for n in block.iter() if n.tag.endswith('}t') and n.text)
            if text.strip(): parts.append(text.strip())
        elif tag == 'tbl':
            for row in block.iter():
                if row.tag.endswith('}tr'):
                    cells = []
                    for cell in row.iter():
                        if cell.tag.endswith('}tc'):
                            ct = ''.join(n.text for n in cell.iter() if n.tag.endswith('}t') and n.text)
                            if ct.strip(): cells.append(ct.strip())
                    if cells: parts.append(' | '.join(cells))
    return '\n\n'.join(parts)

def extract_text(content: bytes, filename: str) -> str:
    try:
        name = filename.lower()
        if name.endswith('.docx'):
            raw = _extract_docx(content)
        elif name.endswith(('.txt', '.md')):
            raw = content.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"Unsupported type: {filename}")
        result = clean_extracted_text(raw)
        print(f"[extract] '{filename}': {len(result)} chars")
        return result
    except Exception as e:
        print(f"[extract] Error: {e}")
        return ""


# ─────────────────────────────────────────
# BROAD QUERY DETECTION
# ─────────────────────────────────────────
_BROAD_RE = re.compile(
    r'\b(list|all|every|each|full list|how many|count|total|complete|'
    r'summaris|summariz|overview|tell me about everyone|who are)\b',
    re.IGNORECASE
)

def _is_broad_query(msg: str) -> bool:
    return bool(_BROAD_RE.search(msg))

def _fetch_all_chunks() -> tuple[str, list[str]]:
    if collection.count() == 0: return "", []
    result = collection.get(include=["documents", "metadatas"])
    parts, sources = [], []
    for doc, meta in zip(result.get("documents", []), result.get("metadatas", [])):
        if doc and doc.strip():
            parts.append(doc.strip())
            src = (meta or {}).get("source", "document")
            if src not in sources: sources.append(src)
    combined = "\n\n---\n\n".join(parts)
    return combined[:MAX_CONTEXT_CHARS], sources


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content: raise HTTPException(400, "Empty file")
        text = extract_text(content, file.filename)
        if not text or len(text) < 50: raise HTTPException(400, "Could not extract text")
        chunks = split_text_smart(text)
        if not chunks: raise HTTPException(400, "No chunks created")
        added = 0
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30: continue
            emb  = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)['embedding']
            lines = [l.strip() for l in chunk.split('\n') if l.strip()]
            title = lines[0][:80] if lines and len(lines[0]) < 100 else "Content"
            collection.add(
                ids=[f"{file.filename}-{i}-{hash(chunk) % 100000}"],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"source": file.filename, "section": title,
                             "chunk_idx": i, "char_count": len(chunk)}]
            )
            added += 1
        return {"status": "success", "chunks_added": added,
                "total_chars": len(text), "filename": file.filename}
    except HTTPException: raise
    except Exception as e:
        print(f"[upload] {e}")
        raise HTTPException(500, str(e))


@app.post("/chat")
async def chat(message: str = Form(...), session_id: str = Form(default="default")):
    if session_id not in chat_history:
        chat_history[session_id] = []

    context_text, sources_used = "", []
    doc_count = collection.count()

    if doc_count > 0:
        try:
            if _is_broad_query(message):
                context_text, sources_used = _fetch_all_chunks()
            else:
                qe      = ollama.embeddings(model=EMBED_MODEL, prompt=message)['embedding']
                results = collection.query(
                    query_embeddings=[qe],
                    n_results=min(10, doc_count),
                    include=['documents', 'metadatas', 'distances']
                )
                if results['documents'] and results['documents'][0]:
                    docs  = results['documents'][0]
                    metas = results['metadatas'][0] if results['metadatas'] else [{}]*len(docs)
                    dists = results['distances'][0]  if results['distances']  else [1.0]*len(docs)
                    sc: dict = {}
                    parts = []
                    for doc, meta, dist in zip(docs, metas, dists):
                        if dist > 0.75: continue
                        src = meta.get('source', 'document')
                        if sc.get(src, 0) >= 5: continue
                        sc[src] = sc.get(src, 0) + 1
                        parts.append(f"[{src} — {meta.get('section','content')}]\n{doc.strip()}")
                        if src not in sources_used: sources_used.append(src)
                    context_text = "\n\n---\n\n".join(parts)
        except Exception as e:
            print(f"[rag] {e}")

    system_prompt = (
        f"""You are a helpful AI assistant. Use the context below when the question is about the uploaded documents. For all other questions, answer from your general knowledge.

DOCUMENT CONTEXT:
{context_text}"""
        if context_text else
        "You are a helpful AI assistant. Answer questions clearly and concisely."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages += chat_history[session_id][-20:]
    messages.append({"role": "user", "content": message})

    def _stream():
        full_response = ""
        try:
            for chunk in ollama.chat(
                model=LLM_MODEL, messages=messages, stream=True,
                keep_alive="30m",
                options={'temperature': 0.2, 'num_predict': 1024,
                         'top_p': 0.9, 'num_ctx': 8192}
            ):
                token = (chunk.get('message') or {}).get('content') or ''
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            print(f"[ollama] {e}")
            err = "Error reaching Ollama. Is it running? Try `ollama serve`."
            full_response = err
            yield f"data: {json.dumps({'type': 'token', 'content': err})}\n\n"

        chat_history[session_id].append({"role": "user",      "content": message})
        chat_history[session_id].append({"role": "assistant", "content": full_response})
        if len(chat_history[session_id]) > 40:
            chat_history[session_id] = chat_history[session_id][-40:]

        yield f"data: {json.dumps({'type': 'done', 'sources': sources_used, 'context_used': bool(context_text), 'docs_indexed': doc_count})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/speak")
async def speak(text: str = Form(...)):
    """Generate speech with Kokoro TTS and return a WAV file."""
    if not _kokoro_ok:
        raise HTTPException(503, "Kokoro not installed. Run: pip install kokoro soundfile")

    pipeline = await _get_kokoro()
    if pipeline is None:
        raise HTTPException(503, "Kokoro failed to load — check server logs.")

    clean = _strip_markdown(text)
    if not clean:
        raise HTTPException(400, "No speakable text.")

    try:
        import soundfile as sf

        # Kokoro generator — yields KPipeline.Result objects (0.9.x API)
        audio_chunks = []
        for result in pipeline(clean, voice=TTS_VOICE, speed=1.0, split_pattern=r'\n+'):
            audio = result.audio
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)

        if not audio_chunks:
            raise HTTPException(500, "Kokoro produced no audio.")

        combined = np.concatenate(audio_chunks)
        sample_rate = 24000   # Kokoro outputs 24 kHz

        buf = io.BytesIO()
        sf.write(buf, combined, sample_rate, format="WAV")
        wav_bytes = buf.getvalue()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Length": str(len(wav_bytes)), "Cache-Control": "no-cache"}
        )
    except HTTPException: raise
    except Exception as e:
        print(f"[tts] {e}")
        raise HTTPException(500, f"TTS failed: {e}")


@app.get("/tts_status")
async def tts_status():
    return {"available": _kokoro_ok}


@app.post("/clear")
async def clear_chat(session_id: str = Form(default="default")):
    chat_history.pop(session_id, None)
    return {"status": "cleared"}


@app.delete("/documents")
async def delete_all_documents():
    global collection
    try:
        chroma_client.delete_collection(COLLECTION)
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"})
        return {"status": "ok", "message": "All documents deleted"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    ollama_ok = False
    try:
        ollama.list(); ollama_ok = True
    except Exception: pass
    return {"status": "ok" if ollama_ok else "degraded",
            "ollama": "connected" if ollama_ok else "unreachable",
            "llm_model": LLM_MODEL, "embed_model": EMBED_MODEL,
            "tts": "kokoro" if _kokoro_ok else "unavailable",
            "tts_voice": TTS_VOICE,
            "docs_indexed": collection.count(), "sessions": len(chat_history)}


if __name__ == "__main__":
    import uvicorn
    os.makedirs("templates", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")