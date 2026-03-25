from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from docx import Document
import ollama
import chromadb
import io
import os
import re
import nltk
import asyncio

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize

app = FastAPI(title="RAG Assistant", version="3.0")

# --- CONFIG ---
LLM_MODEL   = "qwen3.5:latest"
EMBED_MODEL = "nomic-embed-text"
COLLECTION  = "documents"

# --- CHROMADB ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

# --- IN-MEMORY CHAT HISTORY ---
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
        r'(?<![A-Za-z\d])'
        r'([A-Za-z])'
        r'(?:[ \t]([A-Za-z])){2,}'
        r'(?![A-Za-z\d])'
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
        text = _C.sub(r'\1\2', text)
        text = _D.sub(r'\1\2\3', text)
        text = _E.sub(r'\1\2', text)
        if text == prev: break
    return text


def _fix_punctuation(text: str) -> str:
    text = re.sub(r'(\w)\.\s+([a-z]{2,})', r'\1.\2', text)
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([,:;])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'\s*[·•∙]\s*', ' · ', text)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'\s*--+\s*', ' — ', text)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    return text


def clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    for lig, rep in _LIGATURE_MAP.items():
        text = text.replace(lig, rep)
    text = _collapse_spaced_letters(text)
    text = _fix_broken_words(text)
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
    if not t:
        return False
    words = t.split()
    if not (1 <= len(words) <= 6):
        return False
    if ':' in t or '\u203a' in t or '/' in t:
        return False
    if t.isupper():
        return False
    if not t[0].isupper():
        return False
    if not any(w[0].isupper() for w in words if w.isalpha()):
        return False
    return True


def _looks_like_job_title(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    words = t.split()
    if not (1 <= len(words) <= 6):
        return False
    if ':' in t or '\u203a' in t:
        return False
    if not t[0].isupper():
        return False
    role_keywords = {
        'developer', 'engineer', 'analyst', 'specialist', 'designer',
        'architect', 'manager', 'consultant', 'lead', 'director',
        'scientist', 'researcher', 'intern', 'associate', 'full', 'stack',
        'backend', 'frontend', 'cloud', 'data', 'machine', 'web', 'python',
        'technical', 'digital', 'mobile', 'devops', 'qa', 'test'
    }
    lower_words = {w.lower().strip('&,') for w in words}
    return bool(lower_words & role_keywords)


def _split_by_sections(paragraphs: list[str]) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        words = para.split()
        is_allcaps_heading = para.isupper() and 1 <= len(words) <= 8
        next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
        is_person_boundary = _looks_like_name(para) and _looks_like_job_title(next_para)

        if is_allcaps_heading:
            if current_body or current_heading:
                sections.append((current_heading, current_body))
            current_heading = para
            current_body = []
            i += 1
        elif is_person_boundary:
            if current_body or current_heading:
                sections.append((current_heading, current_body))
            current_heading = para + " \u2014 " + next_para
            current_body = []
            i += 2
        else:
            current_body.append(para)
            i += 1

    if current_body or current_heading:
        sections.append((current_heading, current_body))

    return sections




def split_text_smart(text: str, chunk_size: int = 1400, overlap: int = 200) -> list[str]:
    """
    Section-aware chunker: keeps each person/section together in one chunk.
    Falls back to sentence splitting only if a section exceeds chunk_size.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    sections = _split_by_sections(paragraphs)
    chunks: list[str] = []

    for heading, body in sections:
        section_text = (heading + "\n\n" if heading else "") + "\n\n".join(body)
        section_text = section_text.strip()
        if not section_text:
            continue

        if len(section_text) <= chunk_size:
            chunks.append(section_text)
        else:
            prefix = (heading + "\n\n") if heading else ""
            sentences = sent_tokenize(section_text)
            temp = prefix
            for sent in sentences:
                if len(temp) + len(sent) + 1 <= chunk_size:
                    temp += (" " if temp else "") + sent
                else:
                    if temp.strip():
                        chunks.append(temp.strip())
                    temp = prefix + sent
            if temp.strip():
                chunks.append(temp.strip())

    if len(chunks) > 1 and overlap > 0:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + "\n\n" + chunks[i])
        return overlapped

    return chunks


# ─────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────

def _extract_docx(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    parts = []
    for block in doc.element.body:
        tag = block.tag.split('}')[-1]
        if tag == 'p':
            text = ''.join(
                n.text for n in block.iter()
                if n.tag.endswith('}t') and n.text
            )
            if text.strip():
                parts.append(text.strip())
        elif tag == 'tbl':
            for row in block.iter():
                if row.tag.endswith('}tr'):
                    cells = []
                    for cell in row.iter():
                        if cell.tag.endswith('}tc'):
                            cell_text = ''.join(
                                n.text for n in cell.iter()
                                if n.tag.endswith('}t') and n.text
                            )
                            if cell_text.strip():
                                cells.append(cell_text.strip())
                    if cells:
                        parts.append(' | '.join(cells))
    return '\n\n'.join(parts)


def extract_text(content: bytes, filename: str) -> str:
    try:
        name = filename.lower()
        if name.endswith('.docx'):
            raw = _extract_docx(content)
        elif name.endswith(('.txt', '.md')):
            raw = content.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"Unsupported type: {filename}. Use .docx, .txt or .md")
        result = clean_extracted_text(raw)
        print(f"[extract] '{filename}': {len(result)} chars")
        return result
    except Exception as e:
        print(f"[extract] Error: {e}")
        return ""


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return HTMLResponse(open("templates/index.html").read())


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(400, "Empty file")

        text = extract_text(content, file.filename)
        if not text or len(text) < 50:
            raise HTTPException(400, "Could not extract meaningful text")

        chunks = split_text_smart(text)
        if not chunks:
            raise HTTPException(400, "No content chunks created")

        added = 0
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:
                continue
            embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)['embedding']
            lines = [l.strip() for l in chunk.split('\n') if l.strip()]
            title = lines[0][:80] if lines and len(lines[0]) < 100 else "Content"
            collection.add(
                ids=[f"{file.filename}-{i}-{hash(chunk) % 100000}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": file.filename,
                    "section": title,
                    "chunk_idx": i,
                    "char_count": len(chunk)
                }]
            )
            added += 1

        return {
            "status": "success",
            "chunks_added": added,
            "total_chars": len(text),
            "filename": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[upload] Error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")


# ── Broad/listing query detection ──
_BROAD_PATTERNS = re.compile(
    r'\b(list|all|every|each|full list|how many|count|total|complete|summaris|summariz|overview|tell me about everyone|who are)',
    re.IGNORECASE
)

def _is_broad_query(msg: str) -> bool:
    """True when the question asks for an exhaustive list rather than a specific fact."""
    return bool(_BROAD_PATTERNS.search(msg))


def _fetch_all_chunks() -> tuple[str, list[str]]:
    """Return ALL documents from ChromaDB concatenated, plus source list."""
    total = collection.count()
    if total == 0:
        return "", []
    result = collection.get(include=["documents", "metadatas"])
    docs  = result.get("documents", [])
    metas = result.get("metadatas", [])
    parts = []
    sources: list[str] = []
    for doc, meta in zip(docs, metas):
        if doc and doc.strip():
            src = meta.get("source", "document") if meta else "document"
            parts.append(doc.strip())
            if src not in sources:
                sources.append(src)
    return "\n\n---\n\n".join(parts), sources


@app.post("/chat")
async def chat(
    message: str = Form(...),
    session_id: str = Form(default="default"),
):
    # Init session
    if session_id not in chat_history:
        chat_history[session_id] = []

    context_text = ""
    sources_used = []
    doc_count = collection.count()

    if doc_count > 0:
        try:
            if _is_broad_query(message):
                # Broad query — dump ALL chunks so nothing is missed
                context_text, sources_used = _fetch_all_chunks()
            else:
                # Targeted query — semantic search
                query_emb = ollama.embeddings(model=EMBED_MODEL, prompt=message)['embedding']
                results = collection.query(
                    query_embeddings=[query_emb],
                    n_results=min(20, doc_count),
                    include=['documents', 'metadatas', 'distances']
                )
                if results['documents'] and results['documents'][0]:
                    docs  = results['documents'][0]
                    metas = results['metadatas'][0] if results['metadatas'] else [{}] * len(docs)
                    dists = results['distances'][0] if results['distances'] else [1.0] * len(docs)

                    source_counts: dict = {}
                    context_parts = []
                    for doc, meta, dist in zip(docs, metas, dists):
                        if dist > 0.85:
                            continue
                        source = meta.get('source', 'document')
                        if source_counts.get(source, 0) >= 8:
                            continue
                        source_counts[source] = source_counts.get(source, 0) + 1
                        section = meta.get('section', 'content')
                        context_parts.append(f"[{source} — {section}]\n{doc.strip()}")
                        if source not in sources_used:
                            sources_used.append(source)
                    context_text = "\n\n---\n\n".join(context_parts)
        except Exception as e:
            print(f"[rag] Retrieval error: {e}")

    # Build system prompt
    if context_text:
        system_prompt = f"""You are a helpful AI assistant. Answer the user's question using ONLY the context provided below.

Rules:
- Be clear, accurate, and concise.
- If the answer isn't in the context, say so honestly — do not invent information.
- Use markdown formatting where it helps readability (lists, bold, etc).

CONTEXT:
{context_text}"""
    else:
        system_prompt = (
            "You are a helpful AI assistant. "
            "No documents have been uploaded yet, so answer from your general knowledge. "
            "Remind the user they can upload documents for document-specific answers."
        )

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[session_id][-20:]:  # keep last 20 turns
        messages.append(msg)
    messages.append({"role": "user", "content": message})

    # Call Qwen
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            options={
                'temperature': 0.2,
                'num_predict': 2048,
                'top_p': 0.9,
                'num_ctx': 16384,
            }
        )
        ai_message = response['message']['content'].strip()
    except Exception as e:
        print(f"[ollama] Error: {e}")
        ai_message = "I encountered an error. Is Ollama running? Try `ollama serve` in your terminal."

    # Update history
    chat_history[session_id].append({"role": "user",      "content": message})
    chat_history[session_id].append({"role": "assistant", "content": ai_message})

    # Trim history to 40 messages
    if len(chat_history[session_id]) > 40:
        chat_history[session_id] = chat_history[session_id][-40:]

    return {
        "response":     ai_message,
        "context_used": bool(context_text),
        "sources":      sources_used,
        "docs_indexed": doc_count,
    }


@app.post("/clear")
async def clear_chat(session_id: str = Form(default="default")):
    chat_history.pop(session_id, None)
    return {"status": "cleared"}


@app.delete("/documents")
async def delete_all_documents():
    """Wipe the entire vector store and recreate the collection."""
    global collection
    try:
        chroma_client.delete_collection(COLLECTION)
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        return {"status": "ok", "message": "All documents deleted"}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete documents: {e}")


@app.get("/health")
async def health():
    ollama_ok = False
    try:
        ollama.list()
        ollama_ok = True
    except Exception:
        pass
    return {
        "status":       "ok" if ollama_ok else "degraded",
        "ollama":       "connected" if ollama_ok else "unreachable",
        "llm_model":    LLM_MODEL,
        "embed_model":  EMBED_MODEL,
        "docs_indexed": collection.count(),
        "sessions":     len(chat_history),
    }


if __name__ == "__main__":
    import uvicorn
    os.makedirs("templates", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")