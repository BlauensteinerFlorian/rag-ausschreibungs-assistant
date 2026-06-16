"""
Phase 7b – Streamlit-UI
=========================
Ein einfaches Web-Interface für den RAG-Assistenten:
- PDF hochladen (oder vorhandenes auswählen)
- Fragen stellen
- Antworten mit Quellen anzeigen

Hintergrund:
    Streamlit ist das schnellste Framework, um aus Python-Code eine
    interaktive Web-App zu machen. Kein HTML/CSS/JS nötig — nur Python.
    
    Die App hier vereint die Phasen 2-4 in einem Workflow:
    1. PDF-Upload → Text-Extraktion + Chunking (Phase 2)
    2. Chunks embedden (Phase 3)
    3. RAG-Pipeline mit Antwort + Quellen (Phase 4)
    
    Streamlit läuft lokal und ist gratis auf Streamlit Community Cloud
    deploybar. Das macht aus "gebaut" ein "gebaut und deployt".

Verwendung:
    streamlit run phase7b_streamlit_app.py
"""

import streamlit as st
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from mistralai.client import Mistral
from pypdf import PdfReader

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# ─── Konfiguration ───
CHUNK_CHARS = 2500
OVERLAP_CHARS = 250
TOP_K = 3
MODEL = "mistral-small-latest"
TEMPERATURE = 0

# ─── Grounding-Prompt ───
RAG_SYSTEM_PROMPT = """Du bist ein Assistent für öffentliche Ausschreibungen.
Du beantwortest Fragen AUSSCHLIESSLICH auf Basis des unten angegebenen Kontexts.

Wichtige Regeln:
1. Antworte NUR mit Informationen, die im Kontext stehen.
2. Wenn die Antwort nicht im Kontext zu finden ist, sage klar:
   "Diese Information steht nicht in den vorliegenden Dokumenten."
3. Gib nach deiner Antwort an, aus welchen Quell-Chunks du die Information
   entnommen hast, z.B.: "[Quelle: Chunk 3, Chunk 5]"
4. Erfinde keine Informationen hinzu. Kein Allgemeinwissen verwenden.
5. Antworte präzise und auf Deutsch."""


# ═══════════════════════════════════════════════════════════════
# PDF-Verarbeitung (Phase 2)
# ═══════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_file) -> str:
    """Extrahiert Text aus einem hochgeladenen PDF (pypdf)."""
    reader = PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def chunk_text(text: str) -> list[dict]:
    """Schneidet Text in Chunks (Sliding Window)."""
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        chunks.append({
            "chunk_id": chunk_id,
            "text": text[start:end],
            "start_char": start,
            "end_char": end,
        })
        chunk_id += 1
        if end == len(text):
            break
        start = end - OVERLAP_CHARS
    return chunks


# ═══════════════════════════════════════════════════════════════
# Embedding & Retrieval (Phase 3)
# ═══════════════════════════════════════════════════════════════

def create_embeddings(chunks: list[dict]) -> np.ndarray:
    """Erstellt Embeddings für alle Chunks."""
    chunk_texts = [c["text"] for c in chunks]
    response = client.embeddings.create(model="mistral-embed", inputs=chunk_texts)
    return np.array([d.embedding for d in response.data], dtype=np.float64)


def retrieve(query: str, chunk_vecs: np.ndarray, chunks: list[dict]):
    """Semantische Suche nach Top-k Chunks."""
    response = client.embeddings.create(model="mistral-embed", inputs=[query])
    query_vec = np.array(response.data[0].embedding, dtype=np.float64)
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    scores = np.dot(chunk_norms, query_norm)
    top_indices = np.argsort(scores)[::-1][:TOP_K]
    return [(float(scores[i]), chunks[i]) for i in top_indices]


# ═══════════════════════════════════════════════════════════════
# RAG-Antwort (Phase 4)
# ═══════════════════════════════════════════════════════════════

def rag_answer(question: str, chunks: list[dict], chunk_vecs: np.ndarray) -> tuple[str, list]:
    """RAG-Pipeline: Retrieval → Antwort → (Antwort, Quellen)."""
    retrieved = retrieve(question, chunk_vecs, chunks)

    # Kontext für den Prompt bauen
    context_blocks = []
    sources = []
    for score, chunk in retrieved:
        context_blocks.append(
            f"[Chunk {chunk['chunk_id']} | Score: {score:.3f}]\n{chunk['text']}"
        )
        sources.append({
            "chunk_id": chunk["chunk_id"],
            "score": score,
            "preview": chunk["text"][:100].replace("\n", " ") + "...",
        })
    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""KONTEXT (Auszüge aus dem Ausschreibungsdokument):
---
{context_text}
---

FRAGE:
{question}

ANTWORT:"""

    response = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=500,
    )
    return response.choices[0].message.content, sources


# ═══════════════════════════════════════════════════════════════
# Streamlit App
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RAG-Ausschreibungs-Assistent",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG-Ausschreibungs-Assistent")
st.caption("Fragen zu Ausschreibungsdokumenten — mit Quellenangabe, ohne Halluzinationen")

# ─── Sidebar: Upload & Status ───
with st.sidebar:
    st.header("📎 Dokument")
    uploaded_file = st.file_uploader(
        "PDF hochladen",
        type=["pdf"],
        help="Lade eine Ausschreibung oder ein Policy-Dokument als PDF hoch.",
    )

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")

    st.divider()
    st.header("⚙️ Einstellungen")
    top_k = st.slider("Top-k Chunks", 1, 10, TOP_K, help="Wie viele relevante Textabschnitte werden als Kontext genutzt?")
    show_sources = st.checkbox("Quellen anzeigen", value=True)

    st.divider()
    st.caption(f"Modell: {MODEL} | temp={TEMPERATURE}")

# ─── Hauptbereich ───
if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.chunk_vecs = None
    st.session_state.doc_name = None

# PDF verarbeiten
if uploaded_file and st.session_state.doc_name != uploaded_file.name:
    with st.spinner("📄 Verarbeite PDF ..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        chunk_vecs = create_embeddings(chunks)
        st.session_state.chunks = chunks
        st.session_state.chunk_vecs = chunk_vecs
        st.session_state.doc_name = uploaded_file.name
    st.success(f"✅ {len(chunks)} Chunks aus {len(text):,} Zeichen erstellt")

# Chat-Interface
st.divider()

if st.session_state.chunks:
    question = st.chat_input("Frage zum Dokument stellen ...")
    
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Suche relevante Passagen + generiere Antwort ..."):
                answer, sources = rag_answer(
                    question,
                    st.session_state.chunks,
                    st.session_state.chunk_vecs,
                )
            
            st.markdown(answer)
            
            if show_sources and sources:
                with st.expander("📊 Verwendete Quellen"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {src['chunk_id']}** (Score: {src['score']:.3f})")
                        st.caption(src["preview"])
                        if i < len(sources):
                            st.divider()
else:
    st.info("👈 Lade ein PDF hoch, um loszulegen.")
