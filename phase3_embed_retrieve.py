"""
Phase 3 – Embedden & Retrieven
================================
Chunks mit mistral-embed vektorisieren, speichern und per Cosinus-Ähnlichkeit
die Top-k relevanten Chunks zu einer Frage finden.
Lernziele: Embeddings verstehen, semantische Suche mit numpy selbst bauen.

Verwendung:
    python phase3_embed_retrieve.py <pfad-zur-pdf>
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 3  # Anzahl der relevantesten Chunks


def load_chunks(pdf_name: str) -> list[dict]:
    """Lädt die Chunks aus Phase 2."""
    path = CHUNKS_DIR / f"{pdf_name}_chunks.json"
    if not path.exists():
        print(f"❗ Keine Chunks gefunden für '{pdf_name}'. Erst Phase 2 ausführen.")
        sys.exit(1)
    return json.loads(path.read_text())


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Schickt eine Liste von Texten an mistral-embed und gibt die Embeddings
    als numpy-Array zurück (shape: [n_texte, embedding_dim]).
    """
    print(f"   🧠 Embedde {len(texts)} Texte mit mistral-embed ...")
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts,
    )
    vectors = [d.embedding for d in response.data]
    return np.array(vectors, dtype=np.float64)


def cosine_similarity(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    """
    Berechnet Cosinus-Ähnlichkeit zwischen einem Query-Vektor und
    einer Matrix von Chunk-Vektoren.
    Returns: 1D-Array mit Ähnlichkeitswerten [0..1] pro Chunk.
    """
    # Normieren
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    # Dot-Produkt der normierten Vektoren = Cosinus-Ähnlichkeit
    return np.dot(chunk_norms, query_norm)


def retrieve_top_k(
    query: str,
    chunk_vecs: np.ndarray,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[tuple[float, dict]]:
    """
    Embeddet die Query, berechnet Cosinus-Ähnlichkeit zu allen Chunks
    und gibt die Top-k Chunks mit Scores zurück.
    """
    print(f"\n🔍 Query: \"{query}\"")
    query_vec = embed_texts([query])[0]

    scores = cosine_similarity(query_vec, chunk_vecs)

    # Top-k Indizes
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [(float(scores[i]), chunks[i]) for i in top_indices]
    return results


def main():
    if len(sys.argv) < 2:
        print("❗ Bitte PDF-Namen angeben (ohne .pdf): python phase3_embed_retrieve.py musterbekanntmachung_vgv")
        sys.exit(1)

    pdf_name = Path(sys.argv[1]).stem

    # ─── Chunks laden ───
    chunks = load_chunks(pdf_name)
    print(f"📋 {len(chunks)} Chunks geladen")

    # ─── Embeddings erstellen oder laden ───
    emb_path = EMBEDDINGS_DIR / f"{pdf_name}_embeddings.npy"
    if emb_path.exists():
        print(f"💾 Lade gespeicherte Embeddings: {emb_path}")
        chunk_vecs = np.load(emb_path)
    else:
        chunk_texts = [c["text"] for c in chunks]
        chunk_vecs = embed_texts(chunk_texts)
        np.save(emb_path, chunk_vecs)
        print(f"💾 Embeddings gespeichert: {emb_path}")

    print(f"   Shape: {chunk_vecs.shape} (Dimensionalität: {chunk_vecs.shape[1]})")

    # ─── Test-Queries ───
    test_queries = [
        "Welche Fristen gelten für die Angebotsabgabe?",
        "Wie viele Lose gibt es und was beinhalten sie?",
        "Welche Eignungskriterien werden verlangt?",
    ]

    for query in test_queries:
        results = retrieve_top_k(query, chunk_vecs, chunks, top_k=TOP_K)

        print(f"\n{'─' * 60}")
        print(f"Top-{TOP_K} Chunks für: \"{query}\"")
        print(f"{'─' * 60}")
        for rank, (score, chunk) in enumerate(results, 1):
            preview = chunk["text"].replace("\n", " ")[:120]
            print(f"  #{rank} Score={score:.3f} | Chunk {chunk['chunk_id']} | \"{preview}...\"")

    print(f"\n✅ Phase 3 abgeschlossen.")
    print(f"   Die Top-Chunks zeigen hohe Scores bei semantisch passendem Inhalt.")
    print(f"   In Phase 4 werden diese Chunks als Kontext in den Prompt eingebaut.")


if __name__ == "__main__":
    main()
