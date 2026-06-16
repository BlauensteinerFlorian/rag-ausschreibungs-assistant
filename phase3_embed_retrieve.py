"""
Phase 3 – Embedden & Retrieven
================================
Chunks mit mistral-embed vektorisieren, speichern und per Cosinus-Ähnlichkeit
die Top-k relevanten Chunks zu einer Frage finden.

Hintergrund:
    Embeddings sind dichte Vektoren (hier: 1024 Zahlen), die die Semantik
    eines Texts in einem hochdimensionalen Raum abbilden. Texte mit
    ähnlicher Bedeutung landen nahe beieinander.

    mistral-embed ist ein speziell für semantische Suche trainiertes Modell.
    Es wandelt beliebigen Text (bis 8k Token) in einen 1024-dimensionalen
    Vektor um — unabhängig von der Textlänge immer gleich groß.

    Cosinus-Ähnlichkeit misst den Winkel zwischen zwei Vektoren:
    - 1.0 = identische Richtung (gleiche Bedeutung)
    - 0.0 = orthogonal (nicht verwandt)
    - -1.0 = entgegengesetzt (selten bei Embeddings)

    Berechnung: cos(a,b) = (a·b) / (|a|·|b|)
    Also: beide Vektoren normalisieren, dann Skalarprodukt (dot product).

    Warum numpy und nicht gleich eine Vektordatenbank (FAISS, Chroma, ...)?
    → Lernzweck. ~10 Zeilen numpy zeigen, wie semantische Suche wirklich
      funktioniert. Später (Phase 7 optional) kann man auf FAISS upgraden.

    Die Embeddings werden als .npy gespeichert, damit sie nicht jedes Mal
    neu berechnet werden müssen (spart API-Kosten und Zeit).

Lernziele:
    - Was sind Embeddings? Wie entstehen sie?
    - Wie funktioniert Cosinus-Ähnlichkeit?
    - Wie baut man ein minimales Retrieval-System ohne externe Datenbank?

Verwendung:
    python phase3_embed_retrieve.py <pdf-name-ohne-endung>
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

# ─── Pfade ───
DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Konfiguration ───
TOP_K = 3  # Wie viele der ähnlichsten Chunks zurückgeben?


def load_chunks(pdf_name: str) -> list[dict]:
    """Lädt die in Phase 2 erzeugten Chunks aus der JSON-Datei."""
    path = CHUNKS_DIR / f"{pdf_name}_chunks.json"
    if not path.exists():
        print(f"❗ Keine Chunks gefunden für '{pdf_name}'. Erst Phase 2 ausführen.")
        sys.exit(1)
    return json.loads(path.read_text())


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Wandelt eine Liste von Texten in Embedding-Vektoren um.
    
    Ablauf:
    1. Alle Texte auf einmal an mistral-embed schicken (Batch-Request)
    2. Die Antwort enthält für jeden Text einen 1024-dim Vektor
    3. Zurück als numpy-Array: shape (n_texte, 1024)
    
    mistral-embed verarbeitet bis zu 8k Token pro Text und batcht
    mehrere Texte effizient in einem API-Call.
    """
    print(f"   🧠 Embedde {len(texts)} Texte mit mistral-embed ...")
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts,
    )
    # data ist eine Liste von EmbeddingResponse-Objekten,
    # jedes mit .embedding (Liste von 1024 floats)
    vectors = [d.embedding for d in response.data]
    return np.array(vectors, dtype=np.float64)


def cosine_similarity(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    """
    Cosinus-Ähnlichkeit zwischen einem Query-Vektor und mehreren Chunk-Vektoren.
    
    Mathe:
        cos(θ) = (a · b) / (|a| · |b|)
    
    Implementierung:
    1. Query-Vektor normalisieren (auf Länge 1 bringen)
    2. Alle Chunk-Vektoren zeilenweise normalisieren
    3. Matrix-Multiplikation: Chunks (n×d) · Query (d×1) → Scores (n×1)
       Da beide normalisiert sind, ist das Skalarprodukt = Cosinus-Ähnlichkeit.
    
    Returns: 1D-numpy-Array mit einem Score [0..1] pro Chunk.
    """
    # Normalisieren: Vektor durch seine Länge (L2-Norm) teilen
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    # Skalarprodukt der normalisierten Vektoren
    return np.dot(chunk_norms, query_norm)


def retrieve_top_k(
    query: str,
    chunk_vecs: np.ndarray,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[tuple[float, dict]]:
    """
    Findet die Top-k Chunks zu einer Frage.
    
    Pipeline:
    1. Query embedden     → 1024-dim Vektor
    2. Cosinus zu allen Chunks berechnen
    3. Nach Score absteigend sortieren
    4. Top-k zurückgeben
    
    Returns: Liste von (score, chunk_dict) Tupeln, bestes zuerst.
    """
    print(f"\n🔍 Query: \"{query}\"")
    
    # Schritt 1: Query vektorisieren
    query_vec = embed_texts([query])[0]  # [0] weil wir nur einen Vektor brauchen

    # Schritt 2+3: Ähnlichkeit berechnen
    scores = cosine_similarity(query_vec, chunk_vecs)

    # Schritt 4: Top-k Indizes (argsort gibt aufsteigend, also [::-1] für absteigend)
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

    # ─── Embeddings: entweder aus Cache laden oder neu berechnen ───
    emb_path = EMBEDDINGS_DIR / f"{pdf_name}_embeddings.npy"
    if emb_path.exists():
        print(f"💾 Lade gespeicherte Embeddings: {emb_path}")
        chunk_vecs = np.load(emb_path)
    else:
        # Alle Chunk-Texte extrahieren und embedden
        chunk_texts = [c["text"] for c in chunks]
        chunk_vecs = embed_texts(chunk_texts)
        # Für nächsten Lauf speichern (vermeidet erneute API-Kosten)
        np.save(emb_path, chunk_vecs)
        print(f"💾 Embeddings gespeichert: {emb_path}")

    # Shape-Info: (anzahl_chunks, 1024)
    print(f"   Shape: {chunk_vecs.shape} (Dimensionalität: {chunk_vecs.shape[1]})")

    # ─── Test-Queries: Semantische Suche in Aktion zeigen ───
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
            # Score: 0 = keine Ähnlichkeit, 1 = perfekte Übereinstimmung
            # Im RAG-Kontext sind Scores > 0.7 meist gut brauchbar
            preview = chunk["text"].replace("\n", " ")[:120]
            print(f"  #{rank} Score={score:.3f} | Chunk {chunk['chunk_id']} | \"{preview}...\"")

    print(f"\n✅ Phase 3 abgeschlossen.")
    print(f"   Die Top-Chunks zeigen hohe Scores bei semantisch passendem Inhalt.")
    print(f"   In Phase 4 werden diese Chunks als Kontext in den Prompt eingebaut.")


if __name__ == "__main__":
    main()
