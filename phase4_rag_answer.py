"""
Phase 4 – Geerdete Antwort (RAG-Pipeline)
==========================================
Die Top-k Chunks aus Phase 3 als Kontext in den Prompt einbauen und das Modell
zwingen, NUR aus diesem Kontext zu antworten — mit Quellenangabe.

Hintergrund:
    Das ist die eigentliche RAG-Pipeline (Retrieval-Augmented Generation):
    
    1. RETRIEVAL (Phase 3): Semantisch passende Chunks zum Query finden
    2. AUGMENTED: Diese Chunks in den Prompt einfügen
    3. GENERATION (Phase 4): LLM antwortet nur basierend auf diesen Chunks
    
    Der System-Prompt ist kritisch. Er muss das Modell zwingen:
    - NUR aus dem gegebenen Kontext zu antworten
    - Zu sagen, wenn die Antwort nicht im Kontext steht
    - Quell-Chunks zu nennen (Grounding / Nachvollziehbarkeit)
    
    Ohne diesen strikten Prompt würde das LLM sein allgemeines Wissen nutzen
    und könnte Fakten erfinden (Halluzinationen), die nicht im Dokument stehen.
    
    Temperature = 0 ist essentiell: Wir wollen Fakten, keine Kreativität.

Architektur:
    User-Frage
      → embedden (mistral-embed)
      → Cosinus-Ähnlichkeit zu allen Chunks
      → Top-k Chunks auswählen
      → Als Kontext in den Prompt packen
      → LLM antwortet (mistral-small-latest, temperature=0)
      → Antwort + Quellen anzeigen

Lernziele:
    - Grounding: Antworten an belegbare Quellen binden
    - Halluzinationen vermeiden durch strikten System-Prompt
    - Zitate/Nachweise in LLM-Antworten
    - Vollständige RAG-Pipeline verstehen

Verwendung:
    Interaktiv: python phase4_rag_answer.py <pdf-name-ohne-endung>
    Oder direkt:  python phase4_rag_answer.py musterbekanntmachung_vgv
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

# ─── Konfiguration ───
TOP_K = 3           # Anzahl Chunks, die als Kontext dienen
MODEL = "mistral-small-latest"
TEMPERATURE = 0      # Deterministisch: Fakten, keine Kreativität
MAX_TOKENS = 500     # Antwortlänge begrenzen


# ─── RAG-System-Prompt ─────────────────────────────────────────────
# Das ist der wichtigste Prompt im ganzen Projekt. Er definiert die
# Grounding-Regeln, die das Modell zwingen, ehrlich zu sein.
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


def load_chunks(pdf_name: str) -> list[dict]:
    """Chunks aus Phase 2 laden."""
    path = CHUNKS_DIR / f"{pdf_name}_chunks.json"
    if not path.exists():
        print(f"❗ Keine Chunks für '{pdf_name}'. Erst Phase 2 ausführen.")
        sys.exit(1)
    return json.loads(path.read_text())


def load_or_create_embeddings(pdf_name: str, chunks: list[dict]) -> np.ndarray:
    """
    Embeddings laden (Cache) oder neu berechnen.
    Gleiche Logik wie Phase 3, hier als Hilfsfunktion für die Pipeline.
    """
    emb_path = EMBEDDINGS_DIR / f"{pdf_name}_embeddings.npy"
    if emb_path.exists():
        print(f"   💾 Embeddings aus Cache: {emb_path}")
        return np.load(emb_path)

    print(f"   🧠 Berechne Embeddings mit mistral-embed ...")
    chunk_texts = [c["text"] for c in chunks]
    response = client.embeddings.create(model="mistral-embed", inputs=chunk_texts)
    vecs = np.array([d.embedding for d in response.data], dtype=np.float64)
    np.save(emb_path, vecs)
    return vecs


def retrieve_chunks(
    query: str, chunk_vecs: np.ndarray, chunks: list[dict], top_k: int = TOP_K
) -> list[tuple[float, dict]]:
    """
    Semantische Suche: Findet die ähnlichsten Chunks zum Query.
    
    Pipeline:
    Query-Text → embedden → Cosinus-Ähnlichkeit → Top-k Indizes
    """
    # Query embedden
    response = client.embeddings.create(model="mistral-embed", inputs=[query])
    query_vec = np.array(response.data[0].embedding, dtype=np.float64)

    # Cosinus-Ähnlichkeit
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    scores = np.dot(chunk_norms, query_norm)

    # Top-k
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), chunks[i]) for i in top_indices]


def build_rag_prompt(query: str, retrieved: list[tuple[float, dict]]) -> str:
    """
    Baut den User-Prompt mit eingebettetem Kontext.
    
    Struktur:
        KONTEXT (Auszüge aus dem Dokument):
        [Chunk 0] ...
        [Chunk 3] ...
        
        FRAGE:
        ...
        
        ANTWORT:
    
    Das Modell sieht nur diesen Prompt und den System-Prompt.
    Die KONTEXT-Sektion ist das "Augmented" in RAG.
    """
    # Kontext-Teil: alle gefundenen Chunks mit ID und Score
    context_blocks = []
    for score, chunk in retrieved:
        context_blocks.append(
            f"[Chunk {chunk['chunk_id']} | Score: {score:.3f}]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    # Kompletten Prompt zusammenbauen
    prompt = f"""KONTEXT (Auszüge aus dem Ausschreibungsdokument):
---
{context_text}
---

FRAGE:
{query}

ANTWORT:"""

    return prompt


def ask_rag(query: str, pdf_name: str, verbose: bool = True) -> str:
    """
    Vollständige RAG-Pipeline für eine Frage.
    
    1. Chunks laden
    2. Embeddings laden/erstellen
    3. Retrieval: relevante Chunks finden
    4. Prompt mit Kontext bauen
    5. LLM antworten lassen
    6. Antwort + Quellen zurückgeben
    """
    # Schritt 1 & 2: Chunks und Embeddings
    chunks = load_chunks(pdf_name)
    chunk_vecs = load_or_create_embeddings(pdf_name, chunks)

    if verbose:
        print(f"\n📄 Dokument: {pdf_name}")
        print(f"🔍 Frage:    {query}")
        print(f"{'─' * 60}")

    # Schritt 3: Retrieval
    retrieved = retrieve_chunks(query, chunk_vecs, chunks, top_k=TOP_K)

    if verbose:
        print(f"\n📊 Top-{TOP_K} Chunks (Retrieval):")
        for rank, (score, chunk) in enumerate(retrieved, 1):
            preview = chunk["text"].replace("\n", " ")[:100]
            print(f"   #{rank} Chunk {chunk['chunk_id']} | Score={score:.3f} | \"{preview}...\"")

    # Schritt 4: Prompt mit Kontext bauen
    rag_prompt = build_rag_prompt(query, retrieved)

    # Schritt 5: LLM-Call
    if verbose:
        print(f"\n🤖 Sende RAG-Prompt an {MODEL} (temperature={TEMPERATURE}) ...")

    response = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": rag_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    answer = response.choices[0].message.content

    # Schritt 6: Ergebnis
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"📝 Antwort:")
        print(answer)
        print(f"\n   Tokens: {response.usage}")

    return answer


def interactive_mode(pdf_name: str):
    """Interaktiver Modus: Fragen stellen, bis 'exit'."""
    print(f"\n{'=' * 60}")
    print(f"🤖 RAG-Assistent für: {pdf_name}")
    print(f"   Modell: {MODEL} | temperature={TEMPERATURE} | Top-k={TOP_K}")
    print(f"   System-Prompt erzwingt Antworten NUR aus dem Kontext.")
    print(f"   'exit' zum Beenden, 'scores' für Scores, 'chunks' für Chunk-Vorschau")
    print(f"{'=' * 60}")

    # Chunks und Embeddings einmal laden
    chunks = load_chunks(pdf_name)
    chunk_vecs = load_or_create_embeddings(pdf_name, chunks)

    while True:
        try:
            query = input("\n❓ Frage: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Tschüss!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("👋 Tschüss!")
            break

        # Debug-Befehle
        if query.lower() == "scores":
            print("   Scores nicht einzeln abrufbar. Stelle eine konkrete Frage.")
            continue
        if query.lower() == "chunks":
            for c in chunks:
                print(f"   Chunk {c['chunk_id']}: {c['text'][:80]}...")
            continue

        # RAG-Pipeline
        retrieved = retrieve_chunks(query, chunk_vecs, chunks, top_k=TOP_K)
        rag_prompt = build_rag_prompt(query, retrieved)
        response = client.chat.complete(
            model=MODEL,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": rag_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        print(f"\n📊 Chunks: ", end="")
        for score, chunk in retrieved:
            print(f"[{chunk['chunk_id']} s={score:.2f}] ", end="")
        print(f"\n📝 Antwort:\n{response.choices[0].message.content}")


def main():
    if len(sys.argv) < 2:
        print("❗ Bitte PDF-Namen angeben (ohne .pdf):")
        print("   python phase4_rag_answer.py musterbekanntmachung_vgv")
        print("   python phase4_rag_answer.py musterbekanntmachung_vgv -i  (interaktiv)")
        sys.exit(1)

    pdf_name = Path(sys.argv[1]).stem
    interactive = "-i" in sys.argv or "--interactive" in sys.argv

    if interactive:
        interactive_mode(pdf_name)
    else:
        # Demo-Fragen durchlaufen
        demo_questions = [
            "Welche Fristen gelten für die Angebotsabgabe?",
            "Wie viele Lose gibt es und was beinhalten sie?",
            "Welche Eignungskriterien werden verlangt?",
            "Wie hoch ist das geschätzte Budget?",  # Test: steht wahrscheinlich nicht drin
        ]

        for question in demo_questions:
            print(f"\n{'=' * 60}")
            ask_rag(question, pdf_name)

        print(f"\n{'=' * 60}")
        print("✅ Phase 4 abgeschlossen.")
        print("   Das Modell antwortet nur aus dem Kontext und nennt die Quell-Chunks.")
        print("   Bei der Budget-Frage sollte es zugeben, dass diese Info fehlt.")
        print(f"   Für interaktiven Modus: python phase4_rag_answer.py {pdf_name} -i")


if __name__ == "__main__":
    main()
