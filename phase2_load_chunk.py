"""
Phase 2 – Dokument laden & chunken
==================================
PDF-Text mit pypdf extrahieren und in überlappende Chunks schneiden
(ca. 600 Token ≈ 2.500 Zeichen, 10 % Überlappung).

Hintergrund:
    Sprachmodelle haben ein begrenztes Kontextfenster (z. B. 128k Token bei
    mistral-small). Ein komplettes Ausschreibungs-PDF kann aber schnell
    mehrere zehntausend Token umfassen. Würde man alles auf einmal in den
    Prompt packen, wäre a) das Fenster irgendwann voll, und b) das Modell
    verliert bei langen Dokumenten den Fokus („lost in the middle").
    Deshalb: Chunking. Das Dokument wird in handliche Stücke geschnitten,
    aus denen später die semantisch passenden herausgesucht werden.

    Die Überlappung (250 Zeichen) verhindert, dass ein relevanter Satz
    genau an der Chunk-Grenze durchgeschnitten wird und somit keinem
    Chunk vollständig zugeordnet werden kann.

    Faustregel für Token-Schätzung im Deutschen:
    1 Token ≈ 4 Zeichen → 2.500 Zeichen ≈ 625 Token.
    Exakter wäre mistral-common Tokenizer (Upgrade-Schicht), aber für
    den MVP reicht die Zeichen-basierte Schätzung.

Lernziele:
    - Kontextfenster-Grenzen von LLMs verstehen
    - Warum Chunking für RAG nötig ist
    - Trade-off: größere Chunks = mehr Kontext, aber ungenaueres Retrieval

Verwendung:
    python phase2_load_chunk.py <pfad-zur-pdf>
"""

import sys
import json
from pathlib import Path
from pypdf import PdfReader

# ─── Konfiguration ───
CHUNK_CHARS = 2500      # ≈ 600 Token bei deutschem Text (1 Token ≈ 4 Zeichen)
OVERLAP_CHARS = 250     # 10 % von 2500 — verhindert Schnitte mitten im Satz
DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def extract_text(pdf_path: Path) -> str:
    """
    Liest alle Seiten eines PDFs mit pypdf und gibt den Fließtext zurück.
    
    pypdf funktioniert gut für textbasierte PDFs (z. B. aus Word exportiert).
    Bei gescannten Dokumenten (Bild-PDFs) liefern die meisten Seiten
    leere Strings — dafür bräuchte man dann Mistral OCR (mistral-ocr-latest).
    Genau dieser Fall ist als späterer Upgrade-Punkt im Projekt vorgesehen.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text)
        else:
            print(f"   ⚠ Seite {i+1}: kein Text extrahierbar (gescannt? → später Mistral OCR)")
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """
    Schneidet einen langen Text in überlappende Chunks mit fester Größe.
    
    Algorithmus: Sliding Window
    - Start bei Zeichen 0
    - Jeder Chunk umfasst chunk_size Zeichen (bzw. bis zum Textende)
    - Der nächste Chunk beginnt bei (aktuelles Ende − overlap)
    - Dadurch überlappen benachbarte Chunks um 'overlap' Zeichen
    - Der letzte Chunk ist typischerweise kürzer (Rest des Texts)
    
    Returns: Liste von dicts mit:
        - chunk_id: fortlaufende Nummer
        - text: der Chunk-Text
        - start_char: Startposition im Originaltext
        - end_char: Endposition im Originaltext
    """
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_slice = text[start:end]
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_slice,
            "start_char": start,
            "end_char": end,
        })
        chunk_id += 1
        # Wenn wir am Ende sind: aufhören
        if end == len(text):
            break
        # Nächsten Chunk um 'overlap' Zeichen nach links verschieben
        start = end - overlap

    return chunks


def main():
    if len(sys.argv) < 2:
        print("❗ Bitte PDF-Pfad angeben: python phase2_load_chunk.py <pfad>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"❗ Datei nicht gefunden: {pdf_path}")
        sys.exit(1)

    pdf_name = pdf_path.stem  # Dateiname ohne .pdf

    # Schritt 1: Text aus PDF extrahieren
    print(f"📄 Lade: {pdf_path.name}")
    text = extract_text(pdf_path)
    print(f"   Gesamtzeichen: {len(text):,}")
    print(f"   ≈ Token:      {len(text) // 4:,}  (Faustregel: 1 Token ≈ 4 Zeichen Deutsch)")

    # Schritt 2: Text in Chunks schneiden
    print(f"\n✂️  Chunking: {CHUNK_CHARS} Zeichen / Chunk, {OVERLAP_CHARS} Überlappung")
    chunks = chunk_text(text, CHUNK_CHARS, OVERLAP_CHARS)
    print(f"   {len(chunks)} Chunks erzeugt")

    # Statistiken zur Qualitätskontrolle
    sizes = [len(c["text"]) for c in chunks]
    print(f"   ∅ Chunk-Größe: {sum(sizes)//len(sizes):,} Zeichen (~{sum(sizes)//len(sizes)//4} Token)")
    print(f"   Kleinster:     {min(sizes):,} Zeichen")
    print(f"   Größter:       {max(sizes):,} Zeichen")

    # Schritt 3: Chunks als JSON speichern (für Phase 3)
    out_path = CHUNKS_DIR / f"{pdf_name}_chunks.json"
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    print(f"\n💾 Gespeichert: {out_path}")

    # Vorschau der ersten beiden Chunks (zeigt die Überlappung)
    print(f"\n📋 Vorschau erster Chunk:")
    print("─" * 60)
    print(chunks[0]["text"][:300])
    print("─" * 60)
    if len(chunks) > 1:
        print(f"📋 Vorschau zweiter Chunk (mit Überlappung zum ersten):")
        print("─" * 60)
        print(chunks[1]["text"][:300])
        print("─" * 60)


if __name__ == "__main__":
    main()
