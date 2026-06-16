"""
Phase 2 – Dokument laden & chunken
==================================
PDF-Text mit pypdf extrahieren und in überlappende Chunks schneiden
(ca. 600 Token ≈ 2.500 Zeichen, 10 % Überlappung).
Lernziele: Kontextfenster-Grenzen verstehen, warum Chunking nötig ist.

Verwendung:
    python phase2_load_chunk.py <pfad-zur-pdf>
"""

import sys
import json
from pathlib import Path
from pypdf import PdfReader

# ─── Konfiguration ───
CHUNK_CHARS = 2500      # ≈ 600 Token bei deutschem Text
OVERLAP_CHARS = 250     # 10 % von 2500
DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def extract_text(pdf_path: Path) -> str:
    """Liest alle Seiten eines PDFs und gibt den Text als einen String zurück."""
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
    Schneidet Text in überlappende Chunks.
    Returns: Liste von dicts mit chunk_id, text, start_char, end_char
    """
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start_char": start,
            "end_char": end,
        })
        chunk_id += 1
        if end == len(text):
            break
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

    pdf_name = pdf_path.stem

    print(f"📄 Lade: {pdf_path.name}")
    text = extract_text(pdf_path)
    print(f"   Gesamtzeichen: {len(text):,}")
    print(f"   ≈ Token:      {len(text) // 4:,}  (Faustregel: 1 Token ≈ 4 Zeichen Deutsch)")

    print(f"\n✂️  Chunking: {CHUNK_CHARS} Zeichen / Chunk, {OVERLAP_CHARS} Überlappung")
    chunks = chunk_text(text, CHUNK_CHARS, OVERLAP_CHARS)
    print(f"   {len(chunks)} Chunks erzeugt")

    # Statistiken
    sizes = [len(c["text"]) for c in chunks]
    print(f"   ∅ Chunk-Größe: {sum(sizes)//len(sizes):,} Zeichen (~{sum(sizes)//len(sizes)//4} Token)")
    print(f"   Kleinster:     {min(sizes):,} Zeichen")
    print(f"   Größter:       {max(sizes):,} Zeichen")

    # Speichern als JSON
    out_path = CHUNKS_DIR / f"{pdf_name}_chunks.json"
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    print(f"\n💾 Gespeichert: {out_path}")

    # Vorschau
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
