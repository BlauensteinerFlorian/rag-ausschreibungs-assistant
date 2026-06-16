"""
Phase 5 – Strukturierte Extraktion
====================================
Die RAG-Pipeline aus Phase 4 nutzen, um strukturierte JSON-Daten aus
Ausschreibungsdokumenten zu extrahieren — z. B. Lose mit Titel, Frist,
Budget und Kernanforderungen.

Hintergrund:
    Bisher (Phase 4) hat das Modell Freitext-Antworten gegeben. Für die
    automatisierte Weiterverarbeitung (z. B. Los-Screening, Vergleich,
    Datenbank-Import) brauchen wir strukturierte, maschinenlesbare Daten.
    
    Structured Output bedeutet: Das Modell wird gezwungen, gültiges JSON
    nach einem vorgegebenen Schema zu liefern.
    
    Mistral unterstützt `response_format={"type": "json_object"}`.
    Damit garantiert das Modell syntaktisch gültiges JSON.
    
    ABER: Syntaktisch gültig ≠ semantisch korrekt. Das Modell könnte
    Felder vergessen, falsche Typen verwenden oder Zusatzfelder erfinden.
    Deshalb: Pydantic-Validierung als zweite Sicherheitsschicht.
    
    Pydantic prüft:
    - Pflichtfelder vorhanden? (titel muss ein str sein)
    - Optionale Felder vom richtigen Typ? (budget: str | None)
    - Listen-Elemente korrekt? (anforderungen: list[str])
    
    Bei Validierungsfehlern: Retry — einmal neu anfragen mit deutlichem
    Hinweis auf den Fehler. Meist klappt's dann beim zweiten Versuch.

Architektur:
    1. Pydantic-Modell definieren (Was wollen wir extrahieren?)
    2. RAG-Pipeline: Kontext + Frage + JSON-Schema in den Prompt
    3. response_format={"type": "json_object"} setzen
    4. Antwort parsen (json.loads)
    5. Mit Pydantic validieren
    6. Bei Fehler: Retry mit Fehlerbeschreibung
    7. Validiertes Objekt ausgeben

Lernziele:
    - Structured Output: JSON statt Freitext
    - Schema-Definition und Prompt-Engineering für strukturierte Antworten
    - Pydantic als Validierungsschicht
    - Retry-Strategie bei fehlerhaften LLM-Outputs
    - Zuverlässigkeit in LLM-basierten Systemen

Verwendung:
    python phase5_structured_extraction.py musterbekanntmachung_vgv
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from mistralai.client import Mistral
from pydantic import BaseModel, ValidationError

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# ─── Pfade ───
DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# ─── Konfiguration ───
TOP_K = 5           # Mehr Chunks für Extraktion (braucht breiteren Kontext)
MODEL = "mistral-small-latest"
TEMPERATURE = 0      # Deterministisch für konsistente Struktur
MAX_TOKENS = 1000    # JSON braucht mehr Platz


# ═══════════════════════════════════════════════════════════════
# Pydantic-Modelle: Das Schema, das wir vom LLM erwarten
# ═══════════════════════════════════════════════════════════════

class Los(BaseModel):
    """
    Ein einzelnes Los einer Ausschreibung.
    
    - titel: Pflichtfeld, immer ein String (Bezeichnung des Loses)
    - frist: Optional (nicht jede Ausschreibung hat explizite Fristen)
    - budget: Optional (geschätztes Budget, oft nicht öffentlich)
    - anforderungen: Liste der Kernanforderungen, mindestens leere Liste
    
    Die Typ-Annotationen sind gleichzeitig die Validierungsregeln.
    Pydantic prüft sie automatisch beim Erstellen eines Los-Objekts.
    """
    titel: str
    frist: str | None = None
    budget: str | None = None
    kernanforderungen: list[str] = []


class Ausschreibung(BaseModel):
    """
    Die komplette extrahierte Ausschreibung.
    
    - lose: Liste von Los-Objekten (muss mindestens ein Element enthalten,
      oder leer sein wenn keine Lose gefunden wurden)
    """
    lose: list[Los]


# ─── Der Extraktions-Prompt ─────────────────────────────────────
# Hier definieren wir im Prompt selbst das gewünschte JSON-Schema.
# Das Modell bekommt Kontext + Schema + Frage und soll JSON liefern.
EXTRACTION_SYSTEM_PROMPT = """Du bist ein Assistent, der strukturierte Informationen aus Ausschreibungsdokumenten extrahiert.

Du erhältst Auszüge aus einem Ausschreibungsdokument (Kontext).
Extrahiere daraus alle Lose mit folgenden Feldern:
- titel: die genaue Bezeichnung des Loses
- frist: Angebotsfrist oder Einreichungsfrist (wenn angegeben, sonst null)
- budget: geschätztes Budget (wenn angegeben, sonst null)  
- kernanforderungen: Liste der wichtigsten Anforderungen/Leistungen

ANTWORTE AUSSCHLIESSLICH mit gültigem JSON im folgenden Format:
{
  "lose": [
    {
      "titel": "string",
      "frist": "string oder null",
      "budget": "string oder null",
      "kernanforderungen": ["string", ...]
    }
  ]
}

Regeln:
1. Wenn ein Feld nicht im Kontext steht, setze es auf null (bzw. leere Liste bei kernanforderungen).
2. Erfinde NICHTS. Nur was im Kontext steht.
3. Wenn keine Lose gefunden werden, gib {"lose": []} zurück.
4. Antworte NUR mit dem JSON-Objekt, kein weiterer Text."""


# ═══════════════════════════════════════════════════════════════
# Hilfsfunktionen (aus Phase 3/4 übernommen)
# ═══════════════════════════════════════════════════════════════

def load_chunks(pdf_name: str) -> list[dict]:
    """Chunks aus Phase 2 laden."""
    path = CHUNKS_DIR / f"{pdf_name}_chunks.json"
    if not path.exists():
        print(f"❗ Keine Chunks für '{pdf_name}'. Erst Phase 2 ausführen.")
        sys.exit(1)
    return json.loads(path.read_text())


def load_or_create_embeddings(pdf_name: str, chunks: list[dict]) -> np.ndarray:
    """Embeddings aus Cache laden oder neu erstellen."""
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
    """Semantische Suche: Findet die ähnlichsten Chunks zum Query."""
    response = client.embeddings.create(model="mistral-embed", inputs=[query])
    query_vec = np.array(response.data[0].embedding, dtype=np.float64)
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    scores = np.dot(chunk_norms, query_norm)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), chunks[i]) for i in top_indices]


def build_context(retrieved: list[tuple[float, dict]]) -> str:
    """Baut den Kontext-Text aus den Top-k Chunks."""
    blocks = []
    for score, chunk in retrieved:
        blocks.append(
            f"[Chunk {chunk['chunk_id']} | Score: {score:.3f}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(blocks)


# ═══════════════════════════════════════════════════════════════
# Kernlogik: Extrahieren mit Retry
# ═══════════════════════════════════════════════════════════════

def extract_with_retry(pdf_name: str) -> Ausschreibung:
    """
    Extrahiert Lose aus der Ausschreibung, mit Validierung und Retry.
    
    Ablauf:
    1. RAG-Pipeline: Chunks → Embeddings → Retrieval → Kontext
    2. LLM-Call mit response_format json_object → JSON-String
    3. json.loads → Python dict
    4. Pydantic-Validierung (Ausschreibung(**dict))
    5a. Erfolg → validiertes Objekt zurück
    5b. Fehler → Retry (max. 1x) mit Fehlerbeschreibung im Prompt
    """
    # Schritt 1: Kontext besorgen
    chunks = load_chunks(pdf_name)
    chunk_vecs = load_or_create_embeddings(pdf_name, chunks)

    extraction_query = (
        "Extrahiere alle Lose mit Titel, Frist, Budget und Kernanforderungen "
        "aus dieser Ausschreibung."
    )
    retrieved = retrieve_chunks(extraction_query, chunk_vecs, chunks)
    context = build_context(retrieved)

    print(f"\n📊 {len(retrieved)} Chunks als Kontext geladen")

    # Schritt 2-5: LLM-Call + Validierung + ggf. Retry
    for attempt in range(2):  # Max 2 Versuche (Erstversuch + 1 Retry)
        print(f"\n{'─' * 60}")
        print(f"🤖 Versuch {attempt + 1}/2: Extraktion mit {MODEL} ...")

        # Prompt zusammenbauen — beim Retry den Fehler mitgeben
        if attempt == 0:
            user_prompt = f"""Kontext aus dem Ausschreibungsdokument:
---
{context}
---

Extrahiere daraus die Lose im JSON-Format."""
        else:
            # Retry: Fehler mitteilen, um das Modell zu korrigieren
            user_prompt = f"""Kontext aus dem Ausschreibungsdokument:
---
{context}
---

ACHTUNG: Der vorherige Versuch hat einen JSON-Fehler produziert:
{last_error}

Extrahiere ERNEUT die Lose im JSON-Format. Stelle sicher, dass das JSON
exakt dem Schema entspricht und alle Pflichtfelder korrekt gefüllt sind."""

        # API-Call mit JSON-Modus
        # response_format={"type": "json_object"} garantiert syntaktisch
        # gültiges JSON (Mistral kümmert sich um die Klammern/Kommas)
        response = client.chat.complete(
            model=MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},  # ← Structured Output
        )

        raw_json = response.choices[0].message.content
        print(f"   Tokens: {response.usage}")

        # Roh-JSON parsen
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            last_error = str(e)
            print(f"   ❌ JSON-Syntaxfehler: {e}")
            continue  # Retry

        # Pydantic-Validierung
        try:
            ausschreibung = Ausschreibung(**data)
            print(f"   ✅ Pydantic-Validierung erfolgreich!")
            print(f"   📊 {len(ausschreibung.lose)} Los(e) extrahiert")
            return ausschreibung

        except ValidationError as e:
            # Pydantic sagt genau, WAS falsch ist — das geben wir beim
            # Retry als Kontext mit, damit das Modell es korrigieren kann
            last_error = str(e)
            print(f"   ❌ Pydantic-Validierung fehlgeschlagen:")
            print(f"   {e}")
            continue  # Retry

    # Wenn wir hier landen, haben beide Versuche nicht geklappt
    print("\n⚠️  Extraktion nach 2 Versuchen fehlgeschlagen.")
    print("   Roh-JSON des letzten Versuchs:")
    print(f"   {raw_json[:500]}")
    return Ausschreibung(lose=[])


def main():
    if len(sys.argv) < 2:
        print("❗ Bitte PDF-Namen angeben (ohne .pdf):")
        print("   python phase5_structured_extraction.py musterbekanntmachung_vgv")
        sys.exit(1)

    pdf_name = Path(sys.argv[1]).stem

    print(f"\n{'=' * 60}")
    print(f"📋 Phase 5: Strukturierte Extraktion")
    print(f"📄 Dokument: {pdf_name}")
    print(f"🤖 Modell:   {MODEL} | temperature={TEMPERATURE}")
    print(f"{'=' * 60}")

    # Extraktion durchführen
    ausschreibung = extract_with_retry(pdf_name)

    # Ergebnis ausgeben
    if ausschreibung.lose:
        print(f"\n{'=' * 60}")
        print(f"📦 Extrahiertes Ergebnis:")
        print(f"{'=' * 60}")

        # Als formatiertes JSON ausgeben
        # model_dump() konvertiert das Pydantic-Objekt in ein dict
        # exclude_none=True lässt None-Felder weg (sieht schöner aus)
        output = [los.model_dump(exclude_none=False) for los in ausschreibung.lose]
        print(json.dumps({"lose": output}, ensure_ascii=False, indent=2))

        # Zusammenfassung pro Los
        print(f"\n📊 Zusammenfassung:")
        for i, los in enumerate(ausschreibung.lose, 1):
            print(f"   Los {i}: {los.titel}")
            if los.frist:
                print(f"      Frist: {los.frist}")
            if los.budget:
                print(f"      Budget: {los.budget}")
            if los.kernanforderungen:
                print(f"      Anforderungen: {len(los.kernanforderungen)}")
                for req in los.kernanforderungen[:3]:
                    print(f"        - {req}")
                if len(los.kernanforderungen) > 3:
                    print(f"        ... und {len(los.kernanforderungen) - 3} weitere")
    else:
        print("\n⚠️  Keine Lose extrahiert.")

    print(f"\n✅ Phase 5 abgeschlossen.")
    print(f"   Strukturierte JSON-Daten aus der Ausschreibung extrahiert.")
    print(f"   Pydantic garantiert korrekte Struktur und Typen.")


if __name__ == "__main__":
    main()
