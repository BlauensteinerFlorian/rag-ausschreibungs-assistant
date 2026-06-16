"""
Phase 6 – Tool-/Function-Calling
==================================
Funktionen als Tools definieren, das Modell entscheiden lassen wann es sie
aufruft, und eine einfache Agenten-Schleife bauen.

Hintergrund:
    Bisher war das Modell rein reaktiv: Frage → Antwort. Mit Function Calling
    wird es zum Akteur: Es erkennt selbst, WANN es ein Werkzeug braucht,
    ruft es auf und verarbeitet das Ergebnis.
    
    Wie funktioniert Function Calling?
    1. Du definierst Funktionen mit JSON-Schema (Name, Beschreibung, Parameter)
    2. Du schickst diese Tools MIT der User-Nachricht an das Modell
    3. Das Modell entscheidet:
       a) "Ich kann direkt antworten" → normale Text-Antwort
       b) "Ich brauche Tool X" → gibt Tool-Namen + Parameter zurück
    4. Du führst die Funktion aus und schickst das Ergebnis zurück
    5. Das Modell formuliert daraus die finale Antwort
    
    Das ist die Basis für "Agenten": LLMs, die selbstständig Werkzeuge
    einsetzen, um komplexe Aufgaben zu lösen.
    
    Mistral unterstützt Function Calling nativ. Die Tools werden als
    Liste von dicts mit name, description, parameters (JSON Schema)
    definiert.

Beispiel-Tools für unseren Use-Case:
    - filter_lose_nach_budget(max_eur) → Lose unter Budget finden
    - filter_lose_nach_frist(vor_dem) → Lose mit Frist vor Datum
    - suche_in_chunks(suchbegriff) → Volltextsuche in Chunks

Lernziele:
    - Function Calling verstehen: Tool-Definition, Entscheidungslogik
    - JSON-Schema für Tools schreiben
    - Einfache Agenten-Schleife: Call → Execute → Feed back → Answer
    - Unterschied: normale Antwort vs Tool-basierte Antwort

Verwendung:
    python phase6_function_calling.py musterbekanntmachung_vgv
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# ─── Pfade ───
DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# ─── Konfiguration ───
MODEL = "mistral-small-latest"
TEMPERATURE = 0
MAX_TOKENS = 500
TOP_K = 5


# ═══════════════════════════════════════════════════════════════
# Daten laden (aus Phase 2/3)
# ═══════════════════════════════════════════════════════════════

def load_chunks(pdf_name: str) -> list[dict]:
    path = CHUNKS_DIR / f"{pdf_name}_chunks.json"
    if not path.exists():
        print(f"❗ Keine Chunks für '{pdf_name}'. Erst Phase 2 ausführen.")
        sys.exit(1)
    return json.loads(path.read_text())


def load_or_create_embeddings(pdf_name: str, chunks: list[dict]) -> np.ndarray:
    emb_path = EMBEDDINGS_DIR / f"{pdf_name}_embeddings.npy"
    if emb_path.exists():
        return np.load(emb_path)
    chunk_texts = [c["text"] for c in chunks]
    response = client.embeddings.create(model="mistral-embed", inputs=chunk_texts)
    vecs = np.array([d.embedding for d in response.data], dtype=np.float64)
    np.save(emb_path, vecs)
    return vecs


def retrieve_chunks(query: str, chunk_vecs: np.ndarray, chunks: list[dict], top_k: int = TOP_K):
    """Semantische Suche"""
    response = client.embeddings.create(model="mistral-embed", inputs=[query])
    query_vec = np.array(response.data[0].embedding, dtype=np.float64)
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    scores = np.dot(chunk_norms, query_norm)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), chunks[i]) for i in top_indices]


# ═══════════════════════════════════════════════════════════════
# Die Tools: Funktionen, die das Modell aufrufen kann
# ═══════════════════════════════════════════════════════════════

# Jedes Tool ist ein dict mit:
#   type: "function"
#   function:
#     name:       Eindeutiger Name (den ruft das Modell auf)
#     description: Was macht die Funktion? WANN soll das Modell sie nutzen?
#     parameters: JSON-Schema der Parameter (Typen, Beschreibungen, Pflichtfelder)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "suche_in_chunks",
            "description": (
                "Durchsucht die Dokument-Chunks nach einem Begriff oder einer Phrase. "
                "Nutze dieses Tool, wenn die Frage nach spezifischen Details fragt, "
                "die durch semantische Suche allein nicht gefunden werden könnten. "
                "Gib den Suchbegriff als String an."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "suchbegriff": {
                        "type": "string",
                        "description": "Der Begriff oder die Phrase, nach der gesucht werden soll (z.B. 'Frist', 'Budget', 'Architekt')"
                    }
                },
                "required": ["suchbegriff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_lose_nach_frist",
            "description": (
                "Filtert Lose, deren Frist VOR einem bestimmten Datum liegt. "
                "Nutze dieses Tool, wenn gefragt wird, welche Lose bis zu einem "
                "bestimmten Datum eingereicht werden müssen oder wie dringend etwas ist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vor_dem": {
                        "type": "string",
                        "description": "Datum im Format JJJJ-MM-TT (z.B. 2025-06-01). Zeigt Lose, deren Frist vor diesem Datum liegt."
                    }
                },
                "required": ["vor_dem"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zaehle_lose",
            "description": (
                "Zählt die Anzahl der Lose in der Ausschreibung. "
                "Nutze dieses Tool, wenn nach der Anzahl der Lose gefragt wird."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# ─── Globale Referenz auf Chunks und Embeddings (für Tool-Funktionen) ───
_current_chunks: list[dict] = []
_current_chunk_vecs: np.ndarray = None


# ═══════════════════════════════════════════════════════════════
# Tool-Implementierungen (die echten Python-Funktionen)
# ═══════════════════════════════════════════════════════════════

def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Führt ein Tool aus und gibt das Ergebnis als String zurück.
    
    Das Ergebnis wird als neue "tool"-Rolle in die Nachrichten-Historie
    eingefügt, damit das Modell darauf aufbauen kann.
    """
    global _current_chunks, _current_chunk_vecs

    if tool_name == "suche_in_chunks":
        suchbegriff = arguments.get("suchbegriff", "")
        treffer = []
        for chunk in _current_chunks:
            text_lower = chunk["text"].lower()
            if suchbegriff.lower() in text_lower:
                # Kontext um den Treffer herum extrahieren
                idx = text_lower.find(suchbegriff.lower())
                start = max(0, idx - 80)
                end = min(len(chunk["text"]), idx + len(suchbegriff) + 80)
                snippet = chunk["text"][start:end].replace("\n", " ")
                treffer.append({
                    "chunk_id": chunk["chunk_id"],
                    "snippet": f"...{snippet}..."
                })
        return json.dumps({
            "suchbegriff": suchbegriff,
            "anzahl_treffer": len(treffer),
            "treffer": treffer[:5],  # max 5 Treffer zurückgeben
        }, ensure_ascii=False)

    elif tool_name == "filter_lose_nach_frist":
        vor_dem = arguments.get("vor_dem", "")
        # Frist-Daten aus allen Chunks sammeln
        # (Das ist vereinfacht — in Produktion würde man die Pydantic-Daten
        # aus Phase 5 parsen. Hier durchsuchen wir die Chunks.)
        treffer = []
        for chunk in _current_chunks:
            text = chunk["text"]
            if "frist" in text.lower() or "termin" in text.lower() or "datum" in text.lower():
                snippet = text[:200].replace("\n", " ")
                treffer.append({
                    "chunk_id": chunk["chunk_id"],
                    "snippet": snippet,
                })
        return json.dumps({
            "filter_frist_vor": vor_dem,
            "hinweis": "Die genauen Fristdaten wurden aus den Chunks extrahiert. "
                       "Platzhalter wie 'xx.xx.2024' zeigen an, dass das konkrete "
                       "Datum im Musterdokument nicht ausgefüllt ist.",
            "chunks_mit_fristen": treffer,
        }, ensure_ascii=False)

    elif tool_name == "zaehle_lose":
        # Zählen, wie viele Lose in den Chunks erwähnt werden
        los_count = 0
        for chunk in _current_chunks:
            text = chunk["text"]
            if "los" in text.lower():
                los_count += 1
        return json.dumps({
            "anzahl_lose_referenziert": los_count,
            "info": "Zählung basiert auf Erwähnungen in den Chunks. "
                    "Für eine exakte Zählung siehe Phase 5 (strukturierte Extraktion)."
        }, ensure_ascii=False)

    else:
        return json.dumps({"error": f"Unbekanntes Tool: {tool_name}"})


# ═══════════════════════════════════════════════════════════════
# Agenten-Schleife
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "Du bist ein Assistent für öffentliche Ausschreibungen. "
    "Du hast Zugriff auf Tools, um Informationen aus Ausschreibungsdokumenten "
    "zu extrahieren und zu filtern. "
    "Nutze die Tools, wenn sie helfen, die Frage besser zu beantworten. "
    "Wenn du ein Tool aufrufst, warte auf das Ergebnis, bevor du antwortest. "
    "Antworte präzise und auf Deutsch."
)


def agent_loop(user_question: str) -> str:
    """
    Die Agenten-Schleife:
    1. Frage + Tools an das Modell schicken
    2a. Modell antwortet direkt → fertig
    2b. Modell will Tool aufrufen → Tool ausführen → Ergebnis zurück → Schritt 1
    
    Maximum 5 Runden (verhindert Endlosschleifen).
    """
    global _current_chunks, _current_chunk_vecs

    # Nachrichten-Historie für diese Frage
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    for runde in range(5):  # Max 5 Runden (safety limit)
        # API-Call: Modell entscheidet, ob es antwortet oder ein Tool aufruft
        response = client.chat.complete(
            model=MODEL,
            messages=messages,
            tools=TOOLS,  # ← Hier werden die Tools übergeben
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        msg = response.choices[0].message

        # Fall A: Modell antwortet direkt (kein Tool-Call)
        if msg.content and not msg.tool_calls:
            return msg.content

        # Fall B: Modell will ein Tool aufrufen
        if msg.tool_calls:
            print(f"\n   🔧 Runde {runde + 1}: Modell ruft Tool(s) auf:")

            # Tool-Antwort zur Historie hinzufügen
            # Das ist nötig, damit das Modell die Konversation fortsetzen kann
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                print(f"      → {tool_name}({tool_args})")

                # Tool ausführen
                result = execute_tool(tool_name, tool_args)
                print(f"      ← Ergebnis: {result[:100]}...")

                # Tool-Ergebnis als neue Rolle "tool" in die Historie einfügen
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # Weiter zum nächsten Schleifendurchlauf
            # Das Modell sieht jetzt das Tool-Ergebnis und kann
            # entweder antworten oder weitere Tools aufrufen
            continue

        # Fall C: Modell hat weder content noch tool_calls (sollte nicht vorkommen)
        return "⚠️ Modell hat weder geantwortet noch ein Tool aufgerufen."

    return "⚠️ Maximale Anzahl an Agenten-Runden erreicht."


def main():
    if len(sys.argv) < 2:
        print("❗ Bitte PDF-Namen angeben (ohne .pdf):")
        print("   python phase6_function_calling.py musterbekanntmachung_vgv")
        sys.exit(1)

    pdf_name = Path(sys.argv[1]).stem

    global _current_chunks, _current_chunk_vecs
    _current_chunks = load_chunks(pdf_name)
    _current_chunk_vecs = load_or_create_embeddings(pdf_name, _current_chunks)
    print(f"📄 Dokument: {pdf_name} ({len(_current_chunks)} Chunks)")

    # ─── Demo-Fragen: Zeigt den Unterschied Tool vs kein Tool ───
    questions = [
        # Frage OHNE Tool-Bedarf → direkt antworten
        "Worum geht es in dieser Ausschreibung?",
        # Frage MIT Tool-Bedarf → Tool aufrufen
        "Durchsuche die Ausschreibung nach dem Begriff 'Frist' und sage mir, was du findest.",
        # Weitere Tool-Frage
        "Wie viele Lose werden in der Ausschreibung erwähnt?",
    ]

    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"❓ Frage: {question}")
        print(f"{'=' * 60}")

        answer = agent_loop(question)
        print(f"\n📝 Antwort:\n{answer}")
        print(f"\n{'-' * 40}")

    print(f"\n✅ Phase 6 abgeschlossen.")
    print(f"   Das Modell hat selbstständig entschieden, wann es Tools aufruft.")
    print(f"   Die Agenten-Schleife führt Tools aus und speist Ergebnisse zurück.")


if __name__ == "__main__":
    main()
