"""
Phase 7a – Evaluierung (Golden Set)
=====================================
Ein Set von Testfragen mit erwarteten Antworten (Golden Set) gegen die
RAG-Pipeline laufen lassen und die Trefferquote messen.

Hintergrund:
    Evaluierung ist der letzte Schritt, um aus "es funktioniert" ein
    "es funktioniert zuverlässig" zu machen. Ohne Eval weißt du nicht,
    ob eine Änderung (z.B. andere Chunk-Größe, anderes Top-k) das
    System besser oder schlechter macht.
    
    Golden Set: Eine handgepflegte Liste von Frage-Soll-Antwort-Paaren.
    Jede Frage wird durch die RAG-Pipeline geschickt. Dann wird geprüft,
    ob die Antwort die erwarteten Stichworte enthält.
    
    Metriken (einfach):
    - Trefferquote: Anteil der Fragen, bei denen die Antwort die
      erwarteten Stichworte enthält (Recall-orientiert)
    - Die Scores sind bewusst einfach gehalten. In Produktion würde man
      LLM-as-Judge oder semantische Ähnlichkeit der Antworten nutzen.
    
    Mit diesem Eval kannst du später experimentieren:
    - Chunk-Größe variieren (1500 vs 2500 vs 4000 Zeichen)
    - Top-k variieren (3 vs 5 vs 7)
    - Temperatur variieren
    - Prompt-Formulierungen vergleichen
    → Welche Kombination bringt die höchste Trefferquote?

Verwendung:
    python phase7a_eval.py musterbekanntmachung_vgv
"""

import sys
import json
import os
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# ─── Pfade & Konfiguration ───
DATA_DIR = Path("data")
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
TOP_K = 3
MODEL = "mistral-small-latest"
TEMPERATURE = 0

# ─── Grounding-Prompt (wie Phase 4) ───
RAG_SYSTEM_PROMPT = """Du bist ein Assistent für öffentliche Ausschreibungen.
Du beantwortest Fragen AUSSCHLIESSLICH auf Basis des unten angegebenen Kontexts.

Wichtige Regeln:
1. Antworte NUR mit Informationen, die im Kontext stehen.
2. Wenn die Antwort nicht im Kontext zu finden ist, sage klar:
   "Diese Information steht nicht in den vorliegenden Dokumenten."
3. Erfinde keine Informationen hinzu. Kein Allgemeinwissen verwenden.
4. Antworte präzise und auf Deutsch."""


# ═══════════════════════════════════════════════════════════════
# Golden Set: Testfragen + erwartete Stichworte
# ═══════════════════════════════════════════════════════════════
# 
# Jeder Eintrag:
#   question: Die Frage an das RAG-System
#   expected_keywords: Liste von Stichworten, die in einer korrekten
#                      Antwort vorkommen MÜSSEN (werden als Substring-
#                      Match geprüft, case-insensitive).
#   must_not_contain: Liste von Phrasen, die NICHT vorkommen dürfen
#                     (z.B. Halluzinations-Indikatoren)

GOLDEN_SET = [
    {
        "question": "Welche Planungsleistungen werden in dieser Ausschreibung vergeben?",
        "expected_keywords": ["Objektplanung", "Gebäude", "HOAI", "Tragwerksplanung"],
        "must_not_contain": [],
    },
    {
        "question": "In welcher Form wird der Auftrag vergeben (Stufenvertrag)?",
        "expected_keywords": ["Stufenvertrag", "Stufe 1", "Lph"],
        "must_not_contain": [],
    },
    {
        "question": "Wer darf sich bewerben? Welche Berufsgruppen?",
        "expected_keywords": ["Architekt", "Berechtigung", "Berufsbezeichnung"],
        "must_not_contain": [],
    },
    {
        "question": "Wie hoch ist das Budget für dieses Projekt?",
        "expected_keywords": [],  # Sollte nichts finden
        "must_not_contain": [],   # Keine Halluzination
        "expect_no_answer": True,  # Speziell: Modell soll zugeben, dass Info fehlt
    },
    {
        "question": "Welche Eignungsnachweise müssen eingereicht werden?",
        "expected_keywords": ["Referenz", "Berufshaftpflicht", "Eigenerklärung"],
        "must_not_contain": [],
    },
    {
        "question": "Nach welcher Honorarordnung wird abgerechnet?",
        "expected_keywords": ["HOAI", "2021"],
        "must_not_contain": [],
    },
    {
        "question": "Welche Fristen gibt es für Nachprüfungsanträge?",
        "expected_keywords": ["10", "Kalendertage", "15", "30"],
        "must_not_contain": [],
    },
    {
        "question": "Welche Programmiersprache wird für die Implementierung gefordert?",
        "expected_keywords": [],
        "must_not_contain": ["Python", "Java", "C++", "Programmiersprache"],
        "expect_no_answer": True,  # Sollte nicht drinstehen
    },
    {
        "question": "Welche Versicherung muss nachgewiesen werden?",
        "expected_keywords": ["Berufshaftpflichtversicherung", "Sach", "Vermögen"],
        "must_not_contain": [],
    },
    {
        "question": "Nenne alle Leistungsphasen der Objektplanung Gebäude.",
        "expected_keywords": ["Lph 1-9", "HOAI", "§ 34"],
        "must_not_contain": [],
    },
]


# ─── RAG-Funktionen (aus früheren Phasen) ───

def load_chunks(pdf_name: str) -> list[dict]:
    path = CHUNKS_DIR / f"{pdf_name}_chunks.json"
    if not path.exists():
        print(f"❗ Keine Chunks für '{pdf_name}'")
        sys.exit(1)
    return json.loads(path.read_text())


def load_embeddings(pdf_name: str) -> np.ndarray:
    emb_path = EMBEDDINGS_DIR / f"{pdf_name}_embeddings.npy"
    if not emb_path.exists():
        print(f"❗ Keine Embeddings für '{pdf_name}'. Erst Phase 3 ausführen.")
        sys.exit(1)
    return np.load(emb_path)


def retrieve_chunks(query: str, chunk_vecs: np.ndarray, chunks: list[dict], top_k: int):
    response = client.embeddings.create(model="mistral-embed", inputs=[query])
    query_vec = np.array(response.data[0].embedding, dtype=np.float64)
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    scores = np.dot(chunk_norms, query_norm)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), chunks[i]) for i in top_indices]


def rag_answer(question: str, chunks: list[dict], chunk_vecs: np.ndarray) -> str:
    """RAG-Pipeline: Retrieval → Prompt → Antwort."""
    retrieved = retrieve_chunks(question, chunk_vecs, chunks, top_k=TOP_K)

    context_blocks = []
    for score, chunk in retrieved:
        context_blocks.append(
            f"[Chunk {chunk['chunk_id']} | Score: {score:.3f}]\n{chunk['text']}"
        )
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
        max_tokens=400,
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════
# Evaluierung
# ═══════════════════════════════════════════════════════════════

def evaluate_answer(answer: str, test_case: dict) -> dict:
    """
    Prüft eine Antwort gegen die erwarteten Stichworte.
    
    Returns dict mit:
        - passed: bool (alle keywords gefunden? keine verbotenen Phrasen?)
        - keywords_found: welche der expected_keywords wurden gefunden
        - keywords_missed: welche fehlen
        - forbidden_found: welche verbotenen Phrasen kamen vor
        - no_answer_correct: speziell für expect_no_answer-Tests
    """
    answer_lower = answer.lower()
    result = {
        "passed": True,
        "keywords_found": [],
        "keywords_missed": [],
        "forbidden_found": [],
        "no_answer_correct": None,
    }

    # Erwartete Stichworte prüfen
    for kw in test_case.get("expected_keywords", []):
        if kw.lower() in answer_lower:
            result["keywords_found"].append(kw)
        else:
            result["keywords_missed"].append(kw)

    # Verbotene Phrasen prüfen
    for phrase in test_case.get("must_not_contain", []):
        if phrase.lower() in answer_lower:
            result["forbidden_found"].append(phrase)

    # Spezialfall: Antwort soll "steht nicht drin" sein
    if test_case.get("expect_no_answer"):
        no_answer_phrases = [
            "steht nicht", "nicht in den vorliegenden",
            "keine information", "nicht gefunden", "nicht enthalten"
        ]
        result["no_answer_correct"] = any(
            p in answer_lower for p in no_answer_phrases
        )

    # Gesamtergebnis
    if result["keywords_missed"]:
        result["passed"] = False
    if result["forbidden_found"]:
        result["passed"] = False
    if test_case.get("expect_no_answer") and not result["no_answer_correct"]:
        result["passed"] = False

    return result


def run_eval(pdf_name: str):
    """Führt die komplette Evaluierung durch."""
    print(f"\n{'=' * 60}")
    print(f"📊 Phase 7a: Evaluierung (Golden Set)")
    print(f"📄 Dokument: {pdf_name}")
    print(f"📋 {len(GOLDEN_SET)} Testfragen")
    print(f"{'=' * 60}")

    chunks = load_chunks(pdf_name)
    chunk_vecs = load_embeddings(pdf_name)

    results = []
    start_time = time.time()

    for i, test_case in enumerate(GOLDEN_SET, 1):
        question = test_case["question"]
        print(f"\n{'─' * 60}")
        print(f"📝 Test {i}/{len(GOLDEN_SET)}: {question}")

        answer = rag_answer(question, chunks, chunk_vecs)
        print(f"   Antwort: {answer[:120]}...")
        
        eval_result = evaluate_answer(answer, test_case)

        if test_case.get("expect_no_answer"):
            status = "✅" if eval_result["no_answer_correct"] else "❌"
            print(f"   {status} No-Answer-Test: {'richtig erkannt' if eval_result['no_answer_correct'] else 'hat halluziniert!'}")
        else:
            status = "✅" if eval_result["passed"] else "❌"
            if eval_result["keywords_found"]:
                print(f"   ✅ Gefunden: {eval_result['keywords_found']}")
            if eval_result["keywords_missed"]:
                print(f"   ❌ Fehlt: {eval_result['keywords_missed']}")
            if eval_result["forbidden_found"]:
                print(f"   🚫 Verboten: {eval_result['forbidden_found']}")

        results.append({
            "question": question,
            "passed": eval_result["passed"],
            "keywords_found": eval_result["keywords_found"],
            "keywords_missed": eval_result["keywords_missed"],
            "answer_preview": answer[:200],
        })

    elapsed = time.time() - start_time

    # ─── Gesamtergebnis ───
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    quote = (passed / total) * 100 if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"📊 ERGEBNIS")
    print(f"{'=' * 60}")
    print(f"   Trefferquote: {passed}/{total} ({quote:.0f}%)")
    print(f"   Dauer:        {elapsed:.1f}s")
    print(f"   ⌀ pro Frage:  {elapsed/total:.1f}s")

    # Detail-Übersicht
    print(f"\n📋 Details:")
    for i, r in enumerate(results, 1):
        icon = "✅" if r["passed"] else "❌"
        print(f"   {icon} Test {i}: {r['question'][:60]}...")
        if not r["passed"] and not GOLDEN_SET[i-1].get("expect_no_answer"):
            print(f"      Fehlende Keywords: {r['keywords_missed']}")

    # Empfehlungen
    print(f"\n💡 Optimierungsmöglichkeiten:")
    print(f"   - Chunk-Größe variieren (1500/2500/4000 Zeichen)")
    print(f"   - Top-k variieren (3/5/7)")
    print(f"   - System-Prompt umformulieren")
    print(f"   - Anderes Embedding-Modell testen")
    print(f"   - Bei jedem Lauf diese Eval wiederholen → Effekt messen")

    return results


def main():
    if len(sys.argv) < 2:
        print("❗ Bitte PDF-Namen angeben (ohne .pdf):")
        print("   python phase7a_eval.py musterbekanntmachung_vgv")
        sys.exit(1)

    pdf_name = Path(sys.argv[1]).stem
    run_eval(pdf_name)


if __name__ == "__main__":
    main()
