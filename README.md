# RAG-Assistent für Ausschreibungsdokumente

Ein kleiner RAG-Assistent (Retrieval-Augmented Generation), der öffentliche Ausschreibungen und Policy-/Regulierungs-PDFs verarbeitet:

- **(a)** Fragen nur auf Basis der Dokumente beantworten, mit Quellenangabe
- **(b)** Strukturierte Informationen extrahieren (z. B. Lose mit Titel, Frist, Budget, Kernanforderungen als JSON)

**Stack**: Python, Mistral (Chat + Embeddings), pypdf, numpy, pydantic, python-dotenv, streamlit

## Aufbau in Phasen

| Phase | Skript | Thema | Lernziel |
|-------|--------|-------|----------|
| 1 | `phase1_hello_llm.py` | Hello LLM | Messages/Rollen, Temperatur, Tokens |
| 2 | `phase2_load_chunk.py` | Dokument laden & chunken | Kontextfenster, Chunking-Strategie, Überlappung |
| 3 | `phase3_embed_retrieve.py` | Embedden & retrieven | Embeddings, Cosinus-Ähnlichkeit, semantische Suche |
| 4 | `phase4_rag_answer.py` | Geerdete Antwort (RAG) | Grounding, Halluzinationen verhindern, Quellenangabe |
| 5 | `phase5_structured_extraction.py` | Strukturierte Extraktion | Structured Output (JSON), Pydantic-Validierung, Retry |
| 6 | `phase6_function_calling.py` | Tool-/Function-Calling | Function Calling, Agenten-Schleife, Grounding mit Tools |
| 7a | `phase7a_eval.py` | Evaluierung | Golden Set, Trefferquote, Regressionstests |
| 7b | `phase7b_streamlit_app.py` | Streamlit-UI | Interaktive Web-App, PDF-Upload, Chat-Interface |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # dann Mistral-API-Key eintragen
```

## Verwendung

> **Hinweis:** Alle Phasen 2–7 erwarten, dass du zuerst eine PDF in `data/` abgelegt und Phase 2 ausgeführt hast. Beispiel-PDFs findest du z. B. unter `data/musterbekanntmachung_vgv.pdf`.

### Phase 1 – Hello LLM
Einfachster Einstieg: Prompt an Mistral schicken und beobachten, wie Temperatur und System-Prompt die Antwort beeinflussen.
```bash
python phase1_hello_llm.py
```

### Phase 2 – Dokument laden & chunken
PDF-Text extrahieren und in überlappende Chunks schneiden (≈600 Token, 10 % Überlappung).
```bash
python phase2_load_chunk.py data/musterbekanntmachung_vgv.pdf
```
Erzeugt: `data/chunks/musterbekanntmachung_vgv_chunks.json`

### Phase 3 – Embedden & Retrieven
Chunks mit `mistral-embed` vektorisieren und per Cosinus-Ähnlichkeit die Top-k relevanten Chunks zu einer Frage finden.
```bash
python phase3_embed_retrieve.py musterbekanntmachung_vgv
```
Erzeugt: `data/embeddings/musterbekanntmachung_vgv_embeddings.npy`

### Phase 4 – Geerdete Antwort (RAG)
Die eigentliche RAG-Pipeline: Top-k Chunks als Kontext in den Prompt einbauen und das Modell zwingen, **nur aus diesem Kontext** zu antworten — mit Quellenangabe.
```bash
# Demo-Fragen automatisch durchlaufen
python phase4_rag_answer.py musterbekanntmachung_vgv

# Interaktiver Modus
python phase4_rag_answer.py musterbekanntmachung_vgv -i
```

### Phase 5 – Strukturierte Extraktion
Strukturierte JSON-Daten aus dem Dokument extrahieren (z. B. Lose mit Titel, Frist, Budget). Nutzt `response_format={"type": "json_object"}` und Pydantic-Validierung mit Retry.
```bash
python phase5_structured_extraction.py musterbekanntmachung_vgv
```

### Phase 6 – Tool-/Function-Calling
Das Modell entscheidet selbst, wann es Werkzeuge aufruft (z. B. Suche in Chunks, Lose zählen). Einfache Agenten-Schleife mit Grounding-Regeln.
```bash
python phase6_function_calling.py musterbekanntmachung_vgv
```

### Phase 7a – Evaluierung
Golden Set gegen die RAG-Pipeline laufen lassen und Trefferquote messen. Ermöglicht A/B-Tests für Chunk-Größe, Top-k, Prompt-Formulierungen etc.
```bash
python phase7a_eval.py musterbekanntmachung_vgv
```

### Phase 7b – Streamlit-UI
Interaktive Web-App: PDF hochladen, Fragen stellen, Antworten mit Quellen anzeigen.
```bash
streamlit run phase7b_streamlit_app.py
```

## Kernkonzepte

### Grounding = Kontext im Prompt + Regeln im System-Prompt
Das wichtigste Learning des Projekts: **Grounding-Regeln allein reichen nicht.**

- **System-Prompt** enthält die Regeln (z. B. "Antworte nur aus dem Kontext", "Erfinde nichts").
- **Aber**: Wenn der Prompt keinen Dokumenten-Kontext enthält, hat das Modell nichts, aus dem es schöpfen kann. Es greift dann auf sein Allgemeinwissen zurück → Halluzinationen.
- **Lösung**: Chunks müssen **im User-Prompt** als `KONTEXT` eingebettet werden. Erst dann greifen die Grounding-Regeln.

### Temperatur
- `temperature=0`: Deterministisch, reproduzierbar — ideal für Fakten und Grounding.
- `temperature=1.0`: Kreativ, variabel — besser für Brainstorming, schlecht für Ausschreibungen.

### Chunking
- Warum: LLMs haben begrenzte Kontextfenster (z. B. 128k Token) und verlieren bei langen Dokumenten den Fokus ("lost in the middle").
- Wie: Sliding Window mit Überlappung (hier: 2500 Zeichen/Chunk, 250 Zeichen Überlappung).
- Faustregel: 1 Token ≈ 4 Zeichen im Deutschen.

### Embeddings & Retrieval
- `mistral-embed` wandelt Text in 1024-dimensionale Vektoren um.
- Cosinus-Ähnlichkeit misst, wie nah zwei Texte semantisch beieinanderliegen.
- Top-k: Die k ähnlichsten Chunks werden als Kontext für die Antwort verwendet.
