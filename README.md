# RAG-Assistent für Ausschreibungsdokumente

Ein kleiner RAG-Assistent (Retrieval-Augmented Generation), der öffentliche Ausschreibungen und Policy-/Regulierungs-PDFs verarbeitet:

- **(a)** Fragen nur auf Basis der Dokumente beantworten, mit Quellenangabe
- **(b)** Strukturierte Informationen extrahieren (z. B. Lose mit Titel, Frist, Budget, Kernanforderungen als JSON)

**Stack**: Python, Mistral (Chat + Embeddings), pypdf, numpy, pydantic, python-dotenv

## Aufbau in Phasen

| Phase | Thema | Lernziel |
|-------|-------|----------|
| 1 | Hello LLM | Messages/Rollen, Temperatur, Tokens |
| 2 | Dokument laden & chunken | Kontextfenster, Chunking-Strategie |
| 3 | Embedden & retrieven | Embeddings, semantische Suche |
| 4 | Geerdete Antwort | Grounding, Halluzinationen, Zitate |
| 5 | Strukturierte Extraktion | Structured Output, JSON-Schema |
| 6 | Tool-/Function-Calling | Function Calling, Agenten-Schleife |
| 7 | Eval + UI + Deploy | Evaluierung, Streamlit, Deployment |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # dann Mistral-API-Key eintragen
```

## Verwendung

### Phase 1 – Hello LLM
```bash
python phase1_hello_llm.py
```

Weitere Phasen folgen.
