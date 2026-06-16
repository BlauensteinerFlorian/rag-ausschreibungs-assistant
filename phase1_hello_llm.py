"""
Phase 1 – Hello LLM
===================
Einfachster Einstieg: Einen Prompt an Mistral schicken und Antwort ausgeben.
Lernziele: Messages/Rollen (system, user, assistant), Temperatur, Tokens.

Verwendung:
    1. .env mit MISTRAL_API_KEY befüllen
    2. python phase1_hello_llm.py
"""

import os
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# ─── System-Prompt: Persönlichkeit und Aufgabe definieren ───
system_prompt = (
    "Du bist ein hilfsbereiter Assistent für öffentliche Ausschreibungen. "
    "Antworte präzise und auf Deutsch."
)

# ─── User-Nachricht: Die eigentliche Frage ───
user_message = (
    "Erkläre kurz, was eine öffentliche Ausschreibung ist "
    "und welche Rolle der Schwellenwert dabei spielt."
)

# ─── API-Call mit verschiedenen Parametern ───
print("=" * 60)
print("📡 Sende Prompt an Mistral (mistral-small-latest) ...")
print(f"   System: {system_prompt}")
print(f"   User:   {user_message}")
print("=" * 60)

# Variante A: kreativ (temperature=1.0)
print("\n─── Variante A: temperature=1.0 (kreativ) ───")
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
    temperature=1.0,
    max_tokens=300,
)
print(response.choices[0].message.content)
print(f"\n   Tokens: {response.usage}")

# Variante B: deterministisch (temperature=0)
print("\n\n─── Variante B: temperature=0 (deterministisch) ───")
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
    temperature=0,
    max_tokens=300,
)
print(response.choices[0].message.content)
print(f"\n   Tokens: {response.usage}")

# Variante C: ohne System-Prompt (temperature=0)
print("\n\n─── Variante C: ohne System-Prompt (temperature=0) ───")
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {"role": "user", "content": user_message},
    ],
    temperature=0,
    max_tokens=300,
)
print(response.choices[0].message.content)
print(f"\n   Tokens: {response.usage}")

print("\n✅ Phase 1 abgeschlossen. Beobachte, wie Temperatur und System-Prompt "
      "die Antwort beeinflussen.")
