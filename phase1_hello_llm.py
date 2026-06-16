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
from mistralai import Mistral

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

# Variante A: neutral (temperature=0.7)
print("\n─── Variante A: temperature=0.7 (kreativer) ───")
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
    temperature=0.7,
    max_tokens=300,
)
print(response.choices[0].message.content)
print(f"\n   Tokens: {response.usage}")

# Variante B: nüchtern (temperature=0.1)
print("\n\n─── Variante B: temperature=0.1 (faktischer) ───")
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
    temperature=0.1,
    max_tokens=300,
)
print(response.choices[0].message.content)
print(f"\n   Tokens: {response.usage}")

# Variante C: ohne System-Prompt
print("\n\n─── Variante C: ohne System-Prompt ───")
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {"role": "user", "content": user_message},
    ],
    temperature=0.1,
    max_tokens=300,
)
print(response.choices[0].message.content)
print(f"\n   Tokens: {response.usage}")

print("\n✅ Phase 1 abgeschlossen. Beobachte, wie Temperatur und System-Prompt "
      "die Antwort beeinflussen.")
