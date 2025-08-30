#!/usr/bin/env python3
"""
Aufgabe 4:
Ausgangslage: Nutzung der OpenAI API für die Integration.
Tasks:
- Nutzt die OpenAI API, um Text zu generieren.
- Führt simple Tests der Endpoints durch.

API-Schlüssel in Umgebungsvariable laden:
Stelle sicher, dass du den API-Schlüssel als Umgebungsvariable via Terminal gesetzt hast:
    export OPENAI_API_KEY="api_schluessel"
"""

import os
from openai import OpenAI

def load_api_key() -> str:
    # Prüfe zuerst die Umgebungsvariable
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    # Falls nicht vorhanden, versuche, die Datei "api_key.txt" zu lesen
    key_file = "api_key.txt"
    try:
        with open(key_file, "r") as f:
            api_key = f.read().strip()
        if api_key:
            return api_key
    except Exception:
        pass

    raise Exception("API key not found. Set OPENAI_API_KEY or provide the key in api_key.txt.")

def _get_float_env(name: str, default: float) -> float:
    try:
        val = os.environ.get(name)
        if val is None:
            return default
        return float(val)
    except Exception:
        return default

def main() -> None:
    # Lade den API-Schlüssel und erstelle eine OpenAI-Clientinstanz
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    # Defaults: temperature=0.7, top_p=1.0
    temperature = _get_float_env("OPENAI_TEMPERATURE", 0.1)
    top_p = _get_float_env("OPENAI_TOP_P", 1.0)

    if os.environ.get("OPENAI_TOP_K") is not None:
        print("Warning: OPENAI_TOP_K is not supported by the OpenAI SDK. Use OPENAI_TOP_P (0.0-1.0) instead. Ignoring OPENAI_TOP_K.")

    # Sende eine Anfrage an das Modell "gpt-4o"
    chat_completion = client.chat.completions.create(
        model="gpt-4o",  # Alternativ: "gpt-3.5-turbo", falls gewünscht
        messages=[
            {
                "role": "user",
                "content": "Erkläre das 3-Körper Problem in einfachen Worten und klar verständlich",
            }
        ],
        temperature=temperature,
        top_p=top_p,
    )

    # Gib die Antwort der OpenAI API aus
    print(f"Antwort von OpenAI API: {chat_completion.choices[0].message.content}")

if __name__ == "__main__":
    main()
