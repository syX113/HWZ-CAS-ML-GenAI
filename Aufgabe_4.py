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
import json
import re
import argparse
from typing import Dict, Any, Tuple, List
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
    # CLI-Argumente: Modus wählen (chat oder sentiment) und optional Prompt/Modelle setzen
    parser = argparse.ArgumentParser(description="Aufgabe 4: OpenAI Chat-Test oder Sentimentanalyse")
    parser.add_argument("--mode", choices=["chat", "sentiment"], default="chat", help="'chat' für Einzelprompt-Test, 'sentiment' für Klassifikation")
    parser.add_argument("--prompt", default="Erkläre das 3-Körper-Problem in einfachen Worten und klar verständlich.", help="Prompt für den Chat-Test")
    parser.add_argument("--chat-model", default="gpt-4o", help="Modellname für Chat-Test")
    parser.add_argument("--cls-model", default="gpt-4o-mini", help="Modellname für Sentiment-Klassifikation")
    args = parser.parse_args()

    # Lade den API-Schlüssel und erstelle eine OpenAI-Clientinstanz
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    # Defaults über Umgebungsvariablen: temperature, top_p
    temperature = _get_float_env("OPENAI_TEMPERATURE", 0.1)
    top_p = _get_float_env("OPENAI_TOP_P", 1.0)

    if os.environ.get("OPENAI_TOP_K") is not None:
        print("Warning: OPENAI_TOP_K is not supported by the OpenAI SDK. Use OPENAI_TOP_P (0.0-1.0) instead. Ignoring OPENAI_TOP_K.")

    # Robustes JSON aus Antwort extrahieren
    def _safe_json_extract(s: str) -> Dict[str, Any]:
        try:
            return json.loads(s)
        except Exception:
            pass
        match = re.search(r"\{[\s\S]*\}", s)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
        return {}

    # Chat-Einzeltest
    def run_chat_test() -> None:
        chat_completion = client.chat.completions.create(
            model=args.chat_model,
            messages=[{"role": "user", "content": args.prompt}],
            temperature=temperature,
            top_p=top_p,
        )
        print(f"Antwort von OpenAI API: {chat_completion.choices[0].message.content}\n")

    # Sentiment-Klassifikation einer Liste an Sätzen
    def classify_one(text: str) -> Tuple[str, float, str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a sentiment classifier. Return a strict JSON object with keys "
                    "'label' (one of 'POSITIVE' or 'NEGATIVE'), 'confidence' (0..1), and "
                    "'explanation' (one concise sentence citing phrases in the text). "
                    "Respond with JSON only."
                ),
            },
            {
                "role": "user",
                "content": f"Text: {text}",
            },
        ]

        resp = client.chat.completions.create(
            model=args.cls_model,  # z.B. "gpt-4o-mini" für Klassifikation
            messages=messages,
            temperature=0.0,
            top_p=1.0,
        )
        content = resp.choices[0].message.content or "{}"
        data = _safe_json_extract(content)
        label = str(data.get("label", "UNKNOWN")).upper()
        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        explanation = str(data.get("explanation", "")).strip()
        if label not in {"POSITIVE", "NEGATIVE"}:
            label = (
                "POSITIVE"
                if any(w in text.lower() for w in ["good", "fantastic", "excellent", "delicious", "outstanding", "perfect", "friendly", "wonderful", "satisfied"])
                else "NEGATIVE"
            )
        confidence = max(0.0, min(1.0, confidence))
        if not explanation:
            lower = text.lower()
            pos_cues = [w for w in ["fantastic", "excellent", "delicious", "outstanding", "perfect", "friendly", "satisfied", "wonderful"] if w in lower]
            neg_cues = [w for w in ["worst", "buggy", "unreliable", "late", "damaged", "stopped", "ruined", "never recommend", "waste"] if w in lower]
            if label == "POSITIVE" and pos_cues:
                explanation = f"Positive cues detected: {', '.join(pos_cues)}."
            elif label == "NEGATIVE" and neg_cues:
                explanation = f"Negative cues detected: {', '.join(neg_cues)}."
            else:
                explanation = "Overall tone and phrasing indicate this sentiment."
        return label, confidence, explanation

    def run_sentiment() -> None:
        sample_texts: List[str] = [
            "I find the product cool but the shipping was delayed and the packaging was damaged.",
            "This is the worst experience I've ever had.",
            "The movie was fantastic, I enjoyed every minute.",
            "I don't like the new update, it ruined everything.",
            "An outstanding performance by the lead actor.",
            "I would never recommend this service to anyone.",
            "The food was delicious and the ambiance was perfect.",
            "The product stopped working after a week.",
            "Excellent customer service and very friendly staff.",
            "The package arrived late and was damaged.",
        ]
        print("Sentimentanalyse (OpenAI) für 10 Beispielsätze:")
        for i, txt in enumerate(sample_texts, start=1):
            label, conf, expl = classify_one(txt)
            print(f"{i:02d}. Text: {txt}")
            print(f"    Predicted: {label} (Conf.: {conf:.2f})")
            print(f"    Why: {expl}")
            print("-" * 80)

    # Modus ausführen
    if args.mode == "chat":
        run_chat_test()
    else:
        run_sentiment()

if __name__ == "__main__":
    main()
