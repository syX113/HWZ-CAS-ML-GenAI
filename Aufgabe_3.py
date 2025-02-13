#!/usr/bin/env python3
"""
Aufgabe 3:
Ausgangslage: Wählt ein Modell auf Hugging Face (z.B. ein kleineres Modell wie 7b) zur Sentimentanalyse.
Das Modell soll genutzt werden, um Texte zu klassifizieren.

Tasks:
- Fuehrt eine Sentimentanalyse von Text durch.
- Diskutiert anschliessend, welche Metriken (z.B. Accuracy, Precision, Recall, F1-Score) sinnvoll waeren und warum.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # Modellname auswählen (hier wird ein Beispielmodell für Sentimentanalyse verwendet)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Modell laden (auf der CPU)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Erstelle eine Pipeline für die Sentimentanalyse
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    # Liste von Beispielsätzen für die Klassifizierung
    sample_texts = [
        "I absolutely love the new design of your product! It is amazing and well thought out.",
        "This is the worst experience I've ever had.",
        "The movie was fantastic, I enjoyed every minute.",
        "I don't like the new update, it ruined everything.",
        "An outstanding performance by the lead actor.",
        "I would never recommend this service to anyone.",
        "The food was delicious and the ambiance was perfect.",
        "The product stopped working after a week.",
        "Excellent customer service and very friendly staff.",
        "The package arrived late and was damaged.",
        "A wonderful experience, truly exceptional.",
        "It was a complete waste of time.",
        "I am extremely satisfied with my purchase.",
        "The software is buggy and unreliable.",
        "A truly enjoyable and unforgettable journey."
    ]
    
    # Erwartete Labels für die Beispielsätze (True Labels), Reihenfolge entspricht der der Beispielsätze:
    # 1 = positiv, 0 = negativ (diese Labels sind manuell festgelegt, basierend auf der Stimmung der Sätze)
    true_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    # Führe die Sentimentanalyse für jeden Beispielsatz durch
    results = sentiment_pipeline(sample_texts)
    
    # Ausgabe der Ergebnisse: Text, True Label, Predicted Label und Score
    print("Sentimentanalyse Ergebnisse (mit True Labels):")
    predicted_labels = []
    for text, true_label, result in zip(sample_texts, true_labels, results):
        predicted = 1 if result['label'] == 'POSITIVE' else 0
        predicted_labels.append(predicted)
        print(f"Text: {text}")
        print(f"True Label: {true_label} - Predicted: {predicted} (Score: {result['score']:.4f})")
        print("-" * 80)
    
    # Berechne verschiedene Evaluationsmetriken
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    # Ausgabe der Evaluationsergebnisse
    print("Evaluation der Sentimentanalyse:")
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-Score:  {f1:.2f}")

if __name__ == "__main__":
    main()
