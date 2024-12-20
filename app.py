"""
@author: Moussa Kalla
"""

from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Initialiser Flask
app = Flask(__name__)

# Charger le tokenizer et le modèle fine-tuné (modèle BERT pour classification)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Dictionnaire des maladies et services
diseases_services = {
    0: {"maladie": "Insuffisance rénale", "service": "Cardiologie"},
    1: {"maladie": "Paludisme", "service": "Maladie Contagieuse"},
    2: {"maladie": "Covid", "service": "Maladie Contagieuse"},
    3: {"maladie": "Hernies discales", "service": "Dynamisme et moteur"},
    4: {"maladie": "Infections urinaires", "service": "Urologie"},
    5: {"maladie": "Fibromyosite", "service": "Gynécologie"},
}

# Fonction pour faire une prédiction
def predict_disease(symptoms):
    inputs = tokenizer(symptoms, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    disease = diseases_services[prediction]["maladie"]
    service = diseases_services[prediction]["service"]
    return disease, service

# Route pour la page d'accueil
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Récupérer les symptômes soumis par l'utilisateur
        symptoms = request.form['symptoms']
        # Faire une prédiction
        maladie, service = predict_disease(symptoms)
        message = f"Vous avez « {maladie} ». Veuillez vous diriger au service « {service} »."
        return render_template('index.html', prediction=message)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
