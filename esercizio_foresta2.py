"""
Nome Script: esercizio_albero1.py
Descrizione: Un esempio più complesso di utilizzo del modello di foresta casuale per classificare se un tumore alla mammella è benigno o maligno.
Si è usato il dataset a classificiazione binaria Breast Cancer Wisconsin (diagnostic), disponibile tramite la libreria scikit-learn.
Autore: Antonio Napolitano
Versione: 1.0
Data: 02/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""
# importiamo le librerie e il set di dati
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset Breast Cancer Wisconsin
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
feature_names = breast_cancer.feature_names
target_names = breast_cancer.target_names

# Crea un DataFrame per visualizzare meglio i dati (opzionale)
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Divide i dati in set di training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crea un modello di Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Addestra il modello sui dati di training
rf_classifier.fit(X_train, y_train)

# Effettua le previsioni sul set di test
y_pred = rf_classifier.predict(X_test)

# Valuta le prestazioni del modello
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello Random Forest: {accuracy:.2f}")

# Mostra il report di classificazione
print("\nReport di Classificazione:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Mostra la matrice di confusione
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Previsioni')
plt.ylabel('Valori Reali')
plt.title('Matrice di Confusione')
plt.show()

# Visualizza l'importanza delle caratteristiche
feature_importances = rf_classifier.feature_importances_
sorted_indices = feature_importances.argsort()[::-1] # Ordina in modo decrescente

plt.figure(figsize=(12, 8))
plt.title("Importanza delle Caratteristiche")
plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

# Stampa l'importanza delle caratteristiche in ordine decrescente
print("\nImportanza delle Caratteristiche (in ordine decrescente):")
for i in sorted_indices:
    print(f"{feature_names[i]}: {feature_importances[i]:.4f}")