"""
Nome Script: esercizio_albero1.py
Descrizione: Un esempio semplice di come utilizzare un modello di foresta casuale per un problema di classificazione applicato al dataset iris.
Autore: Antonio Napolitano
Versione: 1.0
Data: 02/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""

# importiamo le librerie necessarie
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# carichiamo il dataset iris
iris = load_iris()
# stabiliamo i valori di input e output su cui addestrare il modello
X = iris.data
y = iris.target

# dividiamo i dati di training da quelli di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# creiamo il modello di Random Forest che ci serve per classificare i dati
foresta_casuale = RandomForestClassifier(n_estimators=100, random_state=42)
# dove: n_estimators=100 specifica il numero di alberi nella foresta (100)

# addestriamo il modello
foresta_casuale.fit(X_train, y_train)

# facciamo le previsioni sul modello addestrato
y_predict = foresta_casuale.predict(X_test)

# valutiamo l'accuratezza del modello
accuratezza = accuracy_score(y_test, y_predict)
print(f"Accuratezza del modello Random Forest pari a: {accuratezza:.2f}")

# visualizziamo anche l'importanza delle caratteristiche, in modo da fornire un'indicazione di quanto 
# ogni caratteristica sia stata importante nel processo di classificazione

feature_importances = foresta_casuale.feature_importances_
for i, importance in enumerate(feature_importances):
    print(f"Importanza della caratteristica {iris.feature_names[i]}: {importance:.4f}")

