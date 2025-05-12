"""
Nome Script: test_02.py
Descrizione: In questo script bisogna confrontare più modelli di previsione per determinare quale sia il migliore. L'esercizio deve determinare il prezzo
di auto usate in base a valori quali: il chilometraggio, l'anno di immatricolazione, la cilindrata. Bisogna attribuire un peso alla cilindrata, in modo che
più è alta più il prezzo aumenta. Al contrario, più l'auto è vecchia, più il prezzo diminuisce.
Autore: Antonio Napolitano
Versione: 1.0
Data: 11/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Caricamento del dataset

df = pd.read_csv("auto_usate.csv")
# Visualizzazione delle prime righe del dataset
print(df.head())

# Divisione del dataset in variabili indipendenti e dipendenti
X = df[["anno", "chilometraggio", "cilindrata"]]
y = df["prezzo"]

# calcolo iniziale delle correlazioni con il prezzo
correlazioni = X.corrwith(y)
print("Correlazioni con il prezzo:\n", correlazioni)

# i grafici di dispersione appaiati delle tre correlazioni
# creazione del layout generale
plt.figure(figsize=(15, 5))

# grafico della correlazione anno-prezzo
plt.subplot(1, 3, 1)
plt.scatter(df["anno"], y)
plt.xlabel("Anno")
plt.ylabel("Prezzo")
plt.title(f"Correlazione anno-prezzo: {correlazioni['anno']:.2f}")

# grafico della correlazione chilometraggio-prezzo
plt.subplot(1, 3, 2)
plt.scatter(df['chilometraggio'], y)
plt.xlabel("Chilometraggio")
plt.ylabel("Prezzo")
plt.title(f"Correlazione chilometraggio-prezzo: {correlazioni['anno']:.2f}")

# grafico della correlazione cilindrata-prezzo
plt.subplot(1, 3, 3)
plt.scatter(df['cilindrata'], y)
plt.xlabel("Cilindrata")
plt.ylabel("Prezzo")
plt.title(f"Correlazione cilindrata-prezzo: {correlazioni['cilindrata']:.2f}")

plt.tight_layout()
plt.show()

# Visualizzazione della correlazione tra le variabili
correlation_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice di Correlazione")
plt.show()
# Creazione di un modello di regressione lineare
# aggiungiamo una costante per il modello lineare
X = sm.add_constant(X)

# Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# modello di regressione lineare
modello = sm.OLS(y_train, X_train).fit()

print(f"Risultati del prezzo per la regressione lineare:")
print(modello.summary())

# previsioni sul set di test
y_pred = modello.predict(X_test)

# Calcoliamo le metriche di valutazione
mse = mean_squared_error(y_test, y_pred)
rquadro = r2_score(y_test, y_pred)
print(f"L'errore quadratico medio è {mse:.2f}")
print(f"Rquadro è {rquadro:.2f}")

# modello di regressione polinomiale
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
modello_poly = LinearRegression()
modello_poly.fit(X_poly, y_train)
y_pred_poly = modello_poly.predict(poly.transform(X_test))
mse_poly = mean_squared_error(y_test, y_pred_poly)
rquadro_poly = r2_score(y_test, y_pred_poly)
print(f"L'errore quadratico medio per il modello polinomiale è {mse_poly:.2f}")
print(f"Rquadro per il modello polinomiale è {rquadro_poly:.2f}")


# modello di regressione Random Forest
modello_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modello_rf.fit(X_train, y_train)
y_pred_rf = modello_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rquadro_rf = r2_score(y_test, y_pred_rf)
print(f"L'errore quadratico medio per il modello Random Forest è {mse_rf:.2f}")
print(f"Rquadro per il modello Random Forest è {rquadro_rf:.2f}")


# modello di regressione Decision Tree
modello_dt = DecisionTreeRegressor(random_state=42)
modello_dt.fit(X_train, y_train)
y_pred_dt = modello_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rquadro_dt = r2_score(y_test, y_pred_dt)
print(f"L'errore quadratico medio per il modello Decision Tree è {mse_dt:.2f}")
print(f"Rquadro per il modello Decision Tree è {rquadro_dt:.2f}")


# grafici sinottici
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
# grafico del modello polinomiale
plt.scatter(y_test, y_pred_poly, color="blue", label="Previsioni")
plt.scatter(y_test, y_test, color="red", label="Valori Reali")
plt.xlabel("Valori Reali")
plt.ylabel("Previsioni")
plt.title("Modello Polinomiale")
plt.legend()


plt.subplot(1, 3, 2)
# grafico del modello Random Forest
plt.scatter(y_test, y_pred_rf, color="blue", label="Previsioni")
plt.scatter(y_test, y_test, color="red", label="Valori Reali")
plt.xlabel("Valori Reali")
plt.ylabel("Previsioni")
plt.title("Modello Random Forest")
plt.legend()


plt.subplot(1, 3, 3)
# grafico del modello Decision Tree
plt.scatter(y_test, y_pred_dt, color="blue", label="Previsioni")
plt.scatter(y_test, y_test, color="red", label="Valori Reali")
plt.xlabel("Valori Reali")
plt.ylabel("Previsioni")
plt.title("Modello Decision Tree")
plt.legend()


plt.tight_layout()
plt.show()

# controllo dell'overfitting sul modello Random Forest
print("\nControllo dell'overfitting sul modello Random Forest")

# previsioni sui dati di training e di test
y_train_pred_rf = modello_rf.predict(X_train)
y_test_pred_rf = modello_rf.predict(X_test)

# Calcolo del Rquadro sui dati di training e di test
rquadro_train_rf = r2_score(y_train, y_train_pred_rf)
rquadro_test_rf = r2_score(y_test, y_test_pred_rf)

print("\nPerformance del modello Random Forest")
print(f"Rquadro sui dati di training: {rquadro_train_rf:.2f}")
print(f"Rquadro sui dati di test: {rquadro_test_rf:.2f}")

'''
Il modello di Random Forest sviluppato sopra si adatta molto bene ai dati di training (98% della varianza), un po' meno bene ai dati di test
(86% della varianza).
Esiste quindi un segnale di LEGGERO OVERFITTING.
Il calo dell'R-quadro da 0.98 a 0.86 sul set di test indica che il modello non generalizza perfettamente a dati nuovi che non ha mai visto
durante l'addestramento.
La diminuzione, tuttavia, è significativa ma non estrema. L'R-quadro a 0.86 sul set di test è comunque una perfomance piuttosto buona, indicando che
il modello è in grado di fare previsioni ragionevolmente accurate sui nuovi dati.
'''
