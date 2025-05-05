"""
Nome Script: esercizio4.py
Descrizione: Uno script sulle Regressione lineare con più variabili indipendenti. Prendiamo ad esempio i prezzi delle case in base
al numero di stanze e alla superficie in metri quadri. In questo caso, la regressione lineare semplice non è sufficiente.
Autore: Antonio Napolitano
Versione: 1.0
Data: 02/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""
# importiamo le librerie necessarie
import numpy as np # per la genstione delle matrici di dati
import pandas as pd # per la gestione dei dati in formato tabellare
from sklearn.linear_model import LinearRegression # per utilizzare il modello di regressione lineare semplice
# importiamo anche matplotlib per visualizzare i dati
import matplotlib.pyplot as plt # per visualizzare i grafici    
# importiamo il modello di addestramento
from sklearn.model_selection import train_test_split # per dividere i dati in set di addestramento e test
from sklearn.metrics import mean_squared_error, r2_score # per calcolare gli errori e il coefficiente di determinazione R^2
# importiamo il modulo per visualizzare i grafici in 3d
from mpl_toolkits.mplot3d import Axes3D 

# Creiamo i dataset di esempio
data = {
    'dimensione': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    'stanze': [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    'prezzo': [100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000]
}
df = pd.DataFrame(data)
# Visualizziamo subito i dati
print(df)

# Definiamo le variabili indipendenti (X) e la variabile dipendente(y)
X = df[['dimensione', 'stanze']].values # matrice delle variabili indipendenti
y = df['prezzo'].values # matrice della variabile dipendente

# Dividiamo i dati in set di training e test (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creiamo il modello di regressione lineare e addestriamolo con i set di training
modello = LinearRegression()
modello.fit(X_train, y_train)

# Intercetta e coefficiente angolare
intercetta = modello.intercept_
coefficiente = modello.coef_

# Stampiamo subito i risultati di intercetta e coefficiente
print(f"L'intercetta é {intercetta}")
print(f"I coefficienti sono: {coefficiente}") # sono due perchè abbiamo due variabili indipendenti (dimensione e stanze)

# Facciamo le previsioni sul set di dati di test
y_pred = modello.predict(X_test)

# valutiamo le previsioni con l'errore quadratico medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Errore quadratico medio (MSE): {mse:.2f}")

# Valutiamo le previsioni con il coefficiente di determinazione R^2
r2 = r2_score(y_test, y_pred)
print(f"Coefficiente di determinazione R^2: {r2:.2f}")

# A questo punto visualizziamo i dati in un grarfico 3D.
# Creiamo un grafico 3D per visualizzare i dati e la retta di regressione
fig = plt.figure(figsize=(8, 8)) # dimensioni del layout che ospita il grafico
ax = fig.add_subplot(111, projection='3d') # creiamo un grafico 3D, 111 mostra la disposizione del grafico. 
# 1 indica che ci sarà un solo grafico, 1 indica che il grafico occupa tutta la superficie del layout
# e 1 indica che il grafico è il primo (e unico) grafico del layout.
# A questo punto possiamo visualizzare i dati di partenza mostrati come punti blu
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='blue', marker='o', label='Dati di Training')
# e i dati di test mostrati come punti rossi
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='red', marker='^', label='Dati di test')

# A questo punto possiamo visualizzare la retta di regressione
# Creiamo una griglia di punti per la superficie di regressione
X1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30) # dimensione. 30 punti tra il minimo e il massimo
X2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 30) # stanze. 30 punti tra il minimo e il massimo

# Creaiamo una griglia per mostrare tutte le combinazioni di dimensione e stanze
XX, YY = np.meshgrid(X1_range, X2_range) # meshgrid crea una griglia di punti 2D
# calcoliamo i prezzi previsti per ogni combinazione di dimensione e stanze
# utilizzando il modello di regressione lineare
ZZ = modello.predict(np.c_[XX.ravel(), YY.ravel()]) # ravel serve a trasformare la matrice in un array 1D
# e c_ serve a concatenare le due matrici in una sola matrice 2D

# Visualizziamo la superficie di regressione
ax.plot_surface(XX, YY, ZZ.reshape(XX.shape), alpha=0.5, color='green', label='Superficie di Regressione') # alpha serve a rendere la superficie trasparente

ax.set_xlabel('Dimensione (mq)')
ax.set_ylabel('Numero di stanze')
ax.set_zlabel('Prezzo (€)')
ax.set_title('Regressione Lineare con due variabili indipendenti')
ax.legend()
plt.show()