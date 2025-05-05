"""
Nome Script: esercizio1.py
Descrizione: Una regressione lineare semplice con una procedura di addestramento con un solo valore
Autore: Antonio Napolitano
Versione: 1.0
Data: 01/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""
# importiamo le librerie necessarie:
import numpy as np # per la genstione delle matrici di dati
from sklearn.linear_model import LinearRegression # per utilizzare il modello di regressione lineare semplice
import matplotlib.pyplot as plt # per visualizzare i grafici


# Utilizzo di due semplici set di dati. La prima matrice è la variabile indipendente (X); la seconda la variabile dipendente (y).
# Dati per la variabile X (input)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) # il reshape serve a trasformare la matrice 1D in una colonna di dati, come richiesto da scikit-learn.
y = np.array([2, 4, 5, 6, 7])

# Creo il modello di regressione lineare
modello = LinearRegression()

# Lo addestro
modello.fit(X, y) # questo metodi crea la retta di regressione lineare che utilizzeremo per fare le previsioni

# per farlo, ci serve il coefficiente angolare della retta (ovvero la sua pendenza) e l'itercetta sull'asse y
# l'intercetta:
intercetta = modello.intercept_

# il coefficente angolare:
coefficiente = modello.coef_[0] # solo il primo elemento della variabile indipendente

# printiamo i risultati
print(f"Intercetta della retta è: {intercetta}")
print(f"Il coefficente della retta è: {coefficiente}")

# adesso facciamo le previsioni su un uunico nuovo valore di X (6)
nuovo_X = np.array([6]).reshape(-1, 1)
predetto_y = modello.predict(nuovo_X)
print(f"Previsione per X = 6: {predetto_y[0]:.2f}")

# visualizziamo i risultati con plt.scatter per vedere i dati reali
plt.scatter(X, y, color='blue', label='Dati')
# Aggiungiamo anche il punto previsto per X = 6
plt.scatter(nuovo_X, predetto_y, color='green', label=f'Previsione (X=6, y={predetto_y[0]:.2f})', zorder=5)
# visualizziamo anche la retta di regressione in rosso
plt.plot(X, modello.predict(X), color='red', label=f'Regressione Lineare (y = {coefficiente:.2f}x + {intercetta:.2f})')
plt.xlabel('Variabile Indipendente (X)')
plt.ylabel('Variabile Dipendente (y)')
plt.title('Regressione Lineare')
plt.legend()
plt.grid(True)
plt.show()

# questo modello di regressione lineare è piuttosto adeguato a fare previsioni attendibili


