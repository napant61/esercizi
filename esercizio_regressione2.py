"""
Nome Script: esercizio2.py
Descrizione: Una regressione lineare applicata a valori che non sono perfettamente correlati. In questo esempio si assume
che il voto di esame possa inizialmente crescere all'aumentare delle ore di studio ma che poi diminuisca a causa della
stanchezza o di qualsiasi altro motivo quando le ore diventano eccessive.
Autore: Antonio Napolitano
Versione: 1.0
Data: 01/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""
# importiamo le librerie
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# prepariamo dati verosimili su ore di studio e voti
ore_studio = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
voti = np.array([30, 45, 60, 75, 85, 90, 88, 85, 80, 75, 70, 65])

# un grafico preliminare tanto per visualizzare i dati di partenza
plt.scatter(ore_studio, voti, color='blue', label='Dati reali')
plt.xlabel('Ore di studio')
plt.ylabel('Voti esame')
plt.title('Relazione grafica tra ore di studio e voti ottenuti')
plt.legend()
plt.grid()
plt.show()

# la curva è simile a una U rovesciata. Quindi la relazione non è strettamente lineare.
# Applico comunque il modello di regressione lineare per vedere i limiti del modello in casi analoghi.

# creo e addestro il modello di regressione lineare
modello = LinearRegression()
modello.fit(ore_studio, voti)

# intercetta sull'asse y e coefficiente angolare
intercetta = modello.intercept_
coefficiente = modello.coef_[0]

# stampa dei risultati su intercetta e coefficiente
print(f"Intercetta: {intercetta}")
print(f"Coefficiente: {coefficiente}")

# previsioni con il modello sopra:
voti_previsti = modello.predict(ore_studio)

# Visualizzare i dati e la retta di regressione lineare
plt.scatter(ore_studio, voti, color='blue', label='Dati Reali')
plt.plot(ore_studio, voti_previsti, color='red', label=f'Regressione Lineare (y = {coefficiente:.2f}x + {intercetta:.2f})')
plt.xlabel('Ore di Studio')
plt.ylabel('Punteggio Esame')
plt.title('Regressione Lineare su Dati Non Lineari')
plt.grid(True)
plt.legend()
plt.show()

# La retta di regressione lineare cerca di trovare una relazione media tra le due variabili (ore_studio, voti_previsti)
# ma non cattura la forma a U rovesciata dei dati reali.

# Di conseguenza, la regressione lineare in questo caso non è adatta per modellare relazioni NON lineari, come quella
# vista sopra.

# Per confermare questa conclusione, basta calcolare le correlazioni tra le due variabili (ore_studio, voti).
correlazione = np.corrcoef(ore_studio.flatten(), voti)[0, 1]    
print(f"Correlazione tra ore di studio e voti: {correlazione:.2f}") 
# La correlazione è positiva (0.51), ma non perfetta. Questo indica che la relazione non è lineare.
