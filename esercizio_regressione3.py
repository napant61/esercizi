"""
Nome Script: esercizio3.py
Descrizione: Prendendo i dati dell'esercizio2, applichiamo una regressione polinomiale, così da avere una quadro più preciso delle relazioni
tra ore di studio e punteggio finale quando si verifica un peggioramento della tendenza dopo un certo punto.
Autore: Antonio Napolitano
Versione: 1.0
Data: 02/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""
# librerie
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dati verosimili: ore di studio e punteggi (con tendenza non lineare)
ore_studio = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
voti = np.array([30, 45, 60, 75, 85, 90, 88, 85, 80, 75, 70, 65])

# Creazione del modello di regressione polinomiale
grado = 1  # grado del polinomio

poly = PolynomialFeatures(degree=grado)
X_poly = poly.fit_transform(ore_studio)  # Trasformazione dei dati in un polinomio di grado 2

modello = LinearRegression()
modello.fit(X_poly, voti)  # Addestramento del modello sui dati trasformati

# Intercetta e coefficiente angolare della retta
intercetta = modello.intercept_
coefficiente = modello.coef_  # Ignoriamo il primo coefficiente (intercetta del polinomio)  

print(f"Intercetta: {intercetta}")
print(f"Coefficiente: {coefficiente}")

# previsioni con il modello polinomiale. Dobbiamo trasformare i dati di input delle ore di studio nella stessa forma polinomiale usata per l'addestramento
# previsione per la 13 ora di studio
ore_studio_previsione = np.array([13]).reshape(-1, 1)  # Nuovo valore di ore di studio
# Trasformazione del nuovo valore di ore di studio in un polinomio di grado 2
# (stessa trasformazione usata per l'addestramento del modello)
ore_studio_poly_previsione = poly.transform(ore_studio_previsione)
voti_previsti = modello.predict(ore_studio_poly_previsione)

print(f"Previsione di voto per 13 ore di studio: {voti_previsti[0]:.2f}")

# Generare una serie di punti per la curva di regressione
ore_studio_range = np.linspace(ore_studio.min(), ore_studio.max(), 100).reshape(-1, 1)
ore_studio_range_poly = poly.transform(ore_studio_range)
punteggi_previsti_range_poly = modello.predict(ore_studio_range_poly)

# Visualizzare i dati e la curva di regressione polinomiale
plt.scatter(ore_studio, voti, color='blue', label='Dati Reali')
# visualizzazione del punto previsto per 13 ore di studio
plt.scatter(ore_studio_previsione, voti_previsti, color='green', label=f'Previsione (X=13, y={voti_previsti[0]:.2f})', zorder=5)
plt.plot(ore_studio_range, punteggi_previsti_range_poly, color='green', label=f'Regressione Polinomiale (grado {grado})')
plt.xlabel('Ore di Studio')
plt.ylabel('Punteggio Esame')
plt.title('Regressione Polinomiale su Relazione Non Lineare')
plt.grid(True)
plt.legend()
plt.show()

# Per un grado di polinomio uguale a 3 sono stati ottenuti questi risultati:
# Intercetta: 0.21
# Coefficiente: [ 0.         29.56        -3.04  0.08]

# Previsione di voto per 13 ore di studio: 56.88

# Con un grado di polinomio uguale a 1 i risultati erano simili a quelli della regressione lineare semplice.

# La regressione polinomiale ha catturato meglio la relazione non lineare tra ore di studio e punteggio finale.
# La previsione per 13 ore di studio è 56.88, che è più realistica rispetto alla previsione della regressione lineare semplice.
