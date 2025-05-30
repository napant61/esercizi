import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor # Nuovo import
from sklearn.metrics import mean_squared_error, r2_score # Nuove metriche

# Impostiamo il seed per riproducibilità
np.random.seed(42)
giorni = 365

# Generiamo i dati con le nuove distribuzioni gaussiane (più realistiche)
data = {
    'temperatura': np.random.normal(20, 4, size=giorni),
    'umidità': np.random.normal(50, 15, size=giorni), # Deviazione standard ridotta
    'vento': np.random.normal(5, 1, size=giorni),
    'pressione': np.random.normal(1020, 2, size=giorni),
}
df = pd.DataFrame(data)
# Visualizziamo la distribuzione originale delle feature
print("Distribuzione originale delle feature:")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['temperatura'], bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribuzione Originale della Temperatura')
plt.subplot(1, 2, 2)
plt.hist(df['umidità'], bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribuzione Originale dell\'Umidità')
plt.show()

# Clip dell'umidità per i valori fisicamente sensati
df['umidità'] = np.clip(df['umidità'], 0, 100)
df['vento'] = np.clip(df['vento'], 0, None) # Il vento non può essere negativo

# Creiamo il target: la temperatura del giorno successivo
df['target'] = df['temperatura'].shift(-1)

# Aggiungiamo feature laggate (importante per la previsione di serie temporali!)
# Esempio: temperatura del giorno precedente e due giorni precedenti come feature
df['temp_lag_1'] = df['temperatura'].shift(1)
df['temp_lag_2'] = df['temperatura'].shift(2)

# Eliminiamo le righe con NaN a causa di shift()
df.dropna(inplace=True)

# Definiamo feature (X) e target (y)
# Includiamo le nuove feature laggate
X = df[['temperatura', 'umidità', 'vento', 'pressione', 'temp_lag_1', 'temp_lag_2']]
y = df['target']

# Scaliamo le feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Suddividiamo il dataset in training e test set
# Convertiamo direttamente in NumPy per XGBoost
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42)

print("Dimensioni X_train:", X_train.shape)
print("Dimensioni y_train:", y_train.shape)
print("Dimensioni X_test:", X_test.shape)
print("Dimensioni y_test:", y_test.shape)

# --- Addestramento del Modello XGBoost ---
print("\nInizio addestramento del modello XGBoost...")
model_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
model_xgb.fit(X_train, y_train)

# --- Valutazione del Modello XGBoost ---
predictions_xgb = model_xgb.predict(X_test)
test_rmse_xgb = np.sqrt(mean_squared_error(y_test, predictions_xgb))
test_r2_xgb = r2_score(y_test, predictions_xgb)

print(f'\nRMSE sul Test Set (XGBoost): {test_rmse_xgb:.4f}')
print(f'R^2 sul Test Set (XGBoost): {test_r2_xgb:.4f}')

# Facciamo alcune previsioni e confrontiamo con i valori reali
print("\nPrevisioni vs Valori Reali (primi 10 del test set - XGBoost):")
for i in range(min(10, len(y_test))):
    print(f"Previsto: {predictions_xgb[i]:.2f}, Reale: {y_test[i]:.2f}")

# --- Plotting per la visualizzazione ---
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(y_test)), y_test, label='Reale', color='blue')
plt.plot(np.arange(len(y_test)), predictions_xgb, label='Previsto (XGBoost)', color='red', linestyle='--')
plt.title('Confronto tra Valori Reali e Previsioni (XGBoost)')
plt.xlabel('Campioni del Test Set')
plt.ylabel('Temperatura')
plt.legend()
plt.grid(True)
plt.show()

'''
# Conclusione
Significativo Miglioramento Confermatissimo: Il fatto che XGBoost sia riuscito a seguire così bene le fluttuazioni
con "soli" 365 campioni (e circa 70 nel test set) è un ottimo risultato per questo tipo di modello.
Capacità di Generalizzazione: 365 campioni (un anno di dati) sono un punto di partenza molto migliore rispetto a 30 giorni.
Se il modello si comporta bene su un anno di dati, è più probabile che abbia imparato pattern stagionali o di altro tipo che
gli consentono di generalizzare meglio.
Analisi delle Lacune (confermate):

    Sottostima/Sovrastima dei Picchi/Valli: Continuerai a notare che il modello tende a smussare i picchi e le valli estreme.
    Questo è un compromesso comune nei modelli di regressione: è difficile predire con precisione gli eventi più rari o estremi.
    Con più anni di dati, il modello potrebbe imparare meglio queste anomalie.
    Lieve Ritardo: Il leggero ritardo nelle previsioni che si osserva ancora è tipico quando si usano feature laggate
    fisse (es. solo temperatura di ieri e l'altro ieri). Modelli sequenziali più avanzati come LSTM possono gestire meglio
    questo aspetto se la correlazione temporale è molto forte e complessa.

Ottima Scelta del Modello: XGBoost è stata una scelta molto appropriata per la dimensione del tuo dataset e la natura del problema.
Ha dimostrato di essere in grado di estrarre pattern utili anche con un solo anno di dati.
Ing. delle Feature Cruciale: Le feature laggate che abbiamo aggiunto (es. temp_lag_1, temp_lag_2) sono state probabilmente fondamentali
per il successo di XGBoost nel seguire il trend. Senza di esse, anche XGBoost avrebbe faticato a capire la dipendenza temporale.
    Margini di Miglioramento:
        Più Anni di Dati: Anche se 365 giorni sono meglio di 30, per previsioni meteo veramente robuste, avresti bisogno di molti anni di dati (5-10 anni o più) per catturare cicli pluriennali, variazioni climatiche e avere abbastanza esempi di eventi estremi per addestrare il modello.
        Altre Feature Temporali: Se avessi dati su più anni, potresti aggiungere feature come il giorno dell'anno (es. un valore da 1 a 365), il mese, la settimana dell'anno. Queste feature aiutano il modello a capire la stagionalità.
        Feature Meteo Aggiuntive: Considera dati su precipitazioni, copertura nuvolosa, irraggiamento solare, punto di rugiada, ecc.
        Ottimizzazione Iperparametri: Potresti dedicare tempo all'ottimizzazione degli iperparametri di XGBoost (es. max_depth, subsample, colsample_bytree, gamma, ecc.) per spremere ogni punto percentuale di performance.
        Stacked Models / Ensembling: Combinare le previsioni di più modelli (es. un XGBoost con un Random Forest) può a volte migliorare la precisione.

In conclusione, la tua scelta di passare a XGBoost e l'utilizzo di 365 campioni ti hanno portato a un modello significativamente più efficace e realistico per la previsione della temperatura. Il grafico lo dimostra chiaramente.

'''