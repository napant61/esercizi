"""
Nome Script: prova_01.py
Descrizione: Un esempio molto basilare di modello previsionale basato sulle reti neurali. I set di dati sono generati
casualmente ma sono logicamente verosimili. Il modello prevede se un animale è un gatto o un cane in base a due
caratteristiche: altezza delle orecchie e lunghezza del muso.
Autore: Antonio Napolitano
Versione: 1.0
Data: 08/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
#from tensorflow.keras import layers

# 1. Genera un dataset binario di esempio
np.random.seed(42)
num_samples = 1000
orecchie_gatto = np.random.normal(loc=6.0, scale=1.5, size=num_samples // 2)
muso_gatto = np.random.normal(loc=4.0, scale=1.0, size=num_samples // 2)
orecchie_cane = np.random.normal(loc=8.0, scale=2.0, size=num_samples // 2)
muso_cane = np.random.normal(loc=7.0, scale=1.5, size=num_samples // 2)

X = np.vstack((np.column_stack((orecchie_gatto, muso_gatto)),
               np.column_stack((orecchie_cane, muso_cane))))
y = np.concatenate((np.zeros(num_samples // 2), np.ones(num_samples // 2))) # 0 per gatto, 1 per cane

# 2. Dividi il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scala le caratteristiche (importante per le reti neurali)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Definisci il modello della rete neurale
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(2,)), # Strato di input con 2 neuroni (altezza orecchie, lunghezza muso)
    layers.Dense(1, activation='sigmoid') # Strato di output con 1 neurone (probabilità della classe 1)
])

# 5. Compila il modello
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 6. Addestra il modello
epochs = 20
batch_size = 32
history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

# 7. Valuta il modello sul test set
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Loss sul test set: {loss:.4f}")
print(f"Accuratezza sul test set: {accuracy:.4f}")

# 8. Fai delle predizioni (opzionale)
predictions = model.predict(X_test_scaled)
predictions_binary = (predictions > 0.5).astype(int) # Converti le probabilità in classi (0 o 1)

print("\nPrime 50 predizioni:")
for i in range(50):
    print(f"Dati: {X_test[i]}, Predizione (probabilità): {predictions[i][0]:.4f}, Predizione (classe): {predictions_binary[i][0]}, Reale: {y_test[i]}")

# viasualizziamo i dati in un grafico
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Gatto', alpha=0.5)
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Cane', alpha=0.5)
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='cyan', label='Gatto Test', alpha=0.9, marker='x')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='orange', label='Cane Test', alpha=0.9, marker='x')
plt.title('Distribuzione dei dati di addestramento e test')
plt.xlabel('Altezza orecchie')
plt.ylabel('Lunghezza muso')
plt.legend()
plt.grid()
plt.show()
