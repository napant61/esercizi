import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

altezza = np.array([[165], [180], [172], [158], [175], [20], [105], [168], [200], [260], [34], [186], [276], [175]])

scaler = StandardScaler()
scaler.fit(altezza)

altezza_standardizzata = scaler.transform(altezza)

# Creazione della tabella
tabella = pd.DataFrame({
    "Altezza Reale (cm)": altezza.flatten(),
    "Altezza Standardizzata (Z-score)": altezza_standardizzata.flatten()
})

# Stampa della tabella
print(tabella)

soglia_max = 1.0
soglia_min = 0.8

# valutazione degli outlier
outliers = tabella[(tabella["Altezza Standardizzata (Z-score)"] > soglia_max) | (tabella["Altezza Standardizzata (Z-score)"] < -soglia_min)]
outlier_indices = outliers.index

if not outliers.empty:
    print(f"\nI seguenti punti dati potrebbero essere considerati outlier (con una soglia di Z-score = {soglia_max} e {soglia_min}):")
    print(outliers)
else:
    print(f"\nNessun outlier significativo rilevato con una soglia di Z-score =  oppure {soglia_min}.")

# Creazione del grafico
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tabella.index, y=tabella["Altezza Reale (cm)"], label="Dati Normali")
sns.scatterplot(x=outlier_indices, y=tabella.loc[outlier_indices, "Altezza Reale (cm)"], color='red', label="Outlier", s=100)

# Linee orizzontali per indicare i limiti degli outlier basati sullo Z-score (solo per riferimento)
media_altezza = tabella["Altezza Reale (cm)"].mean()
std_altezza = tabella["Altezza Reale (cm)"].std()
plt.axhline(media_altezza + soglia_max * std_altezza, color='red', linestyle='--', linewidth=0.8, label=f'Soglia Outlier (+{soglia_max} SD)')
plt.axhline(media_altezza - soglia_min * std_altezza, color='blue', linestyle='--', linewidth=0.8, label=f'Soglia Outlier (-{soglia_min} SD)')

# Etichette e titolo
plt.xlabel("Indice del Dato")
plt.ylabel("Altezza Reale (cm)")
plt.title("Visualizzazione degli Outlier nell'Altezza")
plt.legend()
plt.grid(True)
plt.xticks(tabella.index)
plt.tight_layout()
plt.show()

