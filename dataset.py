"""
Nome Script: dataset.py
Descrizione: Questo script carica un dataset di vino rosso normalizzato, aggiunge nuovi dati, normalizza i nuovi dati
e applica due modelli di clustering (K-Means e Clusstering gerarchico) per visualizzare i risultati. Inoltre, calcola
e visualizza un dendrogramma per valutare la struttura dei cluster.
tkinter
Autore: Antonio Napolitano
Versione: 1.0
Data: 18/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('winequality-red.csv', sep=';')
print(data)

# applichiamo la regola della media per i valori NaN
data['fixed acidity'].fillna(data['fixed acidity'].mean(), inplace=True)
data['volatile acidity'].fillna(data['volatile acidity'].mean(), inplace=True)
data['citric acid'].fillna(data['citric acid'].mean(), inplace=True)
data['residual sugar'].fillna(data['residual sugar'].mean(), inplace=True)
data['chlorides'].fillna(data['chlorides'].mean(), inplace=True)
data['free sulfur dioxide'].fillna(data['free sulfur dioxide'].mean(), inplace=True)
data['total sulfur dioxide'].fillna(data['total sulfur dioxide'].mean(), inplace=True)
data['density'].fillna(data['density'].mean(), inplace=True)
data['pH'].fillna(data['pH'].mean(), inplace=True)
data['sulphates'].fillna(data['sulphates'].mean(), inplace=True)
data['alcohol'].fillna(data['alcohol'].mean(), inplace=True)
data['quality'].fillna(data['quality'].mean(), inplace=True)
# visualizziamo i dati con i valori NaN sostituiti
print('\nDati con valori NaN sostituiti:')
print(data.head())

# normalizziamo i dati
data['fixed acidity'] = (data['fixed acidity'] - data['fixed acidity'].min()) / (data['fixed acidity'].max() - data['fixed acidity'].min())
data['volatile acidity'] = (data['volatile acidity'] - data['volatile acidity'].min()) / (data['volatile acidity'].max() - data['volatile acidity'].min())
data['citric acid'] = (data['citric acid'] - data['citric acid'].min()) / (data['citric acid'].max() - data['citric acid'].min())
data['residual sugar'] = (data['residual sugar'] - data['residual sugar'].min()) / (data['residual sugar'].max() - data['residual sugar'].min())
data['chlorides'] = (data['chlorides'] - data['chlorides'].min()) / (data['chlorides'].max() - data['chlorides'].min())
data['free sulfur dioxide'] = (data['free sulfur dioxide'] - data['free sulfur dioxide'].min()) / (data['free sulfur dioxide'].max() - data['free sulfur dioxide'].min())
data['total sulfur dioxide'] = (data['total sulfur dioxide'] - data['total sulfur dioxide'].min()) / (data['total sulfur dioxide'].max() - data['total sulfur dioxide'].min())
data['density'] = (data['density'] - data['density'].min()) / (data['density'].max() - data['density'].min())
data['pH'] = (data['pH'] - data['pH'].min()) / (data['pH'].max() - data['pH'].min())
data['sulphates'] = (data['sulphates'] - data['sulphates'].min()) / (data['sulphates'].max() - data['sulphates'].min())
data['alcohol'] = (data['alcohol'] - data['alcohol'].min()) / (data['alcohol'].max() - data['alcohol'].min())
data['quality'] = (data['quality'] - data['quality'].min()) / (data['quality'].max() - data['quality'].min())

# visualizziamo i dati normalizzati
print(data)

# aggiungiamo 6 nuovi dati non normalizzatii
nuovi_dati = pd.DataFrame({
    'fixed acidity': [7.4, 7.8, 7.8, 11.2, 7.4, 7.9],
    'volatile acidity': [0.7, 0.88, 0.76, 0.28, 0.7, 0.6],
    'citric acid': [0, 0, 0.04, 0.56, 0, 0],
    'residual sugar': [1.9, 2.6, 2.3, 1.9, 1.9, 1.8],
    'chlorides': [0.076, 0.098, 0.092, 0.075, 0.076, 0.065],
    'free sulfur dioxide': [11, 25, 15, 17, 11, 10],
    'total sulfur dioxide': [34, 67, 54, 60, 34, 30],
    'density': [0.9978, 0.9968, 0.9970, 0.9982, 0.9978, 0.9965],
    'pH': [3.51, 3.20, 3.26, 3.16, 3.51, 3.40],
    'sulphates': [0.56, 0.68, 0.65, 0.58, 0.56, np.nan],
    'alcohol': [9.4, np.nan ,9.8 ,9 ,9 ,np.nan],
    'quality': [5 ,5 ,5 ,6 ,5 ,np.nan]
})

# applichiamo la regola della media per i valori NaN
nuovi_dati['fixed acidity'].fillna(nuovi_dati['fixed acidity'].mean(), inplace=True)
nuovi_dati['volatile acidity'].fillna(nuovi_dati['volatile acidity'].mean(), inplace=True)
nuovi_dati['citric acid'].fillna(nuovi_dati['citric acid'].mean(), inplace=True)
nuovi_dati['residual sugar'].fillna(nuovi_dati['residual sugar'].mean(), inplace=True)
nuovi_dati['chlorides'].fillna(nuovi_dati['chlorides'].mean(), inplace=True)
nuovi_dati['free sulfur dioxide'].fillna(nuovi_dati['free sulfur dioxide'].mean(), inplace=True)
nuovi_dati['total sulfur dioxide'].fillna(nuovi_dati['total sulfur dioxide'].mean(), inplace=True)
nuovi_dati['density'].fillna(nuovi_dati['density'].mean(), inplace=True)
nuovi_dati['pH'].fillna(nuovi_dati['pH'].mean(), inplace=True)
nuovi_dati['sulphates'].fillna(nuovi_dati['sulphates'].mean(), inplace=True)
nuovi_dati['alcohol'].fillna(nuovi_dati['alcohol'].mean(), inplace=True)
nuovi_dati['quality'].fillna(nuovi_dati['quality'].mean(), inplace=True)


# visualizziamo i nuovi dati
print('\nNuovi dati:')
print(nuovi_dati.to_string())

# normalizziamo i nuovi dati
for col in nuovi_dati.columns:
    if col in nuovi_dati.columns and pd.api.types.is_numeric_dtype(nuovi_dati[col]):
        min_val = nuovi_dati[col].min()
        max_val = nuovi_dati[col].max()
        nuovi_dati[col] = (nuovi_dati[col] - min_val) / (max_val - min_val)


print('\nNuovi dati normalizzati:')
print(nuovi_dati.to_string())

# aggiungiamo i nuovi dati normalizzati al dataset normalizzato
data = pd.concat([data, nuovi_dati], ignore_index=True)
# visualizziamo il dataset normalizzato
print('\nDataset normalizzato:')
print(data)

# Modelli di clustering

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data)
data['cluster'] = kmeans_labels

# Clustering gerarchico agglomerativo
agglo = AgglomerativeClustering()   
agglo_labels = agglo.fit_predict(data)
data['agglo_cluster'] = agglo_labels

# visualizziamo i cluster in un layout con due grafici appaiati
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# visualizziamo i cluster K-Means
sns.scatterplot(data=data, x='fixed acidity', y='alcohol', hue='cluster', palette='Set1', ax=ax1)
ax1.set_title('K-Means Clustering')
# visualizziamo i cluster Agglomerative
sns.scatterplot(data=data, x='fixed acidity', y='alcohol', hue='agglo_cluster', palette='Set1', ax=ax2)
ax2.set_title('Agglomerative Clustering')
plt.show()

# valutazione dendrogramma
from scipy.cluster.hierarchy import dendrogram, linkage 

# calcoliamo la matrice di linkage
Z = linkage(data, method='ward')
# visualizziamo il dendrogramma
plt.figure(figsize=(10, 7))
dendrogram(Z, leaf_rotation=90., leaf_font_size=12., color_threshold=0)
plt.title('Dendrogram')
plt.xlabel('Campioni')
plt.ylabel('Distanza')
plt.show()
