"""
Nome Script: tutto_python.py
Descrizione: Una serie di esercizi con le funzionalità di base di Python (variabili, if, while, print, ecc.) e l'utilizzo di due librerie
quali pandas e numpy
Autore: Antonio Napolitano
Versione: 1.0
Data: 25/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""

import pandas as pd
import numpy as np

# creiamo un dataset di esempio su cui fare degli esercizi
data = {
    'Nome': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Età': [25, 30, 35, 40, 45],
    'Città': ['Roma', 'Milano', 'Torino', 'Firenze', 'Venezia'],
    'Salario': [50000, 60000, 70000, 80000, 90000]
}
df = pd.DataFrame(data)
# Visualizziamo il dataset
print("Dataset originale:")
print(df)

# mostriamo solo la seconda colonna
print("\nSeconda colonna (Età):")
print(df.iloc[:, 1])

# mostriamo tutti gli elementi che hanno nella stringa la 'il'
print("\nElementi che contengono 'il' in tutto il dataset:")
print(df[df.apply(lambda x: x.astype(str).str.contains('il', case=False).any(), axis=1)])

# se esiste la città di Roma, mostra chi è il residente
print("\nResidente a Roma:")
print(df[df['Città'] == 'Roma'])

# sommiamo i salari
print("\nSomma dei salari:")
print(df['Salario'].sum())  

# facciamo la media del salario
print("\nMedia dei salari:")
print(df['Salario'].mean())

# stampiamo il salario minimo
print("\nSalario minimo:")
print(df['Salario'].min())

# cicliamo su ogni riga del dataset e stampiamo il nome e l'età
print("\nNome ed Età di ogni residente:")
for index, row in df.iterrows():
    print(f"Nome: {row['Nome']}, Età: {row['Età']}")

# applichiamo un ciclo while per stampare i nomi fino a quando non troviamo 'Charlie'
print("\nNomi fino a 'Charlie':")
i = 0
while i < len(df):
    if df.iloc[i]['Nome'] == 'Charlie':
        break
    print(df.iloc[i]['Nome'])
    i += 1

# città fino a 'Firenze'
print("\nCittà fino a 'Firenze':")
i = 0
while i < len(df):
    if df.iloc[i]['Città'] == 'Firenze':
        break   
    print(df.iloc[i]['Città'])
    i += 1

# stampiamo l'età media
print("\nEtà media:")
print(df['Età'].mean())

# mostriamo la riga corrispondente al nome Alice
print("\nSe esiste Alice:")
print(df[df['Nome'] == 'Alice'])

# cicliamo sul dataset e stampiamo nome e salario
for index, row in df.iterrows():
    print(f"Nome: {row['Nome']}, Salario: {row['Salario']}")

# applichiamo un esercizio con numpy per calcolare la somma dei salari
print("\nSomma dei salari usando NumPy:")
salari_array = np.array(df['Salario'])
somma_salari = np.sum(salari_array)
print(somma_salari)