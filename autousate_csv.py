"""
Nome Script: autousate_csv.py
Descrizione: Creazione del csv per creare un archivio di auto usate. Il file CSV contiene le seguenti colonne: anno, chilometraggio, cilindrata e prezzo.
Il file CSV viene creato con la libreria pandas e salvato in un file chiamato "auto_usate.csv". Il file CSV viene poi caricato in un DataFrame di pandas per essere visualizzato.
Autore: Antonio Napolitano
Versione: 1.0
Data: 11/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""

import pandas as pd
import numpy as np

# generiamo dati casuali per riprodurre il dataset in modo corretto
np.random.seed(42)

# Creazione di un DataFrame con dati casuali
# la cilindrata deve influenzare positivamente il prezzo: più è alta più il prezzo aumenta
# l'anno deve influenzare negativamente il prezzo: più è vecchia meno vale
# il chilometraggio deve influenzare negativamente il prezzo: più è alta meno vale
data = {
    "anno": np.random.randint(2000, 2023, size=1000),
    "chilometraggio": np.random.randint(50000, 200000, size=1000),
    "cilindrata": np.random.randint(1000, 5000, size=1000),
}

# Creazione del prezzo in base ai dati
data["prezzo"] = (
    20000 # valorei di riferimento
    - (2023 - data["anno"]) * 1000 # differenza tra anno di riferimento (2023) e anno di immatricolazione. 
    # IL prezzo viene diminuito di 1000 per ogni anno di vecchiaia e sottratto al prezzo base
    - (data["chilometraggio"] / 1000) * 0.5 # per scalare il chilometraggio. Si moltiplica per 0.5 e si sottrae per fare in modo che
    # maggiore è il chilometraggio, minore è il prezzo
    + (data["cilindrata"] / 1000) * 1.5 # per scalare la cilindrata. Si moltiplica per 1.5 e si aggiunge al prezzo base 
    # in modo da aumentare il valore dell'auto all'aumentare della cilindrata
)
# il prezzo minimo deve essere 2500
data["prezzo"] = np.clip(data["prezzo"], a_min=2500, a_max=None)
# Aggiunta di rumore al prezzo
data["prezzo"] += np.random.normal(0, 2000, size=1000)
# Creazione del DataFrame
df = pd.DataFrame(data)
# Visualizzazione delle prime righe del DataFrame
print(df.to_string())

#Salviamo il DataFrame in un file CSV
df.to_csv("auto_usate.csv", index=False, encoding="utf-8")
print("File CSV creato con successo: auto_usate.csv")
