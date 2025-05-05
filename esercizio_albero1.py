"""
Nome Script: esercizio_albero1.py
Descrizione: Un esempio molto basilare di come creare e utilizzare un albero decisionale per la classificazione. Utilizziamo il dataset iris.csv.
Autore: Antonio Napolitano
Versione: 1.0
Data: 02/05/2025
Copyright: © Antonio Napolitano 2025
Licenza: MIT
"""

from sklearn.datasets import load_iris # importa il csv iris per l'esempio
from sklearn.model_selection import train_test_split # importa il modulo di addestramento
from sklearn.tree import DecisionTreeClassifier # importa il modulo per applicare l'albero decisionale
from sklearn.metrics import accuracy_score # importa il modulo per valutare l'accuratezza del modello di previsione
# libreria aggiuntiva per visualizzare l'albero decisionale
from sklearn.tree import export_graphviz
import graphviz

# 1. Carichiamo il dataset
iris = load_iris()
# stabiliamo i valori di input e output
X = iris.data # qui vengono caricate le caratteristiche (lunghezza e larghezza di sepali e petali)
y = iris.target # qui viene mostrata la variabile di target (che tipo di fiore è: 0, 1, 2)

# 2. Dividiamo i dati in set di training e set di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Creiamo il modello di albero decisionale per la classificazione
albero = DecisionTreeClassifier(random_state=42)

# lo alleniamo
albero.fit(X_train, y_train)

# facciamo le previsioni sui dati di test
y_predict = albero.predict(X_test)

# 4. Valutazione dell'accuratezza del modello
accuratezza = accuracy_score(y_test, y_predict)
print(f"Accuratezza del modello (1 = Ottimo; 0 = Male): {accuratezza:.2f}")

# 5. Visualizziamo l'albero decisionale con la libreria graphviz in formato DOT
dot_data = export_graphviz(albero, out_file=None, # albero è l'oggetto precedentemente addestrato con scikit-learn; out_file è il percorso del file (in questo caso nessuno)
                         feature_names=iris.feature_names, # la lista degli indici delle colonne del dataset iris (sepal length, ecc..)
                         class_names=iris.target_names, # i nomi delle classificazioni, setosa, versicolor, virginica (invece che 0, 1, 2)
                         filled=True, rounded=True, # filled colora i nodi-foglia in base alla classe predominante; rounder disegna i bordi
                         special_characters=True) # permette la gestione corretta dei caratteri speciali (come > o =)
graph = graphviz.Source(dot_data) 
graph.render("iris_decision_tree") # Salva l'albero in un file PDF
graph # Visualizza l'albero direttamente (in alcuni ambienti)
