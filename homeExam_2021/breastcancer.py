# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


#OPPGAVE 3a) Steg 1 Leser inn datasett,setter id- til index og vis dimensjoner og topprader for å bli kjent med data
df = pd.read_csv('breastcancer.csv', sep=",", encoding="utf-8",index_col="id")
print("# of rows and cols:", df.shape, "\n")
print("Tre første rader:", df.head(3))
print("Kol-navn:", df.columns)

# Sjekker for hver kolonne hva som finnes av data (kunne gjort manuell inspeksjon, men vanskelig jo større filen er)
for cols in df.columns:
    print("For kolonne" , cols, ":" , Counter(df[cols].values))

#Ser at datasettet har ingen NaN-verdier slik at jeg kan kommentere ut linjen under (droppe dropna)
#df = df.dropna()
#Men ser at jeg må fjerne de radene som har ? i segi bare_nuclei-kolonnen
filer = (df["bare_nuclei"].isin(["?"]))
df = df[~filer]

#Konverter til numeric til slutt
df = df.apply(pd.to_numeric)


#OPPGAVE 3b) DECISION TREE CLASSIFIER
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

#Ingen kategorisk data, men kun numerisk, så slipper å kjøre konvertering
#Splitter datasettet vertikalt i forklaring og prediksjonsvariabel
y = df["classes"]
X = df.drop("classes", axis = 1)

#c) Splitter datasettet horisontalt med Sklearn-metoden train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Steg 3: Lag et tre med treningsdata der maksdybde og gini settes
classifier = DecisionTreeClassifier(min_impurity_decrease=0.01, max_depth=3)
classifier.fit(X_train, y_train)

# Steg 4: Test treet og vis klassifiseringsrapport
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Steg 5: Visning av treet
dotfile = open("dtree1.dot", "w")
dotfile = tree.export_graphviz(classifier, out_file = dotfile, feature_names = X.columns, class_names=["0","1"])