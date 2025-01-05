
#Commencer Exercice #1

import pandas as pd
data = pd.read_csv("cinemas.csv")
pd.set_option("display.max_columns", None)
print(data.isnull().sum())
print(data.describe())

#Supprimer les lignes (y en a 8) où il n'y a pas de valeur pour les entrées de 2021
#Cela permettra de créer le modèle prédictif basé sur 2021 et 2022
data = data[~data['entrées 2021'].isnull()]

#Supprimer les colonnes AE et multiplexe car elles ne contiennent que les valeurs NaN
data = data.drop(columns=['AE','multiplexe'])

#Supprimer les colonnes label Art et Essai et programmateur qui contiennent principalement les valeurs NaN
data = data.drop(columns=['label Art et Essai','programmateur'])

#Supprimer Longitude et Latitude qui sont déjà compris dans la colonne geolocalisation
data = data.drop(columns=['longitude','latitude'])

print(data.loc[:,'région administrative'].unique())
#Supprimer région administrative car toutes les lignes ont la même valeur
data = data.drop(columns='région administrative')

print(data.head())
print(data.loc[:,['écrans','fauteuils','entrées 2022','entrées 2021']].describe())


