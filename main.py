
#Commencer Exercice #1

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

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



#Commencer Exercice #2

data_com = data.groupby('commune')
print(data_com.describe())
data['entrées par fauteuil'] = data['entrées 2022'] / data['fauteuils']
print(data.nlargest(3,'entrées par fauteuil'))
print(data.nsmallest(3,'entrées par fauteuil'))


data_epf = data.nlargest(10, 'entrées par fauteuil')
epf = data_epf.plot(kind = 'bar', x='nom', y='entrées par fauteuil', figsize=(12,6), rot = 45, legend=False)
epf.set_xlabel('Nom du cinéma', fontsize=16)
epf.set_ylabel('Entrées par fauteuil', fontsize=16)
epf.set_title('Entrées par fauteuil des cinémas', fontsize=20)
plt.setp(epf.get_xticklabels(), ha='right')
plt.subplots_adjust(bottom=.4)
plt.show()
