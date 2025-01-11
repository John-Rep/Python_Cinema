
#Commencer Exercice #1

import pandas as pd
import numpy as np
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

#Grouper les lignes de data par commune
data_com = data.groupby('commune').sum().reset_index()

#Calculer les entrées moyennes par fauteuil pour chaque commune
data_com['entrées par fauteuil'] = data_com['entrées 2022'] / data_com['fauteuils']
#Afficher les données des trois communes avec le plus et le moins d'entrées par fauteuil
print(data_com.nlargest(3,'entrées par fauteuil'))
print(data_com.nsmallest(3,'entrées par fauteuil'))

#Séparer les 10 communes avec le plus d'entrées par fauteuil et faire un graphique
data_epf = data_com.nlargest(10, 'entrées par fauteuil')
epf = data_epf.plot(kind = 'bar', x='commune', y='entrées par fauteuil', figsize=(12,6), rot = 45, legend=False)
#Ajuster l'affichage du graphique pour le rendre plus lisible
epf.set_xlabel('Nom de la commune', fontsize=16)
epf.set_ylabel('Entrées par fauteuil', fontsize=16)
epf.set_title('Entrées par fauteuil des communes', fontsize=20)
plt.setp(epf.get_xticklabels(), ha='right')
plt.subplots_adjust(bottom=.4)
plt.show()


#Commencer Exercice #3

#Les données contiennent déjà que les valeurs de 2022

#Calculer les Corrélations entre écrans / fauteuils et les entrées pour l'année 2022
ecran_corr = data['écrans'].corr(data['entrées 2022'])
faut_corr = data['fauteuils'].corr(data['entrées 2022'])
#Afficher les valeurs calculées
print("Corrélation entre écrans et entrées en 2022 : {}".format(ecran_corr))
print("Corrélation entre fauteuils et entrées en 2022 : {}".format(faut_corr))

#Transformer les entrées pour rendre les valeurs plus lisibles dans un graphique
data['entrées 2022 plot'] = data['entrées 2022'] / 1000000
#Générer un nuage de points pour écrans vs entrées
data.plot(kind='scatter', x='écrans', y='entrées 2022 plot', alpha=.3,
          title="Entrées par écrans en 2022", ylabel="Entrées (Millions)",
          xlabel="Écrans")
#Calculer et ajouter la ligne de régression linéaire
m, b = np.polyfit(data['écrans'], data['entrées 2022 plot'], 1)
plt.plot(data['écrans'], m * data['écrans'] + b)
plt.show()
#Générer un nuage de points pour fauteuils vs entrées
data.plot(kind='scatter', x='fauteuils', y='entrées 2022 plot', alpha=.3,
          title="Entrées par fauteuils en 2022", ylabel="Entrées (Millions)",
          xlabel="Fauteuils")
#Calculer et ajouter la ligne de régression linéaire
m, b = np.polyfit(data['fauteuils'], data['entrées 2022 plot'], 1)
plt.plot(data['fauteuils'], m * data['fauteuils'] + b)
plt.show()