# KMeans esta diseñada para la distancia ecuclidiada

# C-Means no se basa por las medias, sino por reglas de
# decisión  de lógica difusa. El problema es entenderla,
# y que la regla de decisión sea compatible con todos.


# Distancia Manhatan = el absoluto de la distancia
# Distancia Chebyshev = el maximo de la distancia

from scipy.spatial.distance import cdist
from scipy.spatial import distance
import numpy as np

import

x = data.iloc[:,1:3]
....

# Se minimiza la varianza en el metodo del codo,
# modernamente no funciona si se cambia la distancia
# porque no es la media de la distancia.
# Mejor, usar el numero de clusters del codo
# y usar ese numero al usar otra distancia diferente.


# La tercera opción es la distancia de Mahalanobis.
# Hay otras distancias, depende la aplicación.
# Mahalanobis es probabilistica, lo que obtiene es
# intentar tener las distancias entre los puntos
# a partir de W que es la tranfromación Z y
# obtiene la matriz de covarianza y su inversa,
# para normalizar los datos a 1.
# Se ocupa para segmentar de acuerdo a la distancia
# entre las varianzas de la distribución de los datos.
# Elije al azar los centroides y calcula su probabilidad
# y en algunos casos no existe y no existen y por
# tanto no funciona el algoritmo.
# Por ello mucho dicen que no sirven para usarla en KMeans.

##############################
# Otro algoritmo: DBSCAN
# predecesor de KMeans
#publicado en los 90s
#a partir de un punto aleatorio encuentra los puntos mas cercanos
#en una circunferencia
#Puede clasificar mal si los datos no están en circunferencia.
# Sirve cuando al graficar los datos están muy separados, si están
# muy juntos no puede encontrar las circunferencias.


##########################################
# Algoritmo TSNE
# El actual, ventajas de kmeans, pca
# Segmenta o clasifica de forma estadística y estocástico (no determinista, al azar).
# TSNE ~= T distribution Stocastic Near Embeddings

# probabilidad conjunta a condicional

# la perplejidad es el 5o momento de la distribucion
# es un valor para saber si la distribucion es muy robusta o delagada
#
