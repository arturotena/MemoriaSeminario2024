# R
# library(reticulate)
# py_install("pandas")
# py_install("matplotlib")
# py_install("scikit-learn")
# py_install("seaborn")
# py_install("imbalanced_learn") # metricas clasificador, genera datos faltantes
# reticulate::virtualenv_install(packages = c("numpy==1.8.0"))
# system2(reticulate::py_exe(), c("-m", "pip", "uninstall -y", 'scikit-learn'))

# continuamos aprendizaje supervizado
# el punto es que siempre se manejan etiquetas,
# siempre se tienen etiquetas, categorías
# depende del algoritmo, ej. 0,1, o -1,1
# para separarlos en clases.

# hemos visto bayes max a posteriori, y las redes neuronales

# como si fueran regresores en lugar de etiquetas se vera en la siguiente clase


# Algoritmo> maquina de soporte de vectores
# Hoy entender que hace la maquina de soporte de vectores.
# Probarlo antes que la r.n. o incluso Bayes.
# Para espacios n-dimensionales
# Es algoritmo clasico avanzado
# Es como la union de r.n. con la probabilistica
# Tip, si no salen clasificados tus datos entonces se necesitan depurar mas los datos, porque otro algoritmo podria clasificar mal. Es decir, regresarse al EDA.
# Versatil, tiene diferentes funciones.
# Seusan 3 kernels normalmente: Normal, NRF, Polinomial

# Otra forma de EDA> algoritmo de proyeccion PCA
# datos n-dimensionales para entenderlos.



###############
# Alg. supervisado: maquina de soporte de vectores
# m.s.v. = msv

from sklearn.datasets import load_iris
iris=load_iris(as_frame=True)
print(iris.keys())

import seaborn as sns

type(iris.frame)
type(iris.frame['target'])
iris.frame["target"]=iris.target_names['iris.target']
_=sns.pairplot(iris.frame, hue='target')
plt.show()
plt.close()

# pairplot muestra funciones de distribucion de probablidad, y de dispersion

la primera grafica de la diagonal muestra que hay inferencia entre las variables (se sobrepoinen): a partir de una variable se puede inferir otra: a ciertos clasificadores les costara trabajo separarlos.
esto se llama extraccion de caracteristicas o atributos importantes

por eso desde el EDA se pueden ver cuales variables pueden ayudar a separar

en la 2a grafica la inferencia crecio

[la diagonal muestra la inferencia de las otras variables]

# ahora se vera tridimensionalmente
# se analizara la dimensionalidad de los datos

# msv analiza donde cortar

# Dimensionalidad PCA (analisis de componentes principales), hace EDA
# esto es extraccion de caracteristicas
# el problema es entender los componentes principales
# transforma espacios, ndimensional a unidimensional:
#     T R^N : R^1
#     transforma espacio vectorial a unidimensional
# el primer componente sera siempre el que tiene mayor varianza, el componente que tiene la mayor varianza nos da el que fluctua mas, y los siguientes disminuyen poco a poco su varianza
# normalmente se ocupan 3 componentes
#
# algoritmo:
1. Transfornar los datos, siempre hay que estandarizar.
        por medio de la Z-transform
           que dice: Z_i = (X_i - mu_x) / sigma_
                mu_x = media de X
                sigma_x = desviacion estandar de X
            =>finalidad se centre como una gaussiana
2. Obtener la matriz de covarianza
        mide la varianza entre 2 variables
        Cov(X1,X2)=np.cov()
3. Obtener los Eigenvectores y Eigenvalores
        (algoritmo Graham-Shmitt) Descomponsicion de una matriz
        AX=lambdax
        lambda es el eigenvalor y x es el eigenvector

Variantes del PCA:
    LDA=Analisis de Descomposicion Lineal
    ZCA = Analisis de Componentes con Cero media (centrados)
    ICA Analisis de Componentes Independientes

https://es.wikipedia.org/wiki/Álgebra_lineal

ICA es el mas poderoso

# Método Fácil

import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d

from sklearn.decomposition import PCA

X_pca=PCA(n_components=3).fit_transform(iris.data)
# el numero de componentes depende del numero de columnas, el maximo son el numero de columnas que tienen los datos

fig=plt.figure(1,figsize=(19,6))
ax=fig.add_subplot(111,projection='3d',elev=-150,azim=110)
scatter=ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    X_pca[:,2],
    c=iris.target,
    s=25)
legend1=ax.legend(scatter.legend_elements()[0],iris.target_names.tolist(),loc='lower right',title='Clases')
ax.add_artist(legend1)


en la grafica de 3d, cada dimension es un eigenvector

permite ver los datos que pertenencen a clases que tienen inferencia

# varianza explicativa
import numpy as np
for i in range(3):
    print(np.var(X_pca[i]))

# Metodo largo, PC con numpy
# 1. haciendolo poco a poco, replicando el algoritmo:
print(iris.data)
X=iris.data
X_mean=np.mean(X)  # X.mean()
X_std=np.std(X)    # X.std()
# Estandarizacion
# Z_i=(X_i-mu_x)/std_x
Z=(X-X_mean)/X_std
print(Z)
# cambiaron los valores de acuerdo a la media
# estan normalizados con la transforacion Z
# Se le llama Wittening (blanque de datos), eliminacion de media, media igual a cero como en una gaussiana
# 2. Luego, calcular la matriz de covarianza
c_x=X.cov()
c_z=Z.cov()
plt.subplot(121)  # filas,columnas,elemento
sns.heatmap(c_x)
plt.title('Cov X')
plt.subplot(122)  # filas,columnas,elemento
sns.heatmap(c_z)
plt.title('Cov Z')
plt.show()
plt.close()
# viendo la matriz de correlacion
corr_x=np.corrcoef(X)
corr_z=np.corrcoef(Z)
plt.subplot(121)  # filas,columnas,elemento
sns.heatmap(corr_x)
plt.title('Corr X')
plt.subplot(122)  # filas,columnas,elemento
sns.heatmap(corr_z)
plt.title('Corr Z')
plt.show()
plt.close()


......

La SVM trata siempre de encontrar un hiperplano de separabilidad.
Si no lo encuentra, tus datos estan mal.
Lineal
Kernel RBF (Gaussian)
Polinomial 
