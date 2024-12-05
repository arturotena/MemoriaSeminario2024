# Hoy en el seminario:

# Ultimos algoritmos de aprendizaje aprendizaje para luego pasar a no supervisado.
# 
# **Para el 14 de diciembre:**
# 
# al 90% el EDA terminado
# y algún algoritmo de aprendizaje
# 
# La idea de ML es como exprimir los datos (mineria de datos), que aportemos de acuerdo al conocimiento del negocio, porque se eligio el algoritmo.
# 
# 
# La SVM Maquina de Soporte de Vectores fue echa para el caso binario... 
# 
# Hay clasificador que hace 1 contra todos, o todos contra todos.
# Entrega los hiperplanos que cortan cada clase.
# 
# 
# Una RVM Maquina de Vectores Relevante es clasificador probabilisstico y los modelos entrenados se vuelven matriz de tipo sparse.
# 
# 
# 
# Se vera no clasificador sino regresion:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR # la maquina de soporte de regresion

# tecnicas para crear datos sinteticos:
# Interpolacion
# Bilineal (promedio de 2 puntos)
# Lanzos




X=np.sort(5*np.random.rand(40,1),axis=0) # una colummna
y=np.cos(X).ravel()

model=SVR(kernel='linear') # w_i*x+b
model.fit(X,y) # sin hacer split porque son poquitos datos
X.shape
y.shape

# no se debe hacer: predecir lo que ya tengo
# en el caso de la lineal debe imprimir la linea a que se parece
y_pred=model.predict(X)

plt.scatter(X,y,c='b')
plt.scatter(X,y_pred,c='r')
plt.show()
plt.close()






X=np.sort(5*np.random.rand(1000,1),axis=0) # una colummna
y=np.cos(X).ravel()

model=SVR(kernel='linear') # w_i*x+b
model.fit(X,y) # sin hacer split porque son poquitos datos
X.shape
y.shape

# no se debe hacer: predecir lo que ya tengo
# en el caso de la lineal debe imprimir la linea a que se parece
y_pred=model.predict(X)

plt.scatter(X,y,c='b')
plt.scatter(X,y_pred,c='r')
plt.show()
plt.close()

# 
# 
# si podria verse como que el hiperplano intenta dividir los puntos en dos planos: los valores del coseno que son positivos y los negativos
# 
# 
# 
# 







# ----------------

X=np.sort(25*np.random.rand(500,1)-10,axis=0) # una colummna
y=np.cos(X).ravel()
plt.scatter(X,y,c='b')
plt.show()
plt.close()

model2=SVR(kernel='rbf', gamma='scale')
model2.fit(X,y)
y_pred2=model2.predict(X)
plt.scatter(X,y,c='b')
plt.scatter(X,y_pred2,c='r')
plt.show()
plt.close()

model2=SVR(kernel='rbf', gamma='auto')
model2.fit(X,y)
y_pred2=model2.predict(X)
plt.scatter(X,y,c='b')
plt.scatter(X,y_pred2,c='r')
plt.show()
plt.close()



model3=SVR(kernel='poly')
model3.fit(X,y)
y_pred3=model3.predict(X)
plt.scatter(X,y,c='b')
plt.scatter(X,y_pred3,c='r')
plt.show()
plt.close()



# simula una red neuronal en la ultima capa
model4=SVR(kernel='sigmoid')
model4.fit(X,y)
y_pred4=model4.predict(X)
plt.scatter(X,y,c='b')
plt.scatter(X,y_pred4,c='r')
plt.show()
plt.close()

model4=SVR(kernel='sigmoid', gamma='auto')
model4.fit(X,y)
y_pred4=model4.predict(X)
plt.scatter(X,y,c='b')
plt.scatter(X,y_pred4,c='r')
plt.show()
plt.close()


# datos con ruido
..
...


# =============
# =============


boston tiene dimensionalidad de 13

from sklearn.datasets import load_boston

import pandas as pd
data_url = "http://lib.stat.cmu.edu/datasets/boston"
df=pd.read_csv(data_url,delim_whitespace=True, skiprows=21, header=None)

df.describe()

df.columns

aux=np.array(df)
aux

df

csv

import os
os.getcwd()
os.chdir('Downloads')
os.chdir('MemoriaSeminario2024')

df=pd.read_csv('boston.csv')
df.shape
df.columns



df.columns.size

df=df.drop(['Unnamed: 0'],axis=1)

df.columns






from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30, shuffle=True)

model=SVR(gamma='scale')

from sklearn.model_selection import cross_val_score

score=cross_val_score(model,X_train,y_train,cv=10) #cv es la validacion cruzada, los divide en 10, valida como se comporta con conjuntos pequeños comparado con grupos grandes

print(score)
print(score.mean()) # coeficiente de determinacion
print(score.std())

model.fit(X_train,y_train)
y_pred=model.predict(y_test)
...
...

...

