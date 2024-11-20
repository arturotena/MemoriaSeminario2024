import os
os.chdir('D:/Proyectos/RStudioProyectos')
import pandas as pd
df=pd.read_csv('50.csv')

df

# Regresión logística
# también se puede entrenar para clasificar una serie de observaciones

from sklearn.datasets import load_digits # una de tantos datasets que contiene
# https://en.wikipedia.org/wiki/MNIST_database
data=load_digits()
print(data.data.shape)

import matplotlib.pyplot as plt
plt.imshow(data)
plt.show()

print(data.data.shape)
plt.imshow(data.data[1])
plt.show()


# --------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits # una de tantos datasets que contiene
data=load_digits()
plt.figure(figsize=(20,4))
for idx,(img,label) in enumerate(zip(data.data[0:10],data.target[0:10])):
  plt.subplot(1,10,idx+1)
  plt.imshow(np.reshape(img,(8,8)),cmap='gray')
  plt.title('Label: %i\n' %label,fontsize=12)
  plt.axis('off')
plt.show()
plt.close()

print(data.target) # son etiquetas

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
  data.data,data.target,test_size=.45)
# verificar el tamaño para ver que lo hizo bien
X_train.shape
X_test.shape
y_train.shape
y_test.shape

from sklearn.linear_model import LogisticRegression
# sirve tanto para predecir etiquetas como variable contínua
# la regresión lineal busca satisfacer la ecuacion de una linea
# la regresión logistica busca satisfacer la ecuacion sigmoide https://es.wikipedia.org/wiki/Funci%C3%B3n_sigmoide
# la regresión logistica se ocupa para pocos datos menores a 1 millón
# la regresión logistica es muy variable, puede fallar si no se cumplen las precondiciones de los datos
# la regresión logistica tiene problema con las filas (colas),
#    si están muy alejados de la media

logistic=LogisticRegression()
# multi_class=ovr todos los datos para hacer regresion de 1
logistic.fit(X_train,y_train)

plt.imshow(np.reshape(X_test[0],(8,8)))
plt.show()
plt.close()
logistic.predict(X_test[0].reshape(1,-1)) # predice que digito es la figura mostrada arriba

# se puede mostrar las probabilidades de cada etiqueta
print(logistic.predict_proba(X_test[0].reshape(1,-1)))
# mostrar la máxima verosimilitud 
print(logistic.predict_log_proba(X_test[0].reshape(1,-1)))

# ejemplo de la diferencia entre la linea y la sigmoidea
x=np.linspace(-1,1,100)
y=x+0.001
yy=1/1+np.exp(-0.001+x)  # eq sigmoide
plt.plot(y,x)
plt.plot(yy,x)
plt.show()
plt.close()


## Métricas de calidad
# Saber cómo es mi clasificador: bueno o malo.

# predecir los demás
y_preds=logistic.predict(X_test)  # o y_estimada
y_preds

# obtener el score general del predictor
score=logistic.score(X_test,y_test)
print(score) # muestra 0.97, falla el 3%

# la matriz de confusión muestra en donde 
# se equivoco el clasificador

from sklearn import metrics
import seaborn as sns
cm=metrics.confusion_matrix(y_test,y_preds) # lo que sabemos que es correcto y las predecidas
print(cm) #intenta ser una variable identidad, donde no es cero es error
plt.figure()
#sns.heatmap(cm,square=True,cmap='cool')
sns.heatmap(cm,annot=True,fmt='.2f',square=True,cmap='hot')
plt.ylabel('Etiqueta real y_test')
plt.xlabel('Etiqueta predicha y_preds')
plt.show()
plt.close()
# permite ver en qué clase o etiqueta se está equivocando más
# y después preguntarnos porqué falla

# cómo obtener el accuracy de la predicción
# es decir toda la precisión del modelo
# Opción 1= TP,TN, FP,FN = TruePositive,TrueNegative, FalsePositive,FalseNegative
print(metrics.accuracy_score(y_test, y_preds))


#########################################################
#########################################################
# metricas para clasificacion:
# accuracy sensibility specificity

# metricas para regresion: error cuadratico medio , R2, el valor Dcuadrada
#########################################################
#########################################################


print(metrics.recall_score(y_test,y_preds))
# pero no es aplicable asi porque no es comparacion 1 contra 1
# sino 1 contra todos

print(metrics.classification_report(y_test,y_preds))
# reportar:
# weighted avg
# accuracy es metrica accuracy - qué tanto sí acertó el modelo
# precision es metrica sensibilidad - que tanto tu modelo confía en que hizo la clasificación en cierta clase 
# recall es metrica especificidad
# f1-score media armonica entre precission y recall

# precision sensilbilidad a los datos
# accuracy es la precision del modelo
# recall o especificidad en que clase está fallando más

# se calculan a partir del conteo de casos TP,TN, FP,FN

# problema de machine learning: support varía mucho por clase dependiendo de cuantos elementos se entrenan y su distribucion

# 
