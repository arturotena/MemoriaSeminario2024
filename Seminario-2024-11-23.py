# R
# library(reticulate)
# py_install("pandas")
# py_install("matplotlib")
# py_install("scikit-learn")
# py_install("seaborn")
# py_install("imbalanced_learn") # metricas clasificador, genera datos faltantes
# reticulate::virtualenv_install(packages = c("numpy==1.8.0"))
# system2(reticulate::py_exe(), c("-m", "pip", "uninstall -y", 'scikit-learn'))




# Regresores basados en funciones probabilisticas
# (funciones gausianas)

# p(a|b)=( P(b) P(b|a) ) / P(a)

# proceso gaussiano

# muestras -> datos -> unidimensionales, bidim, multi (max 3,4), hiperdimensional

# funcion deterministica, tiene una funcion matematica, siempre sabemos el comportamiento.
# en la vida real son funciones estocásticas, no tienen función matemática, pero se pueden idealizar con una PDF de tipo Gaussiano. (ej, precios de acciones es estocástico)

# ejemplo de funcion probabilistica:
import numpy as np
def deterministica(x):
    y=0.5+np.sin(x*5)
    return y
import matplotlib.pyplot as plt

x=(np.linspace(0,10,100)  # 100 números de 0 a 10]
    .reshape(-1,1))  # (n renglones, 1 columna)
y=deterministica(x)
plt.plot(x,y)
plt.show()
plt.close()

# las regresiones hacen que los datos se parezcan a una funcion matematica

# proceso gausiano: que todo esté en función de la normal:
# f(x)=(W_i)(x)+b   ---se parezca a ---> p(f|x)=N(f|mu,K) -->  N() es la funcion normal, mu es la media, K es la desviación estándar o varianza
# funcion de densidad de probabilidad:
# N()=1/2*sqrt(pi)*sigma*exp(x-mu)^2/sigma^2
# https://www.gstatic.com/education/formulas2/553212783/es/normal_distribution.svg

x_train=(np.random.uniform(0,10,100) # 100 datos aleatorios uniformes de 0 a 100
    .reshape(-1,1)) # pasar de fila a columna
y_train=deterministica(x_train)
plt.plot(x,y,c='orange')
plt.scatter(x_train[:,0],y_train)
plt.show()
plt.close()
# la regresión probabilistica nos permite saber si los datos de nuestra muestra (en este caso azules) se parecen a una función (por ejemplo al seno naranja en este caso)


# haciendo regresión:
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF # Funcion de Base Radial --> se parece a la Gaussiana
kernel=1.0*RBF()
# RBF ya tiene sus hiperparámetros predeterminados pero se pueden optimizar
# bloque de procesamiento:
model=GaussianProcessRegressor(kernel=kernel)
model.fit(x_train,y_train) # intentara el regresor ajustarse a los datos reales (que son aleatorios uniformes que en realidad se parecen a la funcion senoidal y no a la funcion gausiana)
# Muestra: ConvergenceWarning: lbfgs failed to converge (status=2):
# ABNORMAL_TERMINATION_IN_LNSRCH.
# significa que no convergen
# Puede ser que falten datos o los parametros de la funcion deberian ser diferente.
# se puede por ejemplo crear nuestra propia funcion de iteraciones
# otra solucion que dice el warning es escalar de los datos
#
# mu deberia ser 0.5, y sigma aproximadamente igual a 1, dados los datos que tenemos (viendo la linea narnja)
y_mu,y_std=model.predict(x,return_std=True) # ¿se parece e la funcion x (que en la realidad no tenemos, pero es el ejemplo, la funcion x es la senoidal naranja)?
print(y_mu,y_std) # son las estimaciones
# usar la std cuando los datos no estan normalizados
# usar la varianza cuando los datos estan normalizados
plt.subplot(1,2,1)
plt.plot(x,y,color='orange') # funcion a a proximar (senoidal)
plt.subplot(1,2,2)
plt.scatter(x_train[:,0],y_train,color='black') # datos reales
plt.errorbar(x,y_mu,y_std)
plt.show()
plt.close()
# el regresor aprendio muy bien porque los datos son perfectos

# ruido:
# y=0.5*sin(3x)  ---> en la realidad: y(x)=0.5*sin(3x)+ruido(x)
# ruido multiplicativo y ruido aditivo
# en ingenieria es el filtrado, y en machine learning es EDA en donde se quitan los datos no significativos
# en la vida real hay ruido en las señales (datos), ejemplo el ruido en la Televisión antigua

 #----------------------------
# generaremos datos con ruido

x=(np.linspace(0,10,100)  # 100 números de 0 a 10]
    .reshape(-1,1))  # (n renglones, 1 columna)
n=np.random.normal(0,0.3,100).reshape(-1,1) # ruido: 100 datos con media,varianza
y=deterministica(x)+n # datos 
plt.plot(x,y)
plt.show()
plt.close()


y_mu,y_std=model.predict(x,return_std=True) 
print(y_mu,y_std) # son las estimaciones
plt.subplot(1,2,1)
plt.plot(x,y,color='orange') # funcion a a proximar (senoidal)
plt.subplot(1,2,2)
plt.scatter(x_train[:,0],y_train,color='black') # datos reales
plt.errorbar(x,y_mu,y_std)
plt.show()
plt.close()
# el regresor aprendio muy bien porque los datos son perfectos
# esta sobreentrenado

# un regresor de proceso gausiano predice la media condicional... (f|u,K).... K es la matriz de covarianza: [varx,covxy][covxy,vary]

# ver tambien https://scikit-learn.org/1.5/auto_examples/gaussian_process/plot_gpr_noisy_targets.html

# ya terminamos los regresores
# =========================================================
# =========================================================
# =========================================================
# =========================================================
# =========================================================
# =========================================================
# =========================================================
# =========================================================


# Modelos de clasificación supervisados

# En la clasificacion la etiqueta son variable categorica
# (en los regresores son valores numericos)

# Bayes piensa que todo es distribucion gaussiana

# -------------------------------------------------
# Naive bayes classifier (también llamado MAP, estimador de maximo a posteri)
# (obtiene el máximo de la distribución)
# P(y|x)=P(x&y)/P(x) --> Probabilidad a posteriori = probabilidad condicional * Probabilidad a priori / Probabilidad de los datos que estabilicen el sistema

# y # etiqueta
# y=argumento_maximo(P(x tal que y) por (y)) --> argumento maximo es la probabilidad maxima que tengamos y la suma de las probabilidades debe ser 1

from sklearn.datasets import load_iris
data=load_iris()

import matplotlib.pyplot as plt

plt.scatter(data.data[:,0],data.data[:,1]) # datos, 
plt.show()
plt.close()
# sin ver las etiquetas

plt.scatter(data.data[:,0],data.data[:,1], c=data.target) # datos, etiquetas
plt.show()
plt.close()
# hay 3 etiquetas o clases
# la pregunta es como el algoritmo encuentra la clase que le corresponde

# x son los datos reales
x=data.data
# y son las etiquetas
y=data.target
for i in range(len(y)):
    print(x[i],y[i])

# queda en pausa para la siguiente sesion hacer el EDA

# divide para entrenar y test
from sklearn.model_selection import train_test_split
x_train,x_test, ytrain,ytest = (
  train_test_split(x,y,test_size=0.3,shuffle=True))
for i in range(5):
  print(x_train[i],y_train[i])

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
# regresa warning

y_preds=model.predict(x_test)
for i in range(len(y_test)):
    print(y_test[i],y_preds[i])

plt.subplot(121)
plt.scatter(x_test[:0],x_test[:1],c=y_preds)
plt.title('Prediction')
plt.subplot(122)
plt.scatter(x_test[:0],x_test[:1],c=y_preds)
plt.title('Truth')
plot.show()
plot.close()
# erró en 3

# metrica de calidad
from sklearn.metrics import classification_report
print(classification_report(y_preds,y_test))
