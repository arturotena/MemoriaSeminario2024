# recordar que solo en el predict ven los datos de test

# ver que datos no clasifico correctamente
# los mas facil es ver las etiquetas predichas
# contra las reales (si es que se tienen)

# ahora, analizar porque
# ver las metricas son buenas 
# precision vs recall vs accuracy
accuracy = f1-score =? exactitud
precission = precisión
recall = sensibilidad, sensible a determinada clase
# reentrenar aumentando o disminuyendo los datos y ver si mejora o no 
https://www.themachinelearners.com/metricas-de-clasificacion/#Principales_Metricas_de_clasificacion
La métrica accuracy representa el porcentaje
  total de valores correctamente clasificados,
  tanto positivos como negativos.
La métrica de precisión es utilizada para
  poder saber qué porcentaje de valores que
  se han clasificado como positivos son
  realmente positivos.
La métrica de recall, también conocida
  como el ratio de verdaderos positivos,
  es utilizada para saber cuantos valores
  positivos son correctamente clasificados.
F1 Score Esta es una métrica muy utilizada
  en problemas en los que el conjunto de datos
  a analizar está desbalanceado.
  Esta métrica combina el precision y el
  recall, para obtener un valor mucho más
  objetivo.
  
https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

# ----------------------------
# red neuronal
# el chiste es diseñarla, no hay una guía
# componentes: funcion parametrica, pesos,
#   variable aleatoria,
#   y un bias
# la funcion parametrica es la funcion de
#   activacion
# la red neuronal es un regresor n-dimensional
# la r.n. se ajusta sola, para ello es la funcion
#   de costo, que es la distancia entre
#   la clase real y la prediccion
#   calcula la entropia

entropia la cantidad de bits que se mandan y que arribaron correctamente

en r.n. que tanto reconoce los datos de entrada y se autocorrige

autocorrige es que se modifican los pesos
gradiente estocastico


# bibliotecas
sklearn; keras (manejar mas datos); Torch(dificil porque es bastente personalizable); Tensorflow. Las dos ultimas requieren GPUs

y=G(X,W) -> x son datos, y puede ser una probabilidad de la etiqueta, W son los pesos, G es la funcion

# probar si la r.n. es mas eficiente, ver sus metricas



from sklearn import __version__ as skv
print(skv)

from sklearn.datasets import load_iris
data=load_iris()

x=data.data
y=data.target

from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test = (
  train_test_split(x,y,test_size=0.80,shuffle=True))

from sklearn.naive_bayes import GaussianNB

model0=GaussianNB()
model0.fit(x_train,y_train)

y_preds=model0.predict(x_test)

# ...

# queremos ver que esta pasando y que pasa si se mueven los hiperparametros


from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

model1=MLPClassifier()
model1.fit(x_train,y_train)
y_preds_m1=model1.predict(x_test)
print(classification_report(y_preds_m1,y_test))

model2=MLPClassifier(max_iter=10000)
model2.fit(x_train,y_train)
y_preds_m2=model2.predict(x_test)
print(classification_report(y_preds_m2,y_test))

model3=MLPClassifier(solver='sgd') # stocastic gradiant descent
model3.fit(x_train,y_train)
y_preds_m3=model3.predict(x_test)
print(classification_report(y_preds_m3,y_test))

# cambiar funcion de activacion de f(w*x+b) ===> tanh(w*x+b)
model4=MLPClassifier(activation='tanh')
model4.fit(x_train,y_train)
y_preds_m4=model4.predict(x_test)
print(classification_report(y_preds_m4,y_test))

# crear una r.n. profunda
# cuantas capas? cuantos elementos a cada capa oculta?
model5=MLPClassifier(hidden_layer_sizes=(5,2))
model5.fit(x_train,y_train)
y_preds_m5=model5.predict(x_test)
print(classification_report(y_preds_m5,y_test))

model5=MLPClassifier(hidden_layer_sizes=(3,3,3,2))
model5.fit(x_train,y_train)
y_preds_m5=model5.predict(x_test)
print(classification_report(y_preds_m5,y_test))

# tips:
# entre mas capas es peor entonces probablemente no necesita tantas capas por la cantidad de datos
# no hay reglas para modificar los hiperparámetros

model6=MLPClassifier(hidden_layer_sizes=(3,6,6,3), # por el numero de datos
    max_iter=1000,  # 100000,
    activation='logistic',
    solver='lbfgs', # optimizador para matriz acorde al a regresion lineal
    batch_size=2) # dos datos por iteracion
model6.fit(x_train,y_train)
y_preds_m6=model6.predict(x_test)
print(classification_report(y_preds_m6,y_test))

# imprimir la matriz de confusion, a parte del reporte de clasificacion
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
fig=ConfusionMatrixDisplay.from_estimator(model6,x_test,y_test)
plt.show()
plt.close()

# ver como se ajustan
# obtener la curva de perdida por las iteraciones
# Graficas de complejidad
plt.plot(model2.loss_curve_)
plt.title('Perdida a lo largo de las epocas(iteraciones)')
plt.xlabel('Epochs/Iterations')
plt.ylabel('Cost')
plt.show()
plt.close()
# costo: entropia, entropia alta no detecta nada
# se puede ver que tardo bastante en aprender

# entrenar mediante un grid, combinaciones de hiperparametros
grid_items={
  'max_iter':[1000,5000,10000],
  'activation':['tanh','relu'],
  'solver':['sgd','adam']
}
grid=GridSearchCV(
    model1, grid_items,
    n_jobs=-1, # todos los nucleos en paralelo
    cv=5 # validacion cruzada, segmentar los datos, divide el conjunto de datos de 5 en 5
  )
grid.fit(x_train,y_train)
print(grid.best_params_)
grid_pred=grid.predict(x_test)
print(classification_report(grid_pred,y_test))
