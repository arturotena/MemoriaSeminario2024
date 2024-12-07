# R
# library(reticulate)
# py_install("pandas")
# py_install("matplotlib")
# py_install("scikit-learn")
# py_install("seaborn")
# py_install("imbalanced_learn") # metricas clasificador, genera datos faltantes
# reticulate::virtualenv_install(packages = c("numpy==1.8.0"))
# system2(reticulate::py_exe(), c("-m", "pip", "uninstall -y", 'scikit-learn'))
# scikit-image

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ####################################
# Algoritmos no supervizados

# Para datos sin etiqueta.
# Requieren gran cantidad de nueronas.
# Requieren GPUs.

from sklearn.cluster import KMeans
# por distancia obtiene los medios
# usa normalmente la:
# d=sqr(x^2+y^2) => distancia euclidiana

from skimage.io import imread
from skimage.filters import gaussian

import os
os.getcwd()
os.chdir('..')
img=imread('colores.png')
print(img[0:10])
print(img[0:2][0:2][0:2])
[m,n,c] = img.shape # c son los 3 canales de color
plt.imshow(img)
plt.axis('off')
plt.show()
plt.close()
print(img.shape)
print(np.max(img))


# un canal
# histograma
# plt.hist(img[:,0]) # ,bins='auto')
# plt.show()
# plt.close()

plt.imshow(img[:,:,0],cmap='gray')
plt.show()
plt.close()



# KMeans calcula la media y clasifica los datos segun estan con relacion de esa media


img_0=img/np.max(img) # x/255= ---> [0,1] minimax  # normalizo de 0 a 1
print(np.max(img_0))
plt.imshow(img_0) # no cambia la imagen porque la biblioteca toma el maximo sin importar si es 1 o 255
plt.show()
plt.close()
# _=plt.hist(img_0[:,0]) #,bins='auto')
# plt.show()
# plt.close()



img_1=(img-np.mean(img))/np.std(img) # z transform hace que los datos tiendan a ser gausianos, que se acercan a la media
print(np.max(img_1))
plt.imshow(img_1)
plt.show()
plt.close() # el blanco se hizo gris (0.5)
# _=plt.hist(img_1[:,0]) #,bins='auto')
# plt.show()
# plt.close()




img_g=gaussian(img,sigma=1)
print(np.max(img_g))
plt.imshow(img_g) 
plt.show()
plt.close()


img_g=gaussian(img,sigma=9)
print(np.max(img_g))
plt.imshow(img_g) 
plt.show()
plt.close()


# se pueden pasar sus datos a bidimensional o multidimensional y procesarlos con proceso de imagenes





....


all_pixels=img.reshape((-1,3)) # una sola columna
print(all_pixels.shape)
plt.plot(all_pixels)
plt.show()
plt.close()

plt.plot(all_pixels[:,0])
plt.show()
plt.close()

plt.plot(all_pixels[:,1])
plt.show()
plt.close()

plt.plot(all_pixels[:,2])
plt.show()
plt.close()

plt.hist(all_pixels[:,0])
plt.show()
plt.close()

plt.hist(all_pixels)
plt.show()
plt.close()


dominant_colors=2 # subclases
# tecnica para saber donde se concentra mas informacion para saber cuantas subclases buscar
km = KMeans(n_clusters=dominant_colors)
km.fit(all_pixels)

centers=km.cluster_centers_
print(centers)
# muestra los dos centros en tres canales de color:
# [[254.38470271 254.31619207 253.83367957]
#  [180.63177393 122.36393085  94.98897559]]
centers = np.array(centers,dtype='uint8')
print(centers)

i=1
plt.figure(0,figsize=(8,2))
color=[]
for each_col in centers:
    plt.subplot(1,....

...

metodo del codo
wcss
minimiza la varianza
para obtener el menos numero de grupos

problema con no supervisado es random el ini io cengroide

https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/


Outliers Detection: Anomaly detection is the identification of rare or unusual data points. By clustering data into different groups and observing clusters with significantly fewer data points, the Elbow Method can assist in identifying anomalies or outliers more effectively.
https://medium.com/@zalarushirajsinh07/the-elbow-method-finding-the-optimal-number-of-clusters-d297f5aeb189

