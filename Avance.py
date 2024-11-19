# https://posit.co/blog/three-ways-to-program-in-python-with-rstudio/
# reticulate::py_install('pandas')
# reticulate::py_install('matplotlib')
# https://support.posit.co/hc/en-us/articles/1500007929061-Using-Python-with-the-RStudio-IDE

import os
print(os.getcwd())
#os.chdir('Downloads/MemoriaSeminario2024')

# ------------------------------------------------------------------------------
# 1. Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)

# Verificar las versiones de las bibliotecas, para mejorar la reproducibilidad.
if pd.__version__ != '2.2.2': raise Exception(f'Versión inesperada de Pandas: {pd.__version__}.')
if np.__version__ != '2.1.1': raise Exception(f'Versión inesperada de Numpy: {np.__version__}.')
from matplotlib import __version__ as mpl_version
if mpl_version != '3.9.2': raise Exception(f'Versión inesperada de Matplotlib: {mpl_version}.')

# ------------------------------------------------------------------------------
# 2. Adquisición de datos

# Path de datasets locales
Microdatos_2020_01_csv_path = 'Microdatos_2020_01.csv'
Microdatos_1999_01_csv_path = 'Microdatos_1999_01.csv'

# Crea DataFrame con toda la historia
df_exp1 = pd.read_csv(Microdatos_2020_01_csv_path, encoding='latin-1')
df_exp2 = pd.read_csv(Microdatos_1999_01_csv_path, encoding='latin-1')
df = pd.concat([df_exp1, df_exp2], ignore_index=True)  # ignore_index porque no son relevantes


# ------------------------------------------------------------------------------
# 3. Inspección inicial

# Obtener información general sobre los datos, tal como la cantidad de filas y
# columnas, los valores de los datos, los tipos de datos y los valores faltantes
# en el conjunto de datos.

# Dimensiones de los datos
rows, cols = df.shape
if cols != 7: raise Exception(f"Se detectaron menos columnas que antes.")
if rows <= 1500000: raise Exception(f"Se detectaron menos registros que antes.")
if rows > 1600000: raise Exception(f"Se detectaron más registros que antes.")
print(f'Hay 7 columnas y poco mas de un millón y medio de registros ({rows})')

# Visualización de los primeros y últimos 3 renglones, y 10 aleatorios.
print(df.head(3))
print(df.tail(3))
print(df.sample(10))

print(df.columns)
# Las columnas son: 'FechaEncuesta', 'NombreAbsolutoCorto',
# 'NombreRelativoCorto', 'NombreAbsolutoLargo', 'NombreRelativoLargo',
# 'IdAnalista', 'Dato'

# Tipos de las columnas
print(df.dtypes)
# Sólo 2 se detectan como numéricas


# ------------------------------------------------------------------------------
# 4. Preparación de los datos

# 4.1. Reducción de columnas
# Se eliminan las columnas con nombre 'Absolutas',
# porque son columnas derivadas de la columna FechaEncuesta y las columnas
# con nombre 'Relativo' y, por tanto, no agregan valor para el análisis.
df = df.drop(['NombreAbsolutoCorto', 'NombreAbsolutoLargo'], axis = 1)
df.head()

# 4.2. Busca duplicados sin contar la columna Dato
# Sólo debería haber un dato de expectativa para cada fecha, variable, analista.
s_duplicados=df[['FechaEncuesta', 'NombreRelativoCorto', 'NombreRelativoLargo', 'IdAnalista'
                 ]].duplicated(keep=False)
print('Existen:', s_duplicados[s_duplicados==True].size, 'registros duplicados, con la(s) variable(s):\n',
      df[s_duplicados][['NombreRelativoCorto', 'NombreRelativoLargo']].drop_duplicates(keep='first'))
cuenta_original=df.shape[0]
df=df.drop_duplicates(subset=['FechaEncuesta', 'NombreRelativoCorto', 'NombreRelativoLargo', 'IdAnalista'],
                       keep=False)
cuenta_sin_dups=df.shape[0]
print(f'Antes {cuenta_original} registros, ahora {cuenta_sin_dups} es decir {(cuenta_original - cuenta_sin_dups) / cuenta_original * 100:.1f} % menos')

# 4.3. Busca incongruencias en variables
# Un NombreCorto debe tener un solo NombreLargo y viceversa
# (deben estar pareados los nombres relativos;
# es decir, los valores deben tener una relación biunívoca.)
#
# No se busca incongruencias de variable por fecha, sino en el DataFrame completo,
# por lo que pueden quedar eliminados registros con variables
# que no necesariamente estén duplicadas en una fecha.
def quita_duplicados(df_orig, df_busqueda, str_columna):
  """Elimina de df_orig los registros que tengan en la str_columna
  los valores que estén repetidos en df_busqueda.
  Regresa: el DataFrame sin los registros encontrados."""
  df_columna = df_busqueda[[str_columna]]
  ser_duplicados_booleans = df_columna.duplicated(keep=False)  # todos los valores duplicados
  ser_valores_duplicados = df_columna.loc[ser_duplicados_booleans].drop_duplicates(keep='first')[str_columna]  # solo los duplicados
  df_resultado = df_orig.query(str_columna + ' not in @ser_valores_duplicados')
  cuenta_eliminados = df_orig.query(str_columna + ' in @ser_valores_duplicados').shape[0]
  cuenta_original = df.shape[0]
  pct_eliminado = (1 - (cuenta_original - cuenta_eliminados) / cuenta_original) * 100
  print(f'Se eliminaron {cuenta_eliminados} registros ({pct_eliminado:.1f}%) con {str_columna} duplicados: {ser_valores_duplicados.values}')
  return df_resultado
# NombresRelativos
df_vars_nombres_relativos = df[['NombreRelativoCorto', 'NombreRelativoLargo']].drop_duplicates(keep='first')
df=quita_duplicados(df, df_vars_nombres_relativos, 'NombreRelativoCorto')
df=quita_duplicados(df, df_vars_nombres_relativos, 'NombreRelativoLargo')


xxx aqui voy. revisar si esta abajo todo el codigo del jupyther Avance.jnpy

# 4.4. Valores faltantes
if df.isnull().sum().sum() > 0: raise Exception('Hay valores faltantes y no se trataron')
else: print('No existen valores faltantes')
# No existen valores faltantes

# 4.5. Conversión de tipo de datos
print(f'Antes:\n{df.dtypes}')
# Convierte la FechaEncuesta a datetime
df['FechaEncuesta'] = pd.to_datetime(df['FechaEncuesta'], errors='raise')
print(f'Después:\n{df.dtypes}')
# Observando los valores únicos por columna, no parece haber variables categóricas, sino sólo contínuas
df.nunique()

# 4.6. Agregar columnas calculadas
df['AñoEncuesta'] = df['FechaEncuesta'].dt.year   # Columna con el año
df['MesEncuesta'] = df['FechaEncuesta'].dt.month  # Columna con el número de mes
print(df.dtypes)

# 4.7. Simplificar nombres columnas
print(f'Antes:\n{df.columns}')
df=df.rename(columns={
  'FechaEncuesta'      :'Fecha',
  'NombreRelativoCorto':'IdVariable',
  'NombreRelativoLargo':'Variable',
  'IdAnalista'         :'IdAnalista',
  'Dato'               :'Expectativa',
  'AñoEncuesta'        :'Año',
  'MesEncuesta'        :'Mes'
})
print(f'Después:\n{df.columns}')
print(df.head())

# 4.8. Orden
# Renglones: ordenar por fecha, variable, analista.
print('Antes:\n',df['Año'].unique())
df=df.sort_values(by=['Año','Mes', 'Variable', 'IdAnalista'])
print('Después:\n',df['Año'].unique())
# Columnas
print('Antes:')
print(df.columns)
print(df.head())
df = df.reindex(columns=['Fecha','Año', 'Mes','IdVariable','Variable','IdAnalista','Expectativa'])
print('\nDespués:')
print(df.columns)
print(df.head())

# 4.9 Pasar las variables a columnas

# En el dataset real
idVariable_unicas=df['IdVariable'].unique()
df_subset=df.query("IdVariable in @idVariable_unicas")
df_varscols=df_subset.pivot(
  index=['Año', 'Mes', 'Fecha','IdAnalista'],
  columns=['IdVariable', 'Variable'],
  values='Expectativa')
df_varscols.head()


# **====== PENDIENTE:**
# 
# Convertir variables categóricas (si/no; mucho/poco/nada).
# 
# 5. Estadísticas descriptivas
# 6. Visualización
# 7. Análisis de variables
# univariate, bivariate, or multivariate
# 8. Análisis de series de tiempo
# When we analyze time series data, we can typically uncover patterns or trends that repeat over time and present a temporal seasonality. Key components of time series data include trends, seasonal variations, cyclical variations, and irregular variations or noise.}
# 

# # Estadísticas descriptivas

# ## Análisis de número de respuestas

# In[ ]:


respuestasPorAño = df.groupby(by=["Año"])["Expectativa"].count()
respuestasPorAño.name = 'Número de respuestas por año'
respuestasPorAño.index.name = 'Año de la Enuesta'
respuestasPorAño.to_frame().plot.bar(
  title='Número de respuestas por año de la Encuesta (2024 año en curso)',
  rot=70,
  figsize=(10, 5),
  color='darkblue')

analistasDistintosPorAño = df.groupby(by=["Año"])["IdAnalista"].unique().apply(len)
analistasDistintosPorAño.name = 'Número de analistas distintos'
analistasDistintosPorAño.index.name = 'Año de la Enuesta'
analistasDistintosPorAño.to_frame().plot.bar(
  title='Número de analistas distintos por año de la Encuesta (2024 año en curso)',
  rot=70,
  figsize=(10, 5),
  color ='darkred')

analistasDistintosPorAño = df.groupby(by=["Año"])["Variable"].unique().apply(len)
analistasDistintosPorAño.name = 'Número de preguntas distintas'
analistasDistintosPorAño.index.name = 'Año de la Enuesta'
analistasDistintosPorAño.to_frame().plot.bar(
  title='Número de preguntas por año de la Encuesta (2024 año en curso)',
  rot=70,
  figsize=(10, 5),
  color ='g')


# ---
# 
# **Por tanto, se concluye que el aumento de respuestas desde 2013 se podría explicar por el aumento de preguntas más que por el aumento de analistas.**
# 
# ---



# ------------------------------------------------------------------------------
# ## Análisis de la Expectativa de Inflación General Anual

# In[ ]:


inflacion_general_anual = df.query('IdVariable=="infgent"')

inflacion_general_anual = inflacion_general_anual[['Año','Expectativa']] # Crea dataframe con sólo estas dos columnas
display(inflacion_general_anual)

inflacion_general_anual.plot.scatter(
  x='Año', y='Expectativa',
  rot=70,
  figsize=(10, 5),
  color='purple', alpha=0.2)

# Se asume que la distribución es normal, por lo que hacemos una gráfica de caja
axes = inflacion_general_anual.boxplot(
  column='Expectativa', by='Año',
  ylabel='Porcentaje', xlabel='Año de la encuesta',
  rot=70,
  figsize=(10, 5),
  color='purple')
axes.set_title('Expectativa de Inflación General al cierre del año de la encuesta')


# In[ ]:





# # Correlaciones

# In[ ]:


# Calcula la correlación entre todas las variables y todos los analistas en todas las fechas.
df_corrs=df_varscols.corr()
print(f'Son {df_varscols.columns.size} variables.')
df_corrs.sample(4)


# In[ ]:


f = plt.figure(figsize=(10, 10))
plt.matshow(df_corrs, f)
plt.show()
print('Son demasiadas variables para una sola gráfica.')


# In[ ]:




