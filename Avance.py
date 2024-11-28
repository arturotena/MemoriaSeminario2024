# https://posit.co/blog/three-ways-to-program-in-python-with-rstudio/
# reticulate::py_install('pandas')
# reticulate::py_install('matplotlib')
# https://support.posit.co/hc/en-us/articles/1500007929061-Using-Python-with-the-RStudio-IDE


import random 
for i in range(5):

    # Any number can be used in place of '0'.
    random.seed(0)

    # Generated random number will be between 1 to 1000.
    print(random.randint(1, 1000))




import os

print(os.getcwd())
os.chdir('Downloads')
os.chdir('GitHub')
os.chdir('MemoriaSeminario2024')
os.chdir('datasets')
# if not os.getcwd().endswith('MemoriaSeminario2024'):
#  os.chdir('Downloads/MemoriaSeminario2024')
#  print(os.getcwd())
print(os.getcwd())

semilla=1 #'Una semilla fija para reproducibilidad


# ---------------------------------------------------------------------------
# 1. Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 8)
pd.set_option('display.width', 90)
#pd.set_option('display.max_colwidth', 200)

# Verificar las versiones de las bibliotecas, para mejorar la reproducibilidad.
from matplotlib import __version__ as mpl_version
print(f'Pandas: {pd.__version__}.')
print(f'Numpy: {np.__version__}.')
print(f'Matplotlib: {mpl_version}.')
if pd.__version__ not in ['2.2.2','2.2.3']:
    raise Exception(f'Versión inesperada de Pandas: {pd.__version__}.')
if np.__version__ not in ['2.1.1','2.1.3']:
    raise Exception(f'Versión inesperada de Numpy: {np.__version__}.')
if mpl_version != '3.9.2':
    raise Exception(f'Versión inesperada de Matplotlib: {mpl_version}.')


# ----------------------------------------------------------------------------
# 2. Adquisición de datos

# Path de datasets locales
Microdatos_2020_01_csv_path = 'Microdatos_2020_01.csv'
Microdatos_1999_01_csv_path = 'Microdatos_1999_01.csv'

# Crea DataFrame con toda la historia
df_exp1 = pd.read_csv(Microdatos_2020_01_csv_path, encoding='latin-1')
df_exp2 = pd.read_csv(Microdatos_1999_01_csv_path, encoding='latin-1')
df = pd.concat([df_exp1, df_exp2], ignore_index=True)  # ignore_index porque no son relevantes


# --------------------------------------------------------------------------
# 3. Inspección inicial

# Obtener información general sobre los datos, tal como la cantidad de filas y
# columnas, los valores de los datos, y los tipos de datos.

# Dimensiones de los datos
rows, cols = df.shape
print(f'Hay {cols} columnas y {rows:,} registros.')
if cols != 7: raise Exception(f"Se detectaron menos columnas que antes.")
if rows <= 1500000: raise Exception(f"Se detectaron menos registros que antes.")
if rows > 1600000: raise Exception(f"Se detectaron más registros que antes.")

print('Visualización de los primeros y últimos 3 renglones, y 5 aleatorios:')
print(df.head(3))
print(df.tail(3))
print(df.sample(5, random_state=semilla))

print('Visualización de las columnas:')
print(df.columns)
# Las columnas son: 'FechaEncuesta', 'NombreAbsolutoCorto',
# 'NombreRelativoCorto', 'NombreAbsolutoLargo', 'NombreRelativoLargo',
# 'IdAnalista', 'Dato'

print('Visualización de los tipos de dato de las columnas:')
print(df.dtypes)
# Sólo 2 se detectan como numéricas

print('Visualización de estadísticas descriptivas de las columnas numéricas:')
df.describe()
# Existe el IdAnalista con valor a cero.

print('Visualización de estadísticas descriptivas de la longitud de IdVariable:')
print(df['IdVariable'].apply(lambda s: len(s)).describe())

print('Visualización de estadísticas descriptivas de la longitud de Variable:')
print(df['Variable'].apply(lambda s: len(s)).describe())


# --------------------------------------------------------------------------
# 4. Preparación de los datos

# 4.1. Limpieza de los datos: búsqueda de duplicados.
s_duplicados=df.duplicated(keep=False)
if s_duplicados[s_duplicados==True].size > 0:
    raise Exception('Hay renglones duplicados y no se trataron')
else:
    print('No existen renglones duplicados')

# 4.2. Reducción de columnas
# Se eliminan las columnas con nombre 'Absolutas', porque son columnas derivadas
# de la columna FechaEncuesta y las columnas con nombre 'Relativo' y, por tanto,
# no agregan valor para el análisis.
df = df.drop(['NombreAbsolutoCorto', 'NombreAbsolutoLargo'], axis = 1)
print(df.dtypes)

# 4.3. Conversión de tipo de datos
print(f'Antes:\n{df.dtypes}')
# Convierte la FechaEncuesta a datetime
df['FechaEncuesta'] = pd.to_datetime(df['FechaEncuesta'], errors='raise')
print(f'Después:\n{df.dtypes}')
# Observando los valores únicos por columna, no parece haber variables categóricas, sino sólo contínuas
print(f'Valores únicos:\n{df.nunique()}')

# 4.4. Agregar columnas calculadas
df['Año'] = df['FechaEncuesta'].dt.year
df['Mes'] = df['FechaEncuesta'].dt.month # número del mes
print(df.dtypes)

# 4.5. Simplificar nombres columnas
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

# 4.5. Valores faltantes
if df.isnull().sum().sum() > 0:
    raise Exception('Hay valores faltantes y no se trataron')
else:
    print('No existen valores faltantes')

# 4.6. Limpieza de los datos: busca duplicados sin contar la columna Dato:
# sólo debería haber un dato de expectativa para cada fecha, variable, analista.
s_duplicados=df[['Fecha', 'IdVariable', 'Variable', 'IdAnalista']
               ].duplicated(keep=False)

print(f'Existen: {s_duplicados[s_duplicados==True].size:,}'
      f' registros duplicados, con la(s) variable(s):\n',
      df[s_duplicados][['IdVariable', 'Variable']
      ].drop_duplicates(keep='first'))
cuenta_original=df.shape[0]
df=df.drop_duplicates(subset=['Fecha', 'IdVariable',
                      'Variable', 'IdAnalista'], keep=False)
cuenta_sin_dups=df.shape[0]
print(f'Antes {cuenta_original:,} registros, ahora {cuenta_sin_dups:,}' +
      f' registros; es decir '
      f'{(cuenta_original-cuenta_sin_dups)/cuenta_original*100:.1f}% menos.')

# 4.7. Limpieza de los datos: Busca incongruencias en variables
# Un NombreCorto debe tener un solo NombreLargo y viceversa.
#
# Deben estar pareados los nombres relativos; es decir,
# los valores deben tener una relación biunívoca.
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
  print(f'Se eliminaron {cuenta_eliminados:,} registros ({pct_eliminado:.1f}%) con {str_columna} duplicados: {ser_valores_duplicados.values}')
  return df_resultado
# NombresRelativos
df_vars_nombres_relativos = df[['IdVariable', 'Variable']].drop_duplicates(keep='first')
df=quita_duplicados(df, df_vars_nombres_relativos, 'IdVariable')
df=quita_duplicados(df, df_vars_nombres_relativos, 'Variable')

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
idVariable_unicas=df['IdVariable'].unique()
df_subset=df.query("IdVariable in @idVariable_unicas")
df_variables_en_columnas=df_subset.pivot(
    index=['Año', 'Mes', 'Fecha','IdAnalista'],
    columns=['IdVariable'], #, 'Variable'],
    values='Expectativa')
print('DataFrame con variables en columnas:' +
      f'\n* Columnas:\n{df_variables_en_columnas.columns[:5].values} ... {df_variables_en_columnas.columns[-5:].values}' +
      f'\n* Índice {df_variables_en_columnas.index.names}:\n{df_variables_en_columnas.index[:5].values} ... {df_variables_en_columnas.index[-5:].values}')
df_variables_en_columnas.describe().T.sample(10, random_state=semilla)

# 4.7. Agrupación de variables
# Agrupar por tema las variables de los distintos horizontes.
df_variables=df[['IdVariable','Variable']].drop_duplicates(keep='first')
df_variables=df_variables.sort_values(['Variable'])
df_variables['PrimerasLetras']=df_variables['Variable'].apply(lambda s: s[:7])
df_variables['DosPalabras']=df_variables['Variable'].apply(lambda s: ' '.join(s.split(' ')[:2]))
df_variables['Tema']=''
print(df_variables.head())

def imprime_array(s, n=-1):
    """Imprime hasta pd.options.display.width caracteres por renglón."""
    for v in (s if n < 0 else s[:n]):
        max=pd.options.display.width
        valor=v[:max] if len(v) <= max else v[:(max - 3)] + '...'
        print('-> ' + valor)

def imprime_siguentes_variables(df_variables, n=-1):
  primeras_letras=df_variables.query('Tema==""').head(1)['PrimerasLetras'].values[0]
  print(f'Variables con prefijo "{primeras_letras}":')
  imprime_array(df_variables.loc[
            (df_variables['PrimerasLetras'] == primeras_letras) &
            (df_variables['Tema'] == ''),
        'Variable'].sort_values().values, n)

def asigna_tema(df_variables, tema):
  primeras_letras=df_variables.query('Tema==""').head(1)['PrimerasLetras'].values[0]
  df_variables.loc[df_variables['PrimerasLetras'] == primeras_letras, 'Tema'] = tema

def imprime_temas():
    #df_variables['Tema'].drop_duplicates().values
    imprime_array(
        df_variables['Tema'].drop_duplicates(keep='first').sort_values().values)

imprime_temas()

# Observando la salida, se decide el tema.

imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, 'Balance económico del sector público; al cierre del año; anual')
imprime_temas()

imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, 'Balanza Comercial; saldo anual al cierre del año; anual')
imprime_temas()

imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, 'Competencia y Crecimiento; nivel')
imprime_temas()

imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, 'Cuenta Corriente; saldo anual al cierre del año; anual')
imprime_temas()

imprime_siguentes_variables(df_variables)
# Son los diferentes tipos de inflación:
#     Inflación general al cierre; al cierre del año; anual
#     Inflación general para dentro de; ; mensual
#         Inflación general para el mes en curso
#         Inflación general para el siguiente mes
#     Inflación general para los próximos; a largo plazo
#     Inflación subyacente al cierre; al cierre del año; anual
#     Inflación subyacente para dentro de; ; mensual
#         Inflación subyacente para el mes en curso
#         Inflación subyacente para el siguiente mes
#     Inflacióngeneral_12m
#     Inflaciónsubyacente_12m

def pone_tema_por_prefijo_variable(df_variables, tema, prefijos:tuple):
    condicion = df_variables['Variable'].str.startswith(prefijos)
    df_variables.loc[condicion, ['Tema']] = tema

imprime_siguentes_variables(df_variables, n=61)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación general al cierre; al cierre del año; anual',
        ('Inflación general al cierre '))
imprime_temas()

imprime_siguentes_variables(df_variables, n=14)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación general para dentro de; ; mensual',
        ('Inflación general para dentro de ',
         'Inflación general para el mes en curso',
         'Inflación general para el siguiente mes'))
imprime_temas()

imprime_siguentes_variables(df_variables, n=3)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación general para los próximos; a largo plazo',
        ('Inflación general para los próximos'))
imprime_temas()

imprime_siguentes_variables(df_variables, n=61)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación subyacente al cierre; al cierre del año; anual',
        ('Inflación subyacente al cierre '))
imprime_temas()

imprime_siguentes_variables(df_variables, n=14)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación subyacente para dentro de; ; mensual',
        ('Inflación subyacente para dentro de ',
         'Inflación subyacente para el mes en curso',
         'Inflación subyacente para el siguiente mes'))
imprime_temas()

imprime_siguentes_variables(df_variables, n=3)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación subyacente para los próximos; a largo plazo',
        ('Inflación subyacente para los próximos'))
imprime_temas()

imprime_siguentes_variables(df_variables)
# Ver su IdVariable también.
print(df_variables.query('PrimerasLetras == "Inflaci" and Tema==""'))
# También se consultó: https://www.banxico.org.mx/SieInternet/consultarDirectori
# oInternetAction.do?sector=24&accion=consultarCuadro&idCuadro=CR155&locale=es
pone_tema_por_prefijo_variable(df_variables,
    'Inflación general para los próximos 12 meses',
        ('Inflacióngeneral_12m_'))
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación subyacente para los próximos 12 meses',
        ('Inflaciónsubyacente_12m_'))
imprime_temas()

imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, 'Intensidad Competencia; nivel')
imprime_temas()

imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, 'Inversión Extranjera Directa; monto al cierre; anual')
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Nivel de la tasa de fondeo interbancaria; al cierre; trimestral',
        ('Nivel de la tasa de fondeo interbancaria al cierre'))
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Nivel de la tasa de interés de los Bonos M a 10 años; al cierre; anual',
        ('Nivel de la tasa de interés de los Bonos M a 10 años al cierre'))
imprime_temas()

imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, 'Nivel de la tasa de interés del cete a 28 días; al cierre; anual')
imprime_temas()

imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, 'Obstáculos Enfrentan Empresarios')
imprime_temas()

imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, 'Probabilidad de reducción en el PIB trimestral; trimestral')
imprime_temas()

imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, 'Saldo de requerimientos financieros del sector público; al cierre; anual')
imprime_temas()

imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, 'Sectores Problemas Competencia')
imprime_temas()
xxx
imprime_siguentes_variables(df_variables)
asigna_tema(df_variables, )
pone_tema_por_prefijo_variable(df_variables,
    ,
        ())
imprime_temas()



xxx
# Ver si corresponden las primeras letras de IdVariable con Tema
xxx ver si el idvariable no corresponde entonces tal vez se eligio mal el tema
xxx o hay incongruencia en esas variables y habria que quitarlas
df_variables[['IdVariable','Tema']].sort_values(['Tema'], ascending=False)


df_variables.loc[df_variables['Tema']==''].shape
if df_variables.loc[df_variables['Tema']==''].shape[0] > 0:
    raise Exception('Aún hay variables sin tema.')

xxx



#### df=df.merge(df_variables, how='left', on=['IdVariable','Variable'])


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


# --------------------------------------------------------------------------
# Estadísticas descriptivas

# Análisis de número de respuestas
respuestasPorAño = df.groupby(by=["Año"])["Expectativa"].count()
respuestasPorAño.name = 'Número de respuestas por año'
respuestasPorAño.index.name = 'Año de la Enuesta'
respuestasPorAño.to_frame().plot.bar(
    title='Número de respuestas por año de la Encuesta (2024 año en curso)',
    rot=70,
    figsize=(10, 5),
    color='darkblue')
plt.show()
plt.close()
analistasDistintosPorAño = df.groupby(by=["Año"])["IdAnalista"].unique().apply(len)
analistasDistintosPorAño.name = 'Número de analistas distintos'
analistasDistintosPorAño.index.name = 'Año de la Enuesta'
analistasDistintosPorAño.to_frame().plot.bar(
    title='Número de analistas distintos por año de la Encuesta (2024 año en curso)',
    rot=70,
    figsize=(10, 5),
    color ='darkred')
plt.show()
plt.close()
analistasDistintosPorAño = df.groupby(by=["Año"])["Variable"].unique().apply(len)
analistasDistintosPorAño.name = 'Número de preguntas distintas'
analistasDistintosPorAño.index.name = 'Año de la Enuesta'
analistasDistintosPorAño.to_frame().plot.bar(
    title='Número de preguntas por año de la Encuesta (2024 año en curso)',
    rot=70,
    figsize=(10, 5),
    color ='g')
plt.show()
plt.close()
# **Por tanto, se concluye que el aumento de respuestas desde 2013 se podría explicar por el aumento de preguntas más que por el aumento de analistas.**

# Análisis de la Expectativa de Inflación General Anual
inflacion_general_anual=df.query('IdVariable=="infgent"')
inflacion_general_anual = inflacion_general_anual[['Año','Expectativa']] # Crea dataframe con sólo estas dos columnas
print(inflacion_general_anual)
x=inflacion_general_anual.plot.scatter(
    x='Año', y='Expectativa',
    rot=70,
    figsize=(10, 5),
    color='purple', alpha=0.2)
plt.show()
plt.close()
# Se asume que la distribución es normal, por lo que hacemos una gráfica de caja
axes = inflacion_general_anual.boxplot(
    column='Expectativa', by='Año',
    ylabel='Porcentaje', xlabel='Año de la encuesta',
    rot=70,
    figsize=(10, 5),
    color='purple')
axes.set_title('Expectativa de Inflación General al cierre del año de la encuesta')
plt.show()
plt.close()


# --------------------------------------------------------------------------
# Correlaciones

# Calcula la correlación entre todas las variables y todos los analistas en todas las fechas.
df_corrs=df_variables_en_columnas.corr()
print(f'Son {df_variables_en_columnas.columns.size} variables.')
df_corrs.sample(4, random_state=semilla)

f = plt.figure(figsize=(10, 10))
plt.matshow(df_corrs, f)
plt.show()
plt.close()
print('Son demasiadas variables para una sola gráfica.')
