# https://posit.co/blog/three-ways-to-program-in-python-with-rstudio/
# reticulate::py_install('pandas')
# reticulate::py_install('matplotlib')
# https://support.posit.co/hc/en-us/articles/1500007929061-Using-Python-with-the-RStudio-IDE


# RStudio
# library(reticulate)
# py_install("pandas")
# py_install("matplotlib")
# py_install("scikit-learn")
# py_install("seaborn")
# py_install("imbalanced_learn") # metricas clasificador, genera datos faltantes
# reticulate::virtualenv_install(packages = c("numpy==1.8.0"))
# system2(reticulate::py_exe(), c("-m", "pip", "uninstall -y", 'scikit-learn'))


import os

print('Directorio inicial:', os.getcwd())
try:
    os.chdir('d:/')
    os.chdir('Proyectos/RStudioProyectos/GitHub')
except FileNotFoundError as e:
    print('No estamos en Windows')

try:
    os.chdir('Downloads')
    os.chdir('GitHub')
except FileNotFoundError as e:
    print('No estamos en Mac')

os.chdir('MemoriaSeminario2024')
os.chdir('datasets')

print('Directorio actual:', os.getcwd())
if not os.getcwd().endswith('datasets'):
    raise Exception('No se pudo encontrar el directorio de trabajo')

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
if pd.__version__ in ['2.2.2','2.2.3']:
    print('Pandas OK')
else:
    raise Exception(f'Versión inesperada de Pandas: {pd.__version__}.')
if np.__version__ in ['2.1.1','2.1.3']:
    print('Numpy OK')
else:
    raise Exception(f'Versión inesperada de Numpy: {np.__version__}.')
if mpl_version == '3.9.2':
    print('Matplotlib OK')
else:
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
print(df.describe())
# Existe el IdAnalista con valor a cero.

print('Visualización de estadísticas descriptivas de la longitud de NombreAbsolutoCorto:')
print(df['NombreAbsolutoCorto'].apply(lambda s: len(s)).describe())

print('Visualización de estadísticas descriptivas de la longitud de NombreAbsolutoLargo:')
print(df['NombreAbsolutoLargo'].apply(lambda s: len(s)).describe())


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

# 4.7. Agrupación de las variables por tema
# Cada tema tiene una o más variables para distintos horizontes de expectativa.
df_variables=df[['IdVariable','Variable']].drop_duplicates(keep='first')
df_variables=df_variables.sort_values(['Variable'])
df_variables['PrimerasLetras']=df_variables['Variable'].apply(lambda s: s[:7])
df_variables['DosPalabras']=df_variables['Variable'].apply(lambda s: ' '.join(s.split(' ')[:2]))
df_variables['Tema']=''
print(df_variables.head())

def imprime_array(s, n=-1, width=-1):
    """Imprime hasta pd.options.display.width caracteres por renglón."""
    for v in (s if n < 0 else s[:n]):
        max=pd.options.display.width if width == -1 else width
        valor=v[:max] if len(v) <= max else v[:(max - 3)] + '...'
        print('-> ' + valor)

def imprime_siguentes_variables(df_variables, n=-1, width=-1):
  primeras_letras=df_variables.query('Tema==""').head(1)['PrimerasLetras'].values[0]
  print(f'Variables con prefijo "{primeras_letras}":')
  imprime_array(df_variables.loc[
            (df_variables['PrimerasLetras'] == primeras_letras) &
            (df_variables['Tema'] == ''),
        'Variable'].sort_values().values, n, width)

def asigna_tema(df_variables, tema):
  primeras_letras=df_variables.query('Tema==""').head(1)['PrimerasLetras'].values[0]
  df_variables.loc[df_variables['PrimerasLetras'] == primeras_letras, 'Tema'] = tema

def imprime_temas():
    #df_variables['Tema'].drop_duplicates().values
    imprime_array(
        df_variables['Tema'].drop_duplicates(keep='first').sort_values().values)

imprime_temas()

# Observando la salida, se decidirá el tema de cada grupo de variables.

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

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Tasa nacional de desocupación; al cierre; anual',
        ('Tasa nacional de desocupación al cierre'))
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Tasa nacional de desocupación promedio; anual',
        ('Tasa nacional de desocupación promedio del '))
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Valor del tipo de cambio al cierre del año en curso',
        ('Valor del tipo de cambio al cierre del año en curso'))
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Valor del tipo de cambio promedio; durante el mes',
        ('Valor del tipo de cambio promedio durante el mes'))
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Variación desestacionalizada del PIB; trimestral',
        ('Variación desestacionalizada del PIB'))
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Variación en el número de trabajadores asegurados',
        ('Variación en el número de trabajadores asegurados'))
imprime_temas()

imprime_siguentes_variables(df_variables,5)
pone_tema_por_prefijo_variable(df_variables,
    'Variación porcentual anual del PIB de Estados Unidos; anual',
        ('Variación porcentual anual del PIB de Estados Unidos'))
imprime_temas()

imprime_siguentes_variables(df_variables, width=200)
pone_tema_por_prefijo_variable(df_variables,
    'Variación porcentual anual del PIB, probabilidad en el rango; anual',
        ('Variación porcentual anual del PIB en '))
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Variación porcentual anual del PIB para los próximos 10 años; a largo plazo',
        ('Variación porcentual anual del PIB para los próximos 10 años'))
imprime_temas()

imprime_siguentes_variables(df_variables, width=150)
pone_tema_por_prefijo_variable(df_variables,
    'Variación porcentual anual del PIB; anual',
        ('Variación porcentual anual del PIB, año anterior al correspondiente del levantamiento de la Encuesta (año t-1)',
         'Variación porcentual anual del PIB, año en curso (año t)',
         'Variación porcentual anual del PIB, siguiente año (año t+1)',
         'Variación porcentual anual del PIB, dentro de dos años (año t+2)',
         'Variación porcentual anual del PIB, dentro de tres años (año t+3)'
         ))
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Variación porcentual anual del PIB; trimstral',
        ('Variación porcentual anual del PIB, '))
imprime_temas()

imprime_siguentes_variables(df_variables)
print(df_variables.loc[df_variables['IdVariable'].str.startswith('coyun')])
print(df_variables.loc[df_variables['Variable'].str.startswith('cemp')])
pone_tema_por_prefijo_variable(df_variables,
    'Coyuntura empleo (?); bueno, malo, no seguro',
        ('cemp'))
imprime_temas()

imprime_siguentes_variables(df_variables)
print(df_variables.loc[df_variables['IdVariable'].str.startswith('clima')])
print(df_variables.loc[df_variables['Variable'].str.startswith('cneg')])
pone_tema_por_prefijo_variable(df_variables,
    'Cambio climático (?): empeorará, mejorará, permanecerá igual',
        ('cneg'))
imprime_temas()

imprime_siguentes_variables(df_variables)
print(df_variables.loc[df_variables['IdVariable'].str.startswith('ecopai')])
print(df_variables.loc[df_variables['Variable'].str.startswith('ep')])
pone_tema_por_prefijo_variable(df_variables,
    'Economía del país (?): no, sí',
        ('ep'))
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación general, probabilidad en el rango en 12 meses; mensual',
        ('inflacióngeneral_prob12m'))
imprime_temas()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación subyacente, probabilidad en el rango en 12 meses; mensual',
        ('inflaciónsubyacente_prob12m'))
imprime_temas()

imprime_siguentes_variables(df_variables)
imprime_array(df_variables.loc[df_variables['Variable'].str.startswith('limcrec')]['Variable'])
pone_tema_por_prefijo_variable(df_variables,
    'Límite de crecimiento; anual',
        ('limcrec'))
imprime_temas()

if df_variables[df_variables['Tema']==''].shape[0] != 0:
    print('Existen variables que no se les ha asignado tema:')
    imprime_siguentes_variables(df_variables)
    raise Exception('Existen variables que no se les ha asignado tema')

imprime_temas()
if df_variables[['Tema']].drop_duplicates(keep='first').shape[0] != 35:
    raise Exception('Cambió el número de temas.')

# Cuántas variables tiene cada tema
print(df_variables.groupby(['Tema'])['Variable'].count())
print(df_variables.groupby(['Tema'])['Variable'].count().sort_values())

# Quita las variables temporales que se usaron para el tema.
df_variables = df_variables.drop(['PrimerasLetras', 'DosPalabras'], axis = 1)

# Asigna el tema al data set.
df = df.merge(df_variables, how='left', on=['IdVariable','Variable'])
df = df.reindex(columns=['Fecha', 'Año', 'Mes', 'Tema', 'IdVariable', 'Variable', 'IdAnalista', 'Expectativa'])
if df.loc[df['Tema'] == ''].shape[0] > 0:
    raise Exception('No todos los renglones quedaron con tema')


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



xxxxxxxxxxxxxx




####





for column in df:
    plt.figure()
    df.boxplot([column])
df.boxplot(['Expectativa'])
df[['IdAnalista']].drop_duplicates().boxplot(['IdAnalista'])
plt.show()
plt.close()
df[['IdAnalista']].drop_duplicates().describe()
type()
xxx
# boxplot y violin
# metodo 1
import seaborn as sns
df_plot=df[['IdAnalista']].drop_duplicates()
sns.violinplot(data=df_plot, inner=None, color='white', linewidth=1)
sns.boxplot(data=df_plot, width=0.3, color='orange')
# metodo 2
#plt.boxplot(df[['IdAnalista']].drop_duplicates().values)
df[['IdAnalista']].drop_duplicates().boxplot(['IdAnalista'])
plt.violinplot(df[['IdAnalista']].drop_duplicates().values)
# muestra
plt.show()
plt.close()



bonost	Nivel de la tasa de interés del cete a 28 días
bonost1	Nivel de la tasa de interés del cete a 28 días
bonost2	Nivel de la tasa de interés del cete a 28 días
bonost3	Nivel de la tasa de interés del cete a 28 días
cetest	Nivel de la tasa de interés del cete a 28 días
cetest1	Nivel de la tasa de interés del cete a 28 días
cetest2	Nivel de la tasa de interés del cete a 28 días
cetest3	Nivel de la tasa de interés del cete a 28 días
fondeot	Nivel de la tasa de interés del cete a 28 días
fondeotmas1	Nivel de la tasa de interés del cete a 28 días
fondeotmas2	Nivel de la tasa de interés del cete a 28 días
fondeotmas3	Nivel de la tasa de interés del cete a 28 días
fondeotmas4	Nivel de la tasa de interés del cete a 28 días
fondeotmas5	Nivel de la tasa de interés del cete a 28 días
fondeotmas6	Nivel de la tasa de interés del cete a 28 días
fondeotmas7	Nivel de la tasa de interés del cete a 28 días
fondeotmas8	Nivel de la tasa de interés del cete a 28 días
fondeotmas9	Nivel de la tasa de interés del cete a 28 días

Inflación general al cierre



xxx


df_variables.query('Tema == "Nivel de la tasa de interés del cete a 28 días; al cierre; anual"')
esto esta maaal se ve en los idvariable

x['PrimerasLetrasIdVariable']=x['IdVariable'].apply(lambda s: s[:10])
....

xxx



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
