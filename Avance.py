# ---------------------------------------------------------------------------
# Para instalar Python en RStudio
#
# library(reticulate)
# reticulate::repl_python()
#
# Otra opción podría ser:
# library(reticulate)
# reticulate::install_python(version = "3.12.5")
#
# Ayuda: https://rstudio.github.io/reticulate/reference/install_python.html


# ---------------------------------------------------------------------------
# Verifica versión de Python para asegurar la reproducibilidad.
import sys
version_python = sys.version.split()[0]
print(f'Python: {version_python}.')
if version_python == '3.12.5':
    print('Python OK')
else:
    reporta_error(f'Versión inesperada de Python: {version_python}.')


# ---------------------------------------------------------------------------
# Funciones de apoyo.

def reporta_error(s):
    '''Imprime el error.'''
    separador = '*'*70 + '\n'
    print(separador*3, s, '\n', separador*3)

def print_df(df):
    '''
    Imprime el pd.DataFrame y lo copia al portapapeles
    para su reporte posterior.
    '''
    df.to_clipboard()
    print(df)


# ---------------------------------------------------------------------------
# Cambia el directorio local al directorio donde están los archivos CSV.

import os

print('Directorio inicial:', os.getcwd())
try:
    os.chdir('d:/')
    os.chdir('Proyectos/RStudioProyectos/GitHub')
    os.chdir('datasets')
except FileNotFoundError as e:
    # No estamos en Windows
    pass

try:
    os.chdir(os.path.expanduser("~"))
    os.chdir('Downloads')
    os.chdir('MemoriaSeminario2024')
    os.chdir('datasets')
except FileNotFoundError as e:
    # No estamos en Mac
    pass

print('Directorio actual:', os.getcwd())
if not os.getcwd().endswith('datasets'):
    reporta_error('No se pudo encontrar el directorio de los data sets')


# ---------------------------------------------------------------------------
# 1. Importar bibliotecas

# Para instalar:
# !pip install pandas==2.2.3
# !pip install numpy==2.2.0
# !pip install Matplotlib==3.9.2
# !pip install seaborn==0.13.2

try:
    import pandas as pd
except ModuleNotFoundError:
    reporta_error('No está instalada la biblioteca Pandas')

try:
    import numpy as np
except ModuleNotFoundError:
    reporta_error('No está instalada la biblioteca NumPy')

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    reporta_error('No está instalada la biblioteca MatplotLib')

try:
    import seaborn as sns
except ModuleNotFoundError:
    reporta_error('No está instalada la biblioteca Seaborn')

# Se verifica la versión de las bibliotecas para asegurar la reproducibilidad.
from matplotlib import __version__ as mpl_version
print(f'Pandas: {pd.__version__}.')
print(f'Numpy: {np.__version__}.')
print(f'Matplotlib: {mpl_version}.')
print(f'Seaborn: {sns.__version__}.')

if pd.__version__ != '2.2.3':
    reporta_error(f'Versión inesperada de Pandas: {pd.__version__}.')

if np.__version__ != '2.2.0':
    reporta_error(f'Versión inesperada de Numpy: {np.__version__}.')

if mpl_version != '3.9.2':
    reporta_error(f'Versión inesperada de Matplotlib: {mpl_version}.')
    
if sns.__version__ != '0.13.2':
    reporta_error(f'Versión inesperada de Seaborn: {sns.__version__}.')

# Se establece la semilla fija para asegurar reproducibilidad de los resultados.
np.random.seed(2)

# Se establece el número de columnas a visualizar, y otros valores relacionados.
pd.set_option('display.max_columns', 8)
# pd.set_option('display.width', 150)
# pd.set_option('display.max_colwidth', 200)


# ----------------------------------------------------------------------------
# 2. Adquisición de datos

# Path de datasets locales
Microdatos_2020_01_csv_path = 'Microdatos_2020_01.csv'
Microdatos_1999_01_csv_path = 'Microdatos_1999_01.csv'

# Crea DataFrame con toda la historia
df_exp1 = pd.read_csv(Microdatos_2020_01_csv_path, encoding='latin-1')
df_exp2 = pd.read_csv(Microdatos_1999_01_csv_path, encoding='latin-1')
df = pd.concat([df_exp1, df_exp2],
               ignore_index=True) # no toma en cuenta los índices al concatenar


# --------------------------------------------------------------------------
# 3. Inspección inicial

# Obtener información general sobre los datos, tal como la cantidad de filas y
# columnas, los valores de los datos, y los tipos de datos.

# Dimensiones de los datos
rows, cols = df.shape
print(f'Hay {cols} columnas y {rows:,} registros.')
if cols != 7: reporta_error(f"Se detectaron menos columnas que antes.")
if rows <= 1500000: reporta_error(f"Se detectaron menos registros que antes.")
if rows > 1600000: reporta_error(f"Se detectaron más registros que antes.")

print('Los primeros 3 renglones:')
print_df(df.head(3))
print('Los últimos 3 renglones:')
print_df(df.tail(3))
print('Aleatoriamente muestra 5 renglones:')
print_df(df.sample(5))

print('Visualización de las columnas:')
print(list(df.columns.values))
# Verifica que las columnas sean las esperadas.
if list(df.columns.values) !=  ['FechaEncuesta', 'NombreAbsolutoCorto',
                                'NombreRelativoCorto', 'NombreAbsolutoLargo',
                                'NombreRelativoLargo', 'IdAnalista', 'Dato']:
    reporta_error('Cambiaron las columnas')

print('Los tipos de dato de las columnas:')
print_df(df.dtypes)
if list(df.dtypes.values) != ['O', 'O', 'O', 'O', 'O', 'int64', 'float64']:
    reporta_error('Cambiaron tipos de las columnas')
# Se observa que sólo 2 columnas se detectan como numéricas.

print('Visualización de estadísticas descriptivas de las columnas numéricas:')
print_df(df.describe())
# Se observa que existe el IdAnalista con valor a cero.

print('Mínima y máxima longitud de NombreAbsolutoCorto:')
print_df(df['NombreAbsolutoCorto'].apply(lambda s: len(s)).agg(['min', 'max']))

print('Mínima y máxima longitud de NombreAbsolutoLargo:')
print_df(df['NombreAbsolutoLargo'].apply(lambda s: len(s)).agg(['min', 'max']))


# --------------------------------------------------------------------------
# 4. Preparación de los datos

# 4.1. Limpieza de los datos: búsqueda de duplicados.
s_duplicados=df.duplicated(keep=False)
if s_duplicados[s_duplicados==True].size > 0:
    reporta_error('Hay renglones duplicados y no se trataron')
else:
    print('No existen renglones duplicados')

# 4.2. Reducción de columnas
# Se eliminan las columnas con nombre 'Absolutas', porque son columnas derivadas
# de la columna FechaEncuesta y las columnas con nombre 'Relativo' y, por tanto,
# no agregan valor para el análisis.
df = df.drop(['NombreAbsolutoCorto', 'NombreAbsolutoLargo'], axis = 1)
print(df.dtypes)

# 4.3. Conversión de tipo de datos
print('Antes:')
print_df(df.dtypes)
# Convierte la FechaEncuesta a datetime
df['FechaEncuesta'] = pd.to_datetime(df['FechaEncuesta'], errors='raise')
print('Después:')
print_df(df.dtypes)
print('Valores únicos:')
print_df(df.nunique())

# 4.4. Agregar columnas calculadas
df['Año'] = df['FechaEncuesta'].dt.year
df['Mes'] = df['FechaEncuesta'].dt.month # número del mes
print_df(df.dtypes)

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
print_df(df.sample(4))

# 4.6. Valores faltantes
s_faltantes_por_columna = df.isnull().sum()
print_df(s_faltantes_por_columna)
if s_faltantes_por_columna.sum() > 0:
    reporta_error('Hay valores faltantes y no se trataron')
else:
    print('No existen valores faltantes')
# Por conocimiento previo de este data set, se sabe que cuando un analista
# no contesta una pregunta de la encuesta entonces no se incluye en el dataset.
# Esto explica la inexistencia de valores faltantes, aún cuando se tiene
# conocimiento previo que no todos los analistas contestan todas las preguntas.

# 4.7. Limpieza de los datos: busca duplicados sin contar la columna Dato:
# sólo debería haber un dato de expectativa para cada fecha, variable, analista.
s_duplicados=df[['Fecha', 'IdVariable', 'Variable', 'IdAnalista']] \
               .duplicated(keep=False)

print('Ejemplo de datos duplicados:')
print_df(df[s_duplicados].sort_values(
            by=['Año','Mes', 'Variable', 'IdAnalista']
        ).head(6))

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

# 4.8. Limpieza de los datos: Busca incongruencias en las variables.
# Un valor en la columna IdVariable debe tener una sola Variable y viceversa;
# es decir estas columnas deben tener una correspondencia biunívoca.
#
# No se busca incongruencias de variable por fecha, sino en
# el DataFrame, por lo que si se encuentra una incongruencia en
# una variable en una fecha entonces se procede a eliminar dicha variable
# en todo el DataFrame.

def quita_duplicados(df_orig, df_busqueda, str_columna):
    """Elimina de df_orig los registros que tengan en la str_columna
    los valores que estén repetidos en df_busqueda.
    Regresa: el DataFrame sin los registros encontrados."""
    df_columna = df_busqueda[[str_columna]]
    s_duplicados_booleans = df_columna.duplicated(keep=False)  # todos los valores duplicados
    s_valores_duplicados = df_columna.loc[s_duplicados_booleans].drop_duplicates(keep='first')[str_columna]
    # s_valores_duplicados tiene solo los duplicados
    df_resultado = df_orig.query(str_columna + ' not in @s_valores_duplicados')
    cuenta_eliminados = df_orig.query(str_columna + ' in @s_valores_duplicados').shape[0]
    cuenta_original = df.shape[0]
    pct_eliminado = (1 - (cuenta_original - cuenta_eliminados) / cuenta_original) * 100
    print(f'Se eliminaron {cuenta_eliminados:,} registros ({pct_eliminado:.1f}%)'
          f' con {str_columna} duplicados:\n{s_valores_duplicados.values}')
    return df_resultado

df_vars = df[['IdVariable', 'Variable']].drop_duplicates(keep='first')
df=quita_duplicados(df, df_vars, 'IdVariable')
df=quita_duplicados(df, df_vars, 'Variable')

# 4.9. Orden
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

# 4.10. Agrupación de las variables por tema
# Cada tema tiene una o más variables para distintos horizontes de expectativa.

df_variables=(df[['IdVariable','Variable']]
                  .drop_duplicates(keep='first')
                  .reindex(columns=['IdVariable','Variable'])
                  .sort_values(['Variable']))

df_variables['PrimerasLetras']=df_variables['Variable'].apply(lambda s: s[:7])
# Se probó con distinto número de letras iniciales hasta encontrar un número
# que agrupara suficientemente a las variables.
print(df_variables['PrimerasLetras'].drop_duplicates().shape[0])

df_variables['DosPalabras']=df_variables['Variable'].apply(
    lambda s: ' '.join(s.split(' ')[:2]))
# De forma análoga, se probó con 1, 2, 3 y 4 palabras, y se encontró que 2
# era el número que permitía agrupar mejor las variables.
print(df_variables['DosPalabras'].drop_duplicates().shape[0])

df_variables['Tema']=''
df_variables['Cifra']=''
df_variables['Horizonte']=''
df_variables['Unidad']=''
print_df(df_variables.head())

def imprime_array(s, n=-1, width=-1):
    """Imprime hasta pd.options.display.width caracteres por renglón."""
    c = 0
    for v in (s if n < 0 else s[:n]):
        max=pd.options.display.width if width == -1 else width
        valor=v[:max] if len(v) <= max else v[:(max - 3)] + '...'
        c = c + 1
        print(f'{c} -> {valor}')

def imprime_siguentes_variables(df_variables, n=-1, width=-1):
  primeras_letras=df_variables.query('Tema==""').head(1)['PrimerasLetras'].values[0]
  print(f'Variables con prefijo "{primeras_letras}":')
  imprime_array(df_variables.loc[
            (df_variables['PrimerasLetras'] == primeras_letras) &
            (df_variables['Tema'] == ''),
        'Variable'].sort_values().values, n, width)

def imprime_temas():
    imprime_array(
        df_variables['Tema'].drop_duplicates(keep='first').sort_values().values)
        
def imprime_cifras():
    imprime_array(
        df_variables['Cifra'].drop_duplicates(keep='first').sort_values().values)

def imprime_horizontes():
    imprime_array(
        df_variables['Horizonte'].drop_duplicates(keep='first').sort_values().values)

def imprime_unidades():
    imprime_array(
        df_variables['Unidad'].drop_duplicates(keep='first').sort_values().values)

def pone_tema_por_prefijo_variable(df_variables, detalles:str, prefijos:tuple):
    '''
    A partir de los detalles asigna las columnas Tema, Cifra, Horizonte, y
    Unidad a los renglones del DataFrame que no tengan Tema asignado y cuya
    columna Variable comience con uno de los prefijos.
    Se espera que detalles tenga el formato: 'tema: unidad; cifra; horizonte'.
    '''
    lst_temas=[un_tema.strip() for un_tema in detalles.split(';')]
    if len(lst_temas) < 3: lst_temas.append('')
    tmp=lst_temas[0].split(':')
    tema=tmp[0]
    unidad=tmp[1]
    cifra=lst_temas[1]
    horizonte=lst_temas[2]
    condicion = (df_variables['Variable'].str.startswith(prefijos)) \
                & (df_variables['Tema'] == '')
    df_variables.loc[condicion, ['Tema']] = tema
    df_variables.loc[condicion, ['Cifra']] = cifra
    df_variables.loc[condicion, ['Horizonte']] = horizonte
    df_variables.loc[condicion, ['Unidad']] = unidad
    cuenta_variables = df_variables.loc[condicion, ['Variable']] \
                            .drop_duplicates(keep='first').shape[0]
    print(f'Tema asignado para {cuenta_variables} variables distintas')

imprime_temas()
imprime_cifras()
imprime_horizontes()
imprime_unidades()
# En este momento no hay ninguno.

# Observando la salida, se decidió el tema de cada grupo de variables.

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Balance económico del sector público: _desconocido; al cierre del periodo; anual',
        ('Balance económico del sector público'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Balanza Comercial: _desconocido; al cierre del periodo; anual',
        ('Balanza'))

print('Revisando:')
imprime_temas()
imprime_cifras()
imprime_horizontes()
imprime_unidades()

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Competencia y Crecimiento: nivel; _desconocido; _desconocido',
        ('Compete'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Cuenta Corriente: _desconocido; al cierre del periodo; anual',
        ('Cuenta '))

imprime_siguentes_variables(df_variables)
# Son los diferentes tipos de inflación.

imprime_siguentes_variables(df_variables, n=61)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación general: porcentaje; al cierre del periodo; anual',
        ('Inflación general al cierre '))

imprime_siguentes_variables(df_variables, n=14)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación general: porcentaje; al cierre del periodo; mensual',
        ('Inflación general para dentro de ',
         'Inflación general para el mes en curso',
         'Inflación general para el siguiente mes'))

imprime_siguentes_variables(df_variables, n=3)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación general: porcentaje; al cierre del periodo; a largo plazo',
        ('Inflación general para los próximos'))

imprime_siguentes_variables(df_variables, n=61)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación subyacente: porcentaje de probabilidad en rango; al cierre del periodo; anual',
        ('Inflación subyacente al cierre del año en curso (año t), probabilidad de que se encuentre en rango',
         'Inflación subyacente al cierre del siguiente año (año t+1), probabilidad de que se encuentre en rango',
         'Inflación subyacente al cierre dentro de dos años (año t+2), probabilidad de que se encuentre en rango',
         'Inflación subyacente al cierre dentro de tres años (año t+3), probabilidad de que se encuentre en rango'))

imprime_siguentes_variables(df_variables, n=5)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación subyacente: porcentaje; al cierre del periodo; anual',
        ('Inflación subyacente al cierre del año en curso (año t)',
         'Inflación subyacente al cierre del siguiente año (año t+1)',
         'Inflación subyacente al cierre dentro de dos años (año t+2)',
         'Inflación subyacente al cierre dentro de tres años (año t+3)'))

imprime_siguentes_variables(df_variables, n=14)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación subyacente: porcentaje; al cierre del periodo; mensual',
        ('Inflación subyacente para dentro de ',
         'Inflación subyacente para el mes en curso',
         'Inflación subyacente para el siguiente mes'))

imprime_siguentes_variables(df_variables, n=3)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación subyacente: porcentaje; al cierre del periodo; a largo plazo',
        ('Inflación subyacente para los próximos'))

imprime_siguentes_variables(df_variables)
# Ver su IdVariable también.
print(df_variables.query('PrimerasLetras == "Inflaci" and Tema==""'))
# También se consultó: https://www.banxico.org.mx/SieInternet/consultarDirectori
# oInternetAction.do?sector=24&accion=consultarCuadro&idCuadro=CR155&locale=es
pone_tema_por_prefijo_variable(df_variables,
    'Inflación general: porcentaje; al cierre del periodo; 12 meses',
        ('Inflacióngeneral_12m_'))
imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación subyacente: porcentaje; al cierre del periodo; 12 meses',
        ('Inflaciónsubyacente_12m_'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Intensidad Competencia: nivel (1 a 7); _desconocido; _desconocido',
        ('Intensi'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Inversión Extranjera Directa: _desconocido; al cierre del periodo; anual',
        ('Inversi'))

imprime_siguentes_variables(df_variables)
imprime_siguentes_variables(df_variables, 11)
pone_tema_por_prefijo_variable(df_variables,
    'Tasa de fondeo interbancaria: porcentaje; al cierre del periodo; trimestral',
        ('Nivel de la tasa de fondeo interbancaria al cierre'))

imprime_siguentes_variables(df_variables)
imprime_siguentes_variables(df_variables, 5)
pone_tema_por_prefijo_variable(df_variables,
    'Tasa de interés de los Bonos M a 10 años: porcentaje; al cierre del periodo; anual',
        ('Nivel de la tasa de interés de los Bonos M a 10 años al cierre'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Tasa de interés del cete a 28 días: porcentaje; al cierre del periodo; anual',
        ('Nivel de la tasa de interés del cete a 28 días al cierre'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Obstáculos Enfrentan Empresarios: _desconocido; _desconocido; _desconocido',
        ('Obstáculos Enfrentan Empresarios'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'PIB, Probabilidad de reducción en el PIB trimestral: porcentaje; _desconocido; trimestral',
        ('Probabilidad de reducción en el PIB trimestral'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Requerimientos financieros del sector público: _desconocido; al cierre del periodo; anual',
        ('Saldo de requerimientos financieros del sector público al cierre del'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Sectores Problemas Competencia: _desconocido; _desconocido; _desconocido',
        ('Sectores Problemas Competencia'))

imprime_siguentes_variables(df_variables)
imprime_siguentes_variables(df_variables,4)
pone_tema_por_prefijo_variable(df_variables,
    'Tasa nacional de desocupación: porcentaje; al cierre del periodo; anual',
        ('Tasa nacional de desocupación al cierre'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Tasa nacional de desocupación: porcentaje; promedio del periodo; anual',
        ('Tasa nacional de desocupación promedio del '))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Tipo de cambio: tipo de cambio; al cierre del periodo; al cierre del año',
        ('Valor del tipo de cambio al cierre del año en curso'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Tipo de cambio: tipo de cambio; promedio del periodo; mensual',
        ('Valor del tipo de cambio promedio durante el mes'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'PIB, Variación desestacionalizada del PIB: porcentaje; al cierre del periodo; trimestral',
        ('Variación desestacionalizada del PIB'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Trabajadores asegurados: variación en el número de personas; al cierre del periodo; _desconocido',
        ('Variación en el número de trabajadores asegurados'))

imprime_siguentes_variables(df_variables,5)
pone_tema_por_prefijo_variable(df_variables,
    'PIB EEUUA, Variación porcentual anual del PIB de Estados Unidos: porcentaje; al cierre del periodo; anual',
        ('Variación porcentual anual del PIB de Estados Unidos'))

imprime_siguentes_variables(df_variables, width=200)
pone_tema_por_prefijo_variable(df_variables,
    'PIB, Variación porcentual anual del PIB: porcentaje de probabilidad en rango; al cierre del periodo; anual',
        ('Variación porcentual anual del PIB en '))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'PIB, Variación porcentual anual del PIB para los próximos 10 años: porcentaje; al cierre del periodo; a largo plazo',
        ('Variación porcentual anual del PIB para los próximos 10 años'))

imprime_siguentes_variables(df_variables, width=150)
pone_tema_por_prefijo_variable(df_variables,
    'PIB, Variación porcentual anual del PIB: porcentaje; al cierre del periodo; anual',
        ('Variación porcentual anual del PIB, año anterior al correspondiente del levantamiento de la Encuesta (año t-1)',
         'Variación porcentual anual del PIB, año en curso (año t)',
         'Variación porcentual anual del PIB, siguiente año (año t+1)',
         'Variación porcentual anual del PIB, dentro de dos años (año t+2)',
         'Variación porcentual anual del PIB, dentro de tres años (año t+3)'
         ))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'PIB, Variación porcentual anual del PIB: porcentaje; al cierre del periodo; trimestral',
        ('Variación porcentual anual del PIB, '))

imprime_siguentes_variables(df_variables)
print(df_variables.loc[df_variables['IdVariable'].str.startswith('coyun')])
print(df_variables.loc[df_variables['Variable'].str.startswith('cemp')])
pone_tema_por_prefijo_variable(df_variables,
    'Coyuntura empleo (?): nivel (bueno, malo, no seguro); _desconocido; _desconocido',
        ('cemp'))

imprime_siguentes_variables(df_variables)
print(df_variables.loc[df_variables['IdVariable'].str.startswith('clima')])
print(df_variables.loc[df_variables['Variable'].str.startswith('cneg')])
pone_tema_por_prefijo_variable(df_variables,
    'Cambio climático (?): nivel (empeorará, mejorará, permanecerá igual); _desconocido; _desconocido',
        ('cneg'))

imprime_siguentes_variables(df_variables)
print(df_variables.loc[df_variables['IdVariable'].str.startswith('ecopai')])
print(df_variables.loc[df_variables['Variable'].str.startswith('ep')])
pone_tema_por_prefijo_variable(df_variables,
    'Economía del país (?): nivel (no, sí); _desconocido; _desconocido',
        ('ep'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación general: porcentaje de probabilidad en rango; al cierre del periodo; 12 meses',
        ('inflacióngeneral_prob12m'))

imprime_siguentes_variables(df_variables)
pone_tema_por_prefijo_variable(df_variables,
    'Inflación subyacente: porcentaje de probabilidad en rango; al cierre del periodo; 12 meses',
        ('inflaciónsubyacente_prob12m'))

imprime_siguentes_variables(df_variables)
imprime_array(df_variables.loc[df_variables['Variable'].str.startswith('limcrec')]['Variable'])
pone_tema_por_prefijo_variable(df_variables,
    'Límite de crecimiento: nivel; _desconocido; anual',
        ('limcrec'))

if df_variables[df_variables['Tema']==''].shape[0] == 0:
    print('Se ha asignado tema a todas las variables.')
else:
    print('Existen variables que no se les ha asignado tema:')
    imprime_siguentes_variables(df_variables)
    reporta_error('Existen variables que no se les ha asignado tema')

# Quita las variables temporales que se usaron para el tema.
df_variables = df_variables.drop(['PrimerasLetras', 'DosPalabras'], axis = 1)

numero_temas = df_variables[['Tema']].drop_duplicates(keep='first').shape[0]
if numero_temas != 26:
    reporta_error(f'Cambió el número de temas: {numero_temas}')

print('Temas:')
imprime_temas()
print('Cifras:')
imprime_cifras()
print('Horizontes:')
imprime_horizontes()
print('Unidad:')
imprime_unidades()

print('Cuántas variables tiene cada tema:')
print(df_variables.groupby(['Tema'])['Variable'].count())
print('Los 5 temas con más variables:')
print(df_variables.groupby(['Tema'])['Variable'].count().sort_values(ascending=False).head(4))

# Asigna tema, cifra, horizonte, y unidad a los renglones del data set.
df = df.merge(df_variables, how='left', on=['IdVariable','Variable'])
df.columns
df_variables.columns
df = df.reindex(columns=[
    'Año', 'Mes', 'Fecha',
    'Tema', 'Cifra', 'Horizonte', 'Unidad',
    'IdVariable', 'Variable', 'IdAnalista', 'Expectativa'])
if (df.loc[df['Tema'] == ''].shape[0] > 0 |
    df.loc[df['Cifra'] == ''].shape[0] > 0 |
    df.loc[df['Horizonte'] == ''].shape[0] > 0 |
    df.loc[df['Unidad'] == ''].shape[0] > 0) :
    reporta_error('No todos los renglones quedaron con tema')

print('Número de renglones del DataFrame por tema:')
print_df(df.groupby(['Tema'])['Expectativa'].count()
             .rename('Número de Expectativas'))
print('Agrupaciones del DataFrame:')
print_df(df.groupby(['Tema', 'Unidad', 'Cifra'])['Expectativa'].count()
             .rename('Número de Expectativas'))


# 5. Correlaciones entre variables

# 5.1. Correlación entre variables de interés

# Elige algunas variables de temas de interés
temas=df_variables['Tema'].drop_duplicates(keep='first').sort_values().values
imprime_array(temas)
temas_interes=(temas[0], temas[1], temas[8], temas[9], temas[11], temas[21])
imprime_array(temas_interes)
df_variables_interes = df_variables.query('Tema in @temas_interes')

print(df_variables_interes.shape)
# En 6 temas de interés hay 193 variables
# Se elegirán las más variables más representativas de los temas de interés

# Se eligen las variables con horizonte anual
print(df_variables_interes[['Tema', 'Horizonte']].drop_duplicates())
df_variables_interes = df_variables_interes.query('Horizonte == "anual"')
print(df_variables_interes[['Tema', 'Horizonte']].drop_duplicates())

# Se eligen las variables con cifra de cierre del periodo
print(df_variables_interes[['Tema', 'Cifra']].drop_duplicates())
df_variables_interes = df_variables_interes.query(
    'Cifra == "al cierre del periodo"')
print(df_variables_interes[['Tema', 'Cifra']].drop_duplicates())

# Se descartan las variables con unidad de porcentaje de probabilidad en rango
print(df_variables_interes[['Tema', 'Unidad']].drop_duplicates())
imprime_array(df_variables_interes['Unidad'].drop_duplicates())
# Es necesario eliminar los de espacios en blanco
df_variables_interes['Unidad'] = df_variables_interes['Unidad'].str.strip()
df_variables_interes = df_variables_interes.query(
    'Unidad != "porcentaje de probabilidad en rango"')
print(df_variables_interes[['Tema', 'Unidad']].drop_duplicates())

# Estas son las variables representativas de los temas de interés
variables_interes = df_variables_interes['IdVariable'].values
imprime_array(variables_interes)

# Se elije una fecha, y se obtiene la correlación de dichas variables
df_interes = df.query('Fecha == "2024-09-01" & IdVariable in @variables_interes')
df_interes.shape

df_interes_varscols = df_interes.pivot(
    index=['IdAnalista'],
    columns=['IdVariable'],
    values='Expectativa')

df_interes_varscols.describe()

corr = df_interes_varscols.corr()
print(corr)

# Correlación en enero 2024 de variables de interés
fig, ax = plt.subplots(figsize=(10, 5))
paleta_divergente=sns.color_palette("vlag", as_cmap=True)
sns.heatmap(
    corr,
    annot=True,              # muestra los valores en cada celda
    fmt=".2f",
    mask=np.triu(corr),      # no muestra el triángulo superior
    cmap=paleta_divergente,  # colores para valores negativos y positivos
    center=0,                # el color central debe ser en el valor cero
    linewidths=1,            # lineas entre celdas
    square=True,             # celdas cuadradas
    ax=ax)
plt.tight_layout()
plt.show()
plt.close()

# En esta figura se demuestra que estuvo correcta la clasificación de las
# variables por tema dado que las variables del mismo tema y distinto horizonte
# muestran un alto valor de correlación.





# xxxx
# documentar los errores que fui encontrando
# la menor cantidad de tablas
# 
# mencionar   por los nombres de las columnas se puede saber tambien si son categoricas pero tienen muchos valores esto es porque los nombres relativos hacen referencia a variables que se estan midiendo en lugar de columnas son renglones
# 
# mencionar que en l cuadro no hay faltantes pero si un encuestado no responde no aparecen
# 
# en el marco teorico explicar brevement los temas
# 
# mencionar subrayar en la narracion mi aportacion de mi trabajo: separar en temas, unidad, horizonte, cifra
# y decir que solo trabajare sobre algunos temas
# 
# hacer una tabla con tema y unidad y cifra y horizonte, y tipo categorica o no, o numericas con valores de tal y tal
# mostrarlo como aportacion de mi trabajo
# 
# 
# en memoria mencionar que si se observa de facto lo esperado que las variables correlacionadas de diferentres horizontes estan correlacionados, muestra que el etiquetado de tema fue correcto
# xxxx




# 
# xxxxx
# 
# 
# infsubt
#     idanalistas
#         fecha
# 
# 
# 2024-09
#     idanalistas
#         inf sub t
#         bal com tmas1
# 
# 
# xxxx
# categorizacion por temas mediante kmeans
# 
# IdVariable   infsubt
# Variable     Inflacion Subyacente al cierre del mes t
# 
# => Tema> Inflacion Subyacente
#    Unidad> Porcentaje
#    Cifra>  Al cierre
#    Horizonte> mensual
# 
# 1)
# IdVariable  Variable
# 1           1
# 2           2
# ...         ...
# 
# 2)
# Entrenar con  4 variables codificadas:
# *IdVariable*  Unidad  Cifra  Horizonte
# 1             1       1      1
# 1             1       1      2
# 1             2       2      1
# ...         ...
# 
# y contrastar contra el numero de temas que ya obtuvimos semanticamente.
# 
# Tema ahorita:
# Ampliación del alcance del análisis
# del Banco de México de las
# Expectativas del Sector Privado
# utilizando métodos de minería de datos
# 
# 
# 
# De un tema, analizar las expectativas los analistas
# 
# eje x analistas
# eje y variable
# intensidad de color el valor
# 
# x analistas
# y una variable
# 
# x variables
# y valor
# 
# la cantidad de veces que cada analista dio su expectativa arriba o abajo del promedio, por mes, cuantas por variable
# para clasificar
# 
# de una solo tema ej inf, gen., con todas los horizontes y unidad
# kmeans / quitar fecha
# agregar columna del promedio
# idanalista, expectativa, promedio de la expectativa, ej2:diferencia entre ellas
# 1           2313         123                         n
# 1           122          23                          n
# 1           123123       32132                       n
# 
# 
# 
# xxxxx



# https://posit.co/blog/three-ways-to-program-in-python-with-rstudio/
# reticulate::py_install('pandas')
# reticulate::py_install('matplotlib')
# https://support.posit.co/hc/en-us/articles/1500007929061-Using-Python-with-the-RStudio-IDE


# primero probar con la maquina de soporte de vectores, 
# si no se puede con eso regresarse al EDA
# si se puede entonces ya elegir con alguno otro
# 
# 
# primero probar con la maquina de soporte de vectores, 
# si no se puede con eso regresarse al EDA
# si se puede entonces ya elegir con alguno otro
# 
# 
# 
# reduccion de caracteristicas: entrenar el modelo con menos columnas
# elegir las columnas con mayor correlacion


# https://rstudio.github.io/reticulate/articles/r_markdown.html

# RStudio
# library(reticulate)
# py_install("pandas")
# py_install("matplotlib")
# py_install("scikit-learn")
# py_install("seaborn")
# py_install("imbalanced_learn") # metricas clasificador, genera datos faltantes
# reticulate::virtualenv_install(packages = c("numpy==1.8.0"))
# system2(reticulate::py_exe(), c("-m", "pip", "uninstall -y", 'scikit-learn'))
# otra  py_install("requests==2.32.3")


# verificar encoding https://stackoverflow.com/questions/492483/setting-the-correct-encoding-when-piping-stdout-in-python




# Graficando
# 
# plt.figure(1)
# plt.subplot(211)
# plt.plot( \
#     df.query('Tema=="Valor del tipo de cambio promedio; durante el mes"') \
#         [['Fecha','Expectativa']].groupby('Fecha').mean().pct_change())
# plt.subplot(212)
# plt.plot( \
#     df.query('Variable=="Inflación general para el mes en curso (mes t)"') \
#         [['Fecha','Expectativa']].groupby('Fecha').mean())
# plt.show()
# plt.close()
# 
# 
# plt.plot( \
#     df.query('Tema=="Valor del tipo de cambio promedio; durante el mes"') \
#         [['Fecha','Expectativa']].groupby('Fecha').mean().pct_change()*10)
# plt.plot( \
#     df.query('Variable=="Inflación general para el mes en curso (mes t)"') \
#         [['Fecha','Expectativa']].groupby('Fecha').mean())
# plt.show()
# plt.close()

# 
# https://machinelearningmastery.com/time-series-data-stationary-python/
# df.query('Tema=="Valor del tipo de cambio promedio; durante el mes" \
#           and Año==2019 and Mes==8') \
#         [['Fecha','Expectativa']].groupby('Fecha').mean().hist()
# plt.show()
# plt.close()
# 
# 
# df.dtypes
# 
# df.query('Tema=="Valor del tipo de cambio promedio; durante el mes"') \
#     [['Fecha','Expectativa']].groupby('Fecha').mean().plot()
# plt.show()
# plt.close()
# 
# 
# df_temas=df[['Tema']].drop_duplicates()
# df_temas.query('Tema.str.contains("nflaci")').values
# # no se ve cual es del mes
# 
# df_variables=df[['Variable']].drop_duplicates()
# df_variables.query('Variable.str.contains("nflaci") and Variable.str.contains("mes") ').values
# # Inflación general para el mes en curso (mes t)
# 
# 
# df.query('Variable=="Inflación general para el mes en curso (mes t)"') \
#     [['Fecha','Expectativa']].groupby('Fecha').mean().plot()
# plt.show()
# plt.close()
# 
# 
# 
# 
# df.query('Tema=="Valor del tipo de cambio promedio; durante el mes"') \
#     [['Fecha','Expectativa']].groupby('Fecha').mean().plot()
# plt.show()
# plt.close()
# 
# 
# df.groupby
#   
#   .describe()
# 
# 
# import seaborn as sns
# 
# _=df.query('Año==2024 and Tema=="Valor del tipo de cambio promedio; durante el mes"')[['Fecha','Expectativa']]
# 
# # https://seaborn.pydata.org/tutorial/color_palettes.html
# sns.violinplot(x='Fecha', y='Expectativa', data=_, palette='pastel', inner='quart')
# sns.stripplot(x ='Fecha', y ='Expectativa', data=_, palette='bright', size=4)  
# plt.show()
# plt.close()
# 
# sns.histplot(data=_, x='Fecha', y='Expectativa')
# plt.show()
# plt.close()
# 
# # https://seaborn.pydata.org/tutorial/distributions.html
# sns.displot(_, y='Expectativa', hue="Fecha", kind="kde", fill=True)
# plt.show()
# plt.close()
# 
# 
# penguins = sns.load_dataset("penguins")
# sns.displot(penguins, x="flipper_length_mm", kind="kde", bw_adjust=.25)
# plt.show()
# plt.close()
# 
# 
# mu, sigma = 0, 0.1 # mean and standard deviation
# _=np.random.normal(mu, sigma, 1000)
# sns.violinplot(data=_, palette='Pastel1', inner='stick')
# sns.swarmplot(data=_, size=5, color='red')
# plt.show()
# plt.close()
# 
# xxxxxxxxxxxxxx
# print('Coeficiente de variación por Tema')
# 
# temaDescribe=df[['Tema','Expectativa']].pivot(columns='Tema').describe().T
# temaDescribe['cv'] = temaDescribe['std'] / temaDescribe['mean']
# print(temaDescribe[['cv']].sort_values(['cv']).to_string())
# 
# https://www.statology.org/coefficient-of-variation-in-python/
# 
# https://www.kaggle.com/code/ajay101tiwari/measures-of-dispersion-python-implementation
# 
# 
# sns.pairplot
# desde el EDA se pueden ver cuales variables pueden ayudar a separar
# la primera grafica de la diagonal muestra que hay inferencia entre las variables (se sobrepoinen): a partir de una variable se puede inferir otra: a ciertos clasificadores les costara trabajo separarlos.
# esto se llama extraccion de caracteristicas o atributos importantes
# 
# prueba de separabilidad de clases, para ver que variables ayudan a seperara en clases, esto lo muestra las graficas de la diagonal
# 
# 
# Dimensionalidad PCA (analisis de componentes principales)
# esto es extraccion de caracteristicas
# 
# 
# 
# ####
# 
# 
# 
# 
# 
# for column in df:
#     plt.figure()
#     df.boxplot([column])
# df.boxplot(['Expectativa'])
# df[['IdAnalista']].drop_duplicates().boxplot(['IdAnalista'])
# plt.show()
# plt.close()
# df[['IdAnalista']].drop_duplicates().describe()
# type()
# xxx
# # boxplot y violin
# # metodo 1
# import seaborn as sns
# df_plot=df[['IdAnalista']].drop_duplicates()
# sns.violinplot(data=df_plot, inner=None, color='white', linewidth=1)
# sns.boxplot(data=df_plot, width=0.3, color='orange')
# # metodo 2
# #plt.boxplot(df[['IdAnalista']].drop_duplicates().values)
# df[['IdAnalista']].drop_duplicates().boxplot(['IdAnalista'])
# plt.violinplot(df[['IdAnalista']].drop_duplicates().values)
# # muestra
# plt.show()
# plt.close()
# 
# 
# 
# bonost	Nivel de la tasa de interés del cete a 28 días
# bonost1	Nivel de la tasa de interés del cete a 28 días
# bonost2	Nivel de la tasa de interés del cete a 28 días
# bonost3	Nivel de la tasa de interés del cete a 28 días
# cetest	Nivel de la tasa de interés del cete a 28 días
# cetest1	Nivel de la tasa de interés del cete a 28 días
# cetest2	Nivel de la tasa de interés del cete a 28 días
# cetest3	Nivel de la tasa de interés del cete a 28 días
# fondeot	Nivel de la tasa de interés del cete a 28 días
# fondeotmas1	Nivel de la tasa de interés del cete a 28 días
# fondeotmas2	Nivel de la tasa de interés del cete a 28 días
# fondeotmas3	Nivel de la tasa de interés del cete a 28 días
# fondeotmas4	Nivel de la tasa de interés del cete a 28 días
# fondeotmas5	Nivel de la tasa de interés del cete a 28 días
# fondeotmas6	Nivel de la tasa de interés del cete a 28 días
# fondeotmas7	Nivel de la tasa de interés del cete a 28 días
# fondeotmas8	Nivel de la tasa de interés del cete a 28 días
# fondeotmas9	Nivel de la tasa de interés del cete a 28 días
# 
# Inflación general al cierre
# 
# 
# 
# xxx
# 
# 
# df_variables.query('Tema == "Nivel de la tasa de interés del cete a 28 días; al cierre del periodo; anual"')
# esto esta maaal se ve en los idvariable
# 
# x['PrimerasLetrasIdVariable']=x['IdVariable'].apply(lambda s: s[:10])
# ....
# 
# xxx
# 
# 
# 
# xxx
# # Ver si corresponden las primeras letras de IdVariable con Tema
# xxx ver si el idvariable no corresponde entonces tal vez se eligio mal el tema
# xxx o hay incongruencia en esas variables y habria que quitarlas
# df_variables[['IdVariable','Tema']].sort_values(['Tema'], ascending=False)
# 
# 
# df_variables.loc[df_variables['Tema']==''].shape
# if df_variables.loc[df_variables['Tema']==''].shape[0] > 0:
#     reporta_error('Aún hay variables sin tema.')
# 
# xxx
# 
# 
# 
# #### df=df.merge(df_variables, how='left', on=['IdVariable','Variable'])
# 
# 
# # **====== PENDIENTE:**
# # 
# # Convertir variables categóricas (si/no; mucho/poco/nada).
# # 
# # 5. Estadísticas descriptivas
# # 6. Visualización
# # 7. Análisis de variables
# # univariate, bivariate, or multivariate
# # 8. Análisis de series de tiempo
# # When we analyze time series data, we can typically uncover patterns or trends that repeat over time and present a temporal seasonality. Key components of time series data include trends, seasonal variations, cyclical variations, and irregular variations or noise.}
# # 
# 
# 
# # --------------------------------------------------------------------------
# # Estadísticas descriptivas
# 
# # Análisis de número de respuestas
# respuestasPorAño = df.groupby(by=["Año"])["Expectativa"].count()
# respuestasPorAño.name = 'Número de respuestas por año'
# respuestasPorAño.index.name = 'Año de la Enuesta'
# respuestasPorAño.to_frame().plot.bar(
#     title='Número de respuestas por año de la Encuesta (2024 año en curso)',
#     rot=70,
#     figsize=(10, 5),
#     color='darkblue')
# plt.show()
# plt.close()
# analistasDistintosPorAño = df.groupby(by=["Año"])["IdAnalista"].unique().apply(len)
# analistasDistintosPorAño.name = 'Número de analistas distintos'
# analistasDistintosPorAño.index.name = 'Año de la Enuesta'
# analistasDistintosPorAño.to_frame().plot.bar(
#     title='Número de analistas distintos por año de la Encuesta (2024 año en curso)',
#     rot=70,
#     figsize=(10, 5),
#     color ='darkred')
# plt.show()
# plt.close()
# analistasDistintosPorAño = df.groupby(by=["Año"])["Variable"].unique().apply(len)
# analistasDistintosPorAño.name = 'Número de preguntas distintas'
# analistasDistintosPorAño.index.name = 'Año de la Enuesta'
# analistasDistintosPorAño.to_frame().plot.bar(
#     title='Número de preguntas por año de la Encuesta (2024 año en curso)',
#     rot=70,
#     figsize=(10, 5),
#     color ='g')
# plt.show()
# plt.close()
# # **Por tanto, se concluye que el aumento de respuestas desde 2013 se podría explicar por el aumento de preguntas más que por el aumento de analistas.**
# 
# # Análisis de la Expectativa de Inflación General Anual
# inflacion_general_anual=df.query('IdVariable=="infgent"')
# inflacion_general_anual = inflacion_general_anual[['Año','Expectativa']] # Crea DataFrame con sólo estas dos columnas
# print(inflacion_general_anual)
# x=inflacion_general_anual.plot.scatter(
#     x='Año', y='Expectativa',
#     rot=70,
#     figsize=(10, 5),
#     color='purple', alpha=0.2)
# plt.show()
# plt.close()
# # Se asume que la distribución es normal, por lo que hacemos una gráfica de caja
# axes = inflacion_general_anual.boxplot(
#     column='Expectativa', by='Año',
#     ylabel='Porcentaje', xlabel='Año de la encuesta',
#     rot=70,
#     figsize=(10, 5),
#     color='purple')
# axes.set_title('Expectativa de Inflación General al cierre del año de la encuesta')
# plt.show()
# plt.close()
# 
# 
# # --------------------------------------------------------------------------
# # Correlaciones
# 
# # Calcula la correlación entre todas las variables y todos los analistas en todas las fechas.
# df_corrs=df_variables_en_columnas.corr()
# print(f'Son {df_variables_en_columnas.columns.size} variables.')
# df_corrs.sample(4)
# 
# f = plt.figure(figsize=(10, 10))
# plt.matshow(df_corrs, f)
# plt.show()
# plt.close()
# print('Son demasiadas variables para una sola gráfica.')
# 
# https://www.jmp.com/es_mx/statistics-knowledge-portal/what-is-correlation.html
# 
# 
# #df['Tema'].drop_duplicates()
# #df[df.Tema.str.contains('PIB')].groupby(['Tema'])['Expectativa'].mean()
# #df[df.Tema.str.contains('Cuenta')].groupby(['Tema'])['Expectativa'].mean()
# _1=df[df.Tema=='Variación porcentual anual del PIB; trimestral'] \
#     [['Fecha','Expectativa']].groupby(['Fecha']).mean()
# _2=df[df.Tema=='Cuenta Corriente; saldo anual al cierre del año; anual'] \
#     [['Fecha','Expectativa']].groupby(['Fecha']).mean()
# 
# 
# 
# 
# entre medias de las expectativas de inflacion y del pib, por ejemplo, a traves del tiempo
# 
# o entre las expectativas del analista 1 vs el 2 a traves del tiempo, y asi todos contra todos
# 
# calculando p?
# 
# 
# https://realpython.com/numpy-scipy-pandas-correlation-python/

# problema con no supervisado es random el ini io cengroide
# 
# https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
# 
# 
# Outliers Detection: Anomaly detection is the identification of rare or unusual data points. By clustering data into different groups and observing clusters with significantly fewer data points, the Elbow Method can assist in identifying anomalies or outliers more effectively.
# https://medium.com/@zalarushirajsinh07/the-elbow-method-finding-the-optimal-number-of-clusters-d297f5aeb189

# https://www.scikit-yb.org/en/latest/api/cluster/elbow.html


# Para normalizar, por ejemplo Rescaling (min-max normalization), https://en.wikipedia.org/wiki/Feature_scaling

# Kmeans con
# Distancia Manhatan = el absoluto de la distancia
# Distancia Chebyshev = el maximo de la distancia

# TSNE https://www.datacamp.com/es/tutorial/introduction-t-sne
# PCA https://www.datacamp.com/tutorial/principal-component-analysis-in-python
