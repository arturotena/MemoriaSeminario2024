# ---------------------------------------------------------------------------
# Para instalar Python en RStudio
#
# library(reticulate)
# reticulate::repl_python()


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
np.random.seed(5)

# Se establece el número de columnas a visualizar, y otros valores relacionados.
pd.set_option('display.max_columns', 8) # 8 columnas por DataFrame
pd.set_option('display.width', 150) # 150 caracteres de ancho de pantalla
pd.set_option('display.max_colwidth', 200) # 200 caracteres por columna


# ----------------------------------------------------------------------------
# 2. Adquisición de datos

df_exp1 = pd.read_csv('Microdatos_2020_01.csv', encoding='latin-1')
df_exp2 = pd.read_csv('Microdatos_1999_01.csv', encoding='latin-1')
df = pd.concat([df_exp1, df_exp2],
               ignore_index=True) # no toma en cuenta los índices al concatenar


# --------------------------------------------------------------------------
# 3. Inspección inicial

# Obtener información general sobre los datos, tal como la cantidad de filas y
# columnas, los valores de los datos, y los tipos de datos.

# Dimensiones de los datos
rows, cols = df.shape
print(f'Hay {cols} columnas y {rows:,} renglones.')
if cols != 7: reporta_error(f"Se detectaron menos columnas que antes.")
if rows <= 1500000: reporta_error(f"Se detectaron menos renglones que antes.")
if rows > 1600000: reporta_error(f"Se detectaron más renglones que antes.")

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

print_df(df['NombreAbsolutoCorto']
             .apply(lambda s: len(s)).agg(['min', 'max']))
print_df(df['NombreAbsolutoLargo']
             .apply(lambda s: len(s)).agg(['min', 'max']))
print_df(df['NombreRelativoCorto']
             .apply(lambda s: len(s)).agg(['min', 'max']))
print_df(df['NombreRelativoLargo']
             .apply(lambda s: len(s)).agg(['min', 'max']))


# --------------------------------------------------------------------------
# 4. Preparación de los datos

# 4.1. Limpieza de los datos: búsqueda de duplicados.
s_duplicados = df.duplicated(keep=False)
cuenta_duplicados = s_duplicados[s_duplicados==True].size
print(f'Existen {cuenta_duplicados} renglones duplicados.')
if cuenta_duplicados > 0:
    reporta_error('Hay renglones duplicados y no se trataron')

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

# 4.4. Columnas calculadas
df['Año'] = df['FechaEncuesta'].dt.year
df['Mes'] = df['FechaEncuesta'].dt.month # número del mes
print_df(df.dtypes)

# 4.5. Simplificación de nombres de columnas
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

# 4.7. Búsqueda de renglones duplicados considerar en cuenta la columna Dato.
# Sólo debería haber un dato de expectativa para cada fecha, variable, analista.
s_duplicados=df[['Fecha', 'IdVariable', 'Variable', 'IdAnalista']] \
                 .duplicated(keep=False)
df.columns
print('Ejemplo de datos duplicados:')
print_df(df[s_duplicados]
             .sort_values(by=['Fecha', 'Variable', 'IdAnalista'])
             .head(6))

print(f'Existen: {s_duplicados[s_duplicados==True].size:,}'
      f' renglones duplicados, con la(s) variable(s):\n',
      df[s_duplicados][['IdVariable', 'Variable']]
          .drop_duplicates(keep='first'))
cuenta_original=df.shape[0]
df=df.drop_duplicates(subset=['Fecha', 'IdVariable',
                      'Variable', 'IdAnalista'], keep=False)
cuenta_sin_dups=df.shape[0]
porciento=(cuenta_original-cuenta_sin_dups)/cuenta_original*100
print(f'Antes {cuenta_original:,} renglones, ahora {cuenta_sin_dups:,}' +
      f' renglones; es decir '
      f'{porciento:.1f}% menos.')

# 4.8. Búsqueda de incongruencias en las variables
# Un valor en la columna IdVariable debe tener una sola Variable y viceversa;
# es decir estas columnas deben tener una correspondencia biunívoca.
#
# No se busca incongruencias de variable por fecha, sino en
# el DataFrame, por lo que si se encuentra una incongruencia en
# una variable en una fecha entonces se procede a eliminar dicha variable
# en todo el DataFrame.

def quita_duplicados(df_orig, df_busqueda, str_columna):
    """Elimina de df_orig los renglones que tengan en la str_columna
    los valores que estén repetidos en df_busqueda.
    Regresa: el DataFrame sin los renglones encontrados."""
    df_columna = df_busqueda[[str_columna]]
    # todos los valores duplicados
    s_duplicados_booleanos = df_columna.duplicated(keep=False)
    s_duplicados = (df_columna.loc[s_duplicados_booleanos]
                        .drop_duplicates(keep='first')[str_columna])
    # s_duplicados tiene solo los duplicados
    df_resultado = df_orig.query(str_columna + ' not in @s_duplicados')
    cuenta_eliminados = df_orig.query(str_columna + ' in @s_duplicados'
                            ).shape[0]
    cuenta_original = df.shape[0]
    pct = (1-(cuenta_original-cuenta_eliminados)/cuenta_original)*100
    print(f'Se eliminaron {cuenta_eliminados:,} renglones ({pct:.1f}%)'
          f' con {str_columna} duplicados:\n{s_duplicados.values}')
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
