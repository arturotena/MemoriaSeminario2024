# instalaciones

    !pip install jupyterlab
    
    
    import jupyterlab
    jupyterlab.__version__
    4.3.3
    
    !pip install notebook
    7.3.1
    
    
    https://blog.jupyter.org/the-jupytext-menu-is-back-3e6212e8c090
    https://security.snyk.io/package/npm/jupyterlab-jupytext
    https://snyk.io/advisor/python/jupytext
    https://pypi.org/project/jupytext/
    !pip install jupytext==1.16.0
    
    import os
    os.getcwd()
    os.chdir('..')
    os.getcwd()
    !jupyter notebook
    
# referencias

    https://www.epa.gov/caddis/exploratory-data-analysis

    https://www.datacamp.com/tutorial/principal-component-analysis-in-python
    https://www.datacamp.com/es/tutorial/introduction-t-sne


# -------------------------------------------------------------------------
# PENDIENTES


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
