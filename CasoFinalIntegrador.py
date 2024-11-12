''' 
Para este ejercicio, trabajaremos con un conjunto de datos de tráfico aéreo, que contiene las siguientes columnas:
    Year: Año del tráfico aéreo.
    Month: Mes del tráfico.
    Operating Airline: Aerolínea operativa.
    GEO Summary: Resumen geográfico (ej., Nacional o Internacional).
    GEO Region: Región geográfica (ej., US, Europa, etc.).
    Activity Type Code: Tipo de actividad (ej., Desembarque, Embarque).
    Passenger Count: Número de pasajeros.
'''
'''
Bloque 1: Comprensión de los Datos y Preparación
'''
'''
Paso 1: Cargar y Explorar los Datos
'''

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Cargar los datos desde un archivo CSV
csv_file_path = 'air_traffic_data.csv'
df = pd.read_csv(csv_file_path)

# Mostrar las primeras filas para entender la estructura del conjunto de datos
print(df.head())
# Obtener información general sobre el DataFrame
print(df.info())
print()

'''
Paso 2: Limpieza de Datos
'''

len_before = len(df)
print(f"Registros antes de eliminar duplicados y nulos: {len_before}")

# Identificar valores nulos
print(df.isnull().sum())
# Verificar si hay valores nulos en el DataFrame
has_nulls = df.isnull().any().any()
print(f"¿Hay valores nulos en el DataFrame?: {has_nulls}")
# Eliminar filas con valores nulos
df = df.dropna()

# Verificar que ya no hay valores nulos
print("\nVerificando valores nulos después de eliminarlos:")
print(df.isnull().sum())
print(f"\n¿Hay valores nulos en el DataFrame?: {df.isnull().any().any()}")
print(df.info())

# Eliminar registros duplicados
df = df.drop_duplicates()
# Verificar que se han eliminado duplicados
len_after = len(df)
print(f"Registros después de eliminar duplicados y nulos: {len_after}")
if len_before == len_after:
  print("No se ha eliminado ningun registro")
else:
  print("Registros eliminados: " + str(len_before - len_after))
print()


'''
Bloque 2: Filtrado y Creación de Subconjuntos de Datos
'''
'''
Paso 1: Filtrar por Aerolínea y Año
'''
# Para empezar, extraigamos datos de una aerolínea específica y un año determinado. Vamos a elegir la aerolínea "ATA Airlines" y el año 2005.

# Filtrar para ATA Airlines en 2005
df_ata_2005 = df[(df['Operating Airline'] == 'ATA Airlines') & (df['Year'] == 2005)]
# Mostrar el resultado
print('Filtrar para ATA Airlines en 2005')
print(df_ata_2005[['Operating Airline', 'Year', 'Month', 'Passenger Count']].head())
print()

'''
Paso 2: Filtrar por Tipo de Vuelo y Región Geográfica
'''
# Extraeremos datos de vuelos internacionales hacia Europa.
# Filtrar vuelos internacionales hacia Europa
df_international_europe = df[(df['GEO Summary'] == 'International') & (df['GEO Region'] == 'Europe')]
# Mostrar el resultado
print('Filtrar vuelos internacionales hacia Europa')
print(df_international_europe[['Operating Airline', 'GEO Region', 'Passenger Count']].head())
print()

'''
Paso 3: Filtrar Combinado con Condiciones Múltiples
'''
# Extraer datos de 2005 a 2020, solo para vuelos nacionales con más de 10,000 pasajeros.
# Filtrar por año, tipo de vuelo y conteo de pasajeros
df_large_national = df[(df['Year'].between(2005, 2020)) & (df['GEO Summary'] == 'Domestic') & (df['Passenger Count'] > 10000)]
# Mostrar el resultado
print('Extraer datos de 2005 a 2020, solo para vuelos nacionales con más de 10,000 pasajeros')
print(df_large_national[['Year', 'Operating Airline', 'Passenger Count']].head())
print()

# Ejercicio: Crear una consulta que muestre solo vuelos internacionales de aerolíneas específicas (ATA Airlines y Air France) en un rango de años (2005 y 2005-2009).
df_international_ata_2005 = df_ata_2005[(df['GEO Summary'] == 'International')]
print('Vuelos Internacionales de ATA Airlines en 2005')
print(df_international_ata_2005[['Operating Airline', 'GEO Region', 'Passenger Count']].head())
print()

print('Vuelos Internacionales de Air France entre 2005 y 2009')
df_international_airfrance = df[(df['GEO Summary'] == 'International') & (df['Operating Airline'] == 'Air France') & (df['Year'].between(2005, 2009))]
print(df_international_airfrance[['Operating Airline', 'GEO Summary', 'Passenger Count', 'Year']].head())
print()



'''
Bloque 3: Análisis Descriptivo y Correlación
'''
'''
Paso 1: Cálculo de Estadísticas Descriptivas
'''
# Realizaremos estadísticas descriptivas sobre el número de pasajeros.

# Calcular estadísticas descriptivas para el número de pasajeros
print(df['Passenger Count'].describe())

'''
Paso 2: Promedio de Pasajeros por Aerolínea
'''
# Calcular el promedio de pasajeros por aerolínea.

# Promedio de pasajeros por aerolínea
avg_passengers_airline = df.groupby('Operating Airline')['Passenger Count'].mean()
print('Promedio de pasajeros por aerolínea')
print(avg_passengers_airline.sort_values(ascending=False).head())

# Ejercicio: Comparar el número promedio de pasajeros por aerolínea y destino.
avg_passengers_airline_destination = df.groupby(['Operating Airline', 'GEO Region'])['Passenger Count'].mean()
print('Comparar el número promedio de pasajeros por aerolínea y destino.')
print(avg_passengers_airline_destination.sort_values(ascending=False).head())
print()


'''
Paso 3: Matriz de Correlación
'''
#Crear una matriz de correlación para analizar relaciones entre las variables numéricas.

# Convertir el DataFrame a un formato numérico si es necesario
df_numeric = df[['Year', 'Passenger Count']]
# Calcular la matriz de correlación
correlation_matrix_year = df_numeric.corr()
print('Matriz de correlacion entre Año y Pasajeros')
print(correlation_matrix_year)
print()

# Otras Matrices de Correlacion mas detalladas con Datos Categoricos
df_encoded = pd.get_dummies(df, columns=['Month'], prefix='m')
correlation_columns = [col for col in df_encoded.columns if 'm_' in col] + ['Passenger Count']
df_encoded_corr = df_encoded[correlation_columns]

correlation_matrix_month = df_encoded_corr.corr()
print("Matriz de correlacion entre cada Mes y Pasajeros")
print(correlation_matrix_month['Passenger Count'].sort_values())
print()

# Interpretar la correlación entre destinos y el número de pasajeros.
df_encoded_geo = pd.get_dummies(df, columns=['GEO Region'], prefix='GEO')
correlation_columns_geo = [col for col in df_encoded_geo.columns if 'GEO_' in col] + ['Passenger Count']
df_encoded_corr_geo = df_encoded_geo[correlation_columns_geo]

correlation_matrix_destinations = df_encoded_corr_geo.corr()
print("Matriz de correlacion entre cada Destino y Pasajeros")
print(correlation_matrix_destinations['Passenger Count'].sort_values())
print()

print('Las correlaciones negativas de GEO_Asia (-0.144), GEO_Europe (-0.114) y otras indican que existe una relación inversa entre estos destinos y el número de pasajeros. Por ejemplo, la correlación de GEO_Asia de -0.144 sugiere que los vuelos hacia Asia tienden a tener un número ligeramente menor de pasajeros en comparación con otros destinos. Cuanto más cercano esté el valor a -1, más fuerte será esta relación negativa, pero en este caso, las correlaciones son bastante débiles (cercanas a 0), lo que implica que, aunque estos destinos puedan tener una ligera tendencia a llevar menos pasajeros, la relación no es particularmente fuerte. Por otro lado, la correlación positiva de GEO_US (0.398) con el número de pasajeros indica que los vuelos hacia los EE. UU. tienden a tener un mayor número de pasajeros. Aunque esta correlación de 0.398 es moderada, es más fuerte que los otros valores y sugiere que los vuelos con más pasajeros están asociados con destinos en los EE. UU., lo que podría indicar que estos destinos son más populares o tienen mayor capacidad, resultando en un mayor número promedio de pasajeros.')
print()


'''
Bloque 4: Análisis de Clusters
'''
# En este bloque aplicaremos clustering para agrupar destinos con características similares en términos de tráfico de pasajeros.
'''
Paso 1: Preparar los Datos para Clustering
'''
# Seleccionaremos variables relevantes como Passenger Count y GEO Region, y normalizaremos los datos.

# Filtrar y escalar los datos numéricos necesarios
df_cluster = df[df['GEO Summary'] == 'International'][['GEO Region', 'Passenger Count']]
# This scales the Passenger Count column to have a mean of 0 and a standard deviation of 1, which is important for clustering algorithms. 
# This ensures that the algorithm is not biased by variables with larger numeric ranges.
df_cluster['Passenger Count'] = StandardScaler().fit_transform(df_cluster[['Passenger Count']])

# Convertir categorías de región geográfica a variables numéricas
df_cluster = pd.get_dummies(df_cluster, columns=['GEO Region'])
print(df['GEO Region'].unique())

'''
Paso 2: Aplicar K-Means Clustering
'''
# Usaremos el algoritmo K-Means para agrupar los destinos.

# Aplicar K-Means con 3 clusters (esto se puede ajustar)
kmeans = KMeans(n_clusters=3, random_state=0)
df_cluster['Cluster'] = kmeans.fit_predict(df_cluster)

# Mostrar los primeros resultados con su cluster asignado
print(df_cluster.head())

'''
Paso 3: Visualizar los Clusters
'''
# Para visualizar los clusters de destinos internacionales.

# Visualización simple de los clusters
plt.scatter(df_cluster['Passenger Count'], df_cluster['GEO Region_Europe'], c=df_cluster['Cluster'])
plt.xlabel('Passenger Count')
plt.ylabel('Region (Europe encoded)')
plt.title('Clusters de Destinos Internacionales')
plt.show()

# Visualización simple de los clusters con otra Region
plt.scatter(df_cluster['Passenger Count'], df_cluster['GEO Region_Asia'], c=df_cluster['Cluster'])
plt.xlabel('Passenger Count')
plt.ylabel('Region (Asia encoded)')
plt.title('Clusters de Destinos Internacionales')
plt.show()

# Visualización simple de los clusters con otra Region
plt.scatter(df_cluster['Passenger Count'], df_cluster['GEO Region_South America'], c=df_cluster['Cluster'])
plt.xlabel('Passenger Count')
plt.ylabel('Region (South America encoded)')
plt.title('Clusters de Destinos Internacionales')
plt.show()

## Understand the Clusters

'''
Bloque 5: Análisis Predictivo y Modelos de Regresión
'''
# Este bloque se centra en el uso de modelos predictivos para estimar el tráfico de pasajeros futuro.
'''
Paso 1: Preparar los Datos para el Modelo de Regresión
'''
# Aquí, seleccionaremos las variables necesarias y dividiremos los datos en conjuntos de entrenamiento y prueba.
'''
Paso 2: Aplicar un Modelo de Regresión Lineal
'''
# Usaremos una regresión lineal para predecir el número de pasajeros.

# MODELO 1
df_encoded = pd.get_dummies(df, columns=['Month'], prefix='m')

X_3 = df[['Year', 'Month']]   # Variables predictoras
X_3 = pd.get_dummies(X_3, columns=['Month'], prefix='Month')
y_3 = df['Passenger Count']   # Variable objetivo
# Dividir los datos en entrenamiento y prueba
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.2, random_state=42)
# Crear y entrenar el modelo de regresión lineal
model_3 = LinearRegression()
model_3.fit(X_train_3, y_train_3)
# Generar predicciones en el conjunto de prueba
y_pred_3 = model_3.predict(X_test_3)


# MODELO 2
##  Crear un modelo de regresión para predecir el tráfico de pasajeros de un destino específico.
X_2 = df[['Year', 'GEO Region']]   # Variables predictoras
y_2 = df['Passenger Count']   # Variable objetivo
X_2 = pd.get_dummies(X_2, columns=['GEO Region'], prefix='GEO')

# Dividir los datos en entrenamiento y prueba
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model_2 = LinearRegression()
model_2.fit(X_train_2, y_train_2)
y_pred_2 = model_2.predict(X_test_2)


# MODELO 3
X = df[['Year', 'Month', 'GEO Summary', 'Operating Airline']]
# Encoding categorical variables with one-hot encoding
X = pd.get_dummies(X, columns=['Month', 'GEO Summary', 'Operating Airline'], drop_first=True)  # `drop_first=True` to avoid multicollinearity
# Defining the target variable
y = df['Passenger Count']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Making predictions on the test set
y_pred = model.predict(X_test)


'''
Paso 3: Evaluar el Modelo
'''
# Evaluaremos la precisión del modelo usando métricas como RMSE y MAE.

# Calcular las métricas de evaluación
# Comparar diferentes modelos de regresión y elegir el más adecuado.
print('Modelo 1')
rmse_3 = mean_squared_error(y_test_3, y_pred_3, squared=False)
mae_3 = mean_absolute_error(y_test_3, y_pred_3)

print("Coeficientes del modelo:", model_3.coef_)
print("Intercepto del modelo:", model_3.intercept_)
print("Error cuadrático medio (RMSE):", rmse_3)
print("Error absoluto medio (MAE):", mae_3)

print('Modelo 2')
rmse_2 = mean_squared_error(y_test_2, y_pred_2, squared=False)
mae_2 = mean_absolute_error(y_test_2, y_pred_2)

print("Coeficientes del modelo:", model_2.coef_)
print("Intercepto del modelo:", model_2.intercept_)
print("Error cuadrático medio (RMSE):", rmse_2)
print("Error absoluto medio (MAE):", mae_2)


print('Modelo 3')
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print("Coeficientes del modelo:", model.coef_)
print("Intercepto del modelo:", model.intercept_)
print("Error cuadrático medio (RMSE):", rmse)
print("Error absoluto medio (MAE):", mae)

# Imprimir los valores completos del modelo elegido
print("Valores del modelo mas eficiente:")
print("Coeficientes del modelo:", model.coef_)
print("Intercepto del modelo:", model.intercept_)
print("Error cuadrático medio (RMSE):", rmse)
print("Error absoluto medio (MAE):", mae)
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio (MSE): {mse}")

'''
Bloque 6: Visualización y Presentación de Resultados
'''
# Este bloque se enfoca en crear visualizaciones para comunicar los hallazgos del análisis y las predicciones.
'''
Paso 1: Visualizar el Tráfico Aéreo Anual
'''
# Usaremos gráficos de líneas para mostrar la evolución del tráfico aéreo en el tiempo.

# Agrupar datos por año y calcular el total de pasajeros
yearly_passengers = df.groupby('Year')['Passenger Count'].sum()

# Crear gráfico de líneas
plt.figure(figsize=(10, 6))
plt.plot(yearly_passengers.index, yearly_passengers.values, marker='o')
plt.title('Evolución Anual del Tráfico de Pasajeros')
plt.xlabel('Año')
plt.ylabel('Número de Pasajeros')
plt.grid()
plt.show()

#  Generar un gráfico de líneas que muestre el tráfico anual de una aerolínea específica.
yearly_passengers_AirFrance = df[df['Operating Airline'] == 'Air France'].groupby('Year')['Passenger Count'].sum()
# Crear gráfico de líneas
plt.figure(figsize=(10, 6))
plt.plot(yearly_passengers_AirFrance.index, yearly_passengers_AirFrance.values, marker='o')
plt.title('Evolución Anual del Tráfico de Pasajeros de AirFrance')
plt.xlabel('Año')
plt.ylabel('Número de Pasajeros')
plt.grid()
plt.show()


yearly_passengers_ATA = df[df['Operating Airline'] == 'ATA Airlines'].groupby('Year')['Passenger Count'].sum()
# Crear gráfico de líneas
plt.figure(figsize=(10, 6))
plt.plot(yearly_passengers_ATA.index, yearly_passengers_ATA.values, marker='o')
plt.title('Evolución Anual del Tráfico de Pasajeros de ATA Airlines')
plt.xlabel('Año')
plt.ylabel('Número de Pasajeros')
plt.grid()
plt.show()

'''
Paso 2: Visualización de Predicciones vs. Valores Reales
'''
# Compararemos los valores reales y predichos en el conjunto de prueba para visualizar el desempeño del modelo.
# Plotting actual vs. predicted values for visual comparison
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Valores Reales", color="blue", alpha=0.6)
plt.plot(y_pred, label="Valores Predichos", color="orange", alpha=0.6)
plt.title("Comparación de Predicciones vs. Valores Reales")
plt.xlabel("Índice")
plt.ylabel("Número de Pasajeros")
plt.legend()
plt.show()

y_test_clipped = pd.Series(y_test.values).clip(upper=200000)
y_pred_clipped = pd.Series(y_pred).clip(upper=200000)

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.plot(y_test_clipped, label='Valores Reales', color='blue')
plt.plot(y_pred_clipped, label='Valores Predichos', color='orange')
plt.title('Comparación de Predicciones vs. Valores Reales (Capped at 200,000)')
plt.xlabel('Índice')
plt.ylabel('Número de Pasajeros')
plt.legend()
plt.show()

indice = range(len(y_test))
# Gráfico de error absoluto
error_absoluto = np.abs(y_test - y_pred)
plt.figure(figsize=(12, 6))
plt.plot(indice, error_absoluto, label="Error Absoluto", color="red", alpha=0.7)
plt.xlabel("Índice")
plt.ylabel("Error Absoluto")
plt.title("Error Absoluto entre Valores Reales y Predichos")
plt.legend()
plt.show()
# Gráfico de Dispersión para ver la correlación entre reales y predichos
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, color="purple")
plt.plot([0, max(y_test)], [0, max(y_test)], color="black", linestyle="--")  # Línea de identidad
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("Gráfico de Dispersión: Valores Reales vs. Predichos")
plt.show()
# Histograma del Error Absoluto
plt.figure(figsize=(10, 6))
plt.hist(error_absoluto, bins=50, color="skyblue", edgecolor="black")
plt.xlabel("Error Absoluto")
plt.ylabel("Frecuencia")
plt.title("Histograma del Error Absoluto entre Valores Reales y Predichos")
plt.show()
# Gráfico de Líneas con Zoom en los primeros 100 puntos
plt.figure(figsize=(12, 6))
plt.plot(indice[:100], y_test[:100], label="Valores Reales", color="blue", alpha=0.6)
plt.plot(indice[:100], y_pred[:100], label="Valores Predichos", color="orange", alpha=0.6)
plt.xlabel("Índice")
plt.ylabel("Número de Pasajeros")
plt.title("Comparación de Predicciones vs. Valores Reales (Primeros 100 Puntos)")
plt.legend()
plt.show()