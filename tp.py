"""
modelo de regresion lineal que prediga el precio promedio del m² según las
variables seleccionadas y aplicando regularización Ridge para comparar
su desempeño. Evaluación de homocesticidad con Durwin-watson. Se implementó validación cruzada.
Evaluacion del modelo por MSE y R² 

fuente del dataset:
Gobierno de la Ciudad de Buenos Aires - Portal Buenos Aires Data
(https://data.buenosaires.gob.ar/dataset/mercado-inmobiliario)
inteligencia artificial de apoyo: ChatGPT

variables:
- barrio: zona geográfica donde se encuentra la propiedad
- año: año de observación
- trimestre: trimestre del año
- precio_prom: precio promedio del m² (USD)
- ambientes: cantidad de ambientes
- estado: condición del departamento (Usado o A estrenar)
- comuna: número de comuna de la propiedad

"""
# explorarcion y limpieza de datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
# cargamos el dataset que previamente elegimos con valores estandarizados
df = pd.read_csv("precio-venta-deptos.csv",sep=';')

print("🔹 Dimensiones:", df.shape) 

#Comprobacion de estado de las primero 5 filas.
print(f"\n{df.head()}\n")

#Revision de datos:

# info general con los tipos de datos para cada columna
print(f"\n{df.info()}\n")

#Datos que faltan segun columna:
print(f"\n valores nulos:\n{df.isnull().sum()}\n")

df['precio_prom'] = pd.to_numeric(df['precio_prom'], errors='coerce')
df['año'] = pd.to_numeric(df['año'], errors='coerce')
df['trimestre'] = pd.to_numeric(df['trimestre'], errors='coerce')
df = df.dropna()

# confirmamos estructura final
print("\n nuestro dataset limpio:", df.shape)

#Manejo de los outliers
for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # Reemplazamos los valores atípicos por la mediana de la columna
    median_value = df[col].median()
    df[col] = np.where((df[col] < lower) | (df[col] > upper), median_value, df[col])

print("\nReemplazamos outliers por la mediana en las variables numéricas.")

#Evaluacion del sesgo de la distribución de los valores sobre el eje x
print(f"\nLas medidas de asimetría son:\n{df.skew(numeric_only=True)}\n")

#Grafico que muestra la distribucion de cada columna
numerical_cols = df.select_dtypes(include=np.number).columns
skewness_values = df[numerical_cols].skew()

fig, axes = plt.subplots(nrows=1, ncols=len(numerical_cols), figsize=(4 * len(numerical_cols), 5))

if len(numerical_cols) == 1:
    axes = [axes]

for i, col in enumerate(numerical_cols):
    sns.histplot(data=df, x=col, ax=axes[i], kde=True)
    axes[i].set_title(f'{col}\nAsimetría: {skewness_values[col]:.2f}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

#Muestra en consola de la Kurtosis: muestra de datos alrededor de la media 
print(f"Las medidas de kurtosis son:\n{df.kurt(numeric_only=True)}\n")

# distribucion del precio promedio
plt.figure(figsize=(10,5))
sns.histplot(df['precio_prom'], bins=30, kde=True)
plt.title('distribución del precio promedio del m²')
plt.show()

# boxplot x estado
plt.figure(figsize=(8,5))
sns.boxplot(x='estado', y='precio_prom', data=df)
plt.title('precio promedio según estado')
plt.show()

# promedio x barrio
top_barrio = df.groupby('barrio')['precio_prom'].mean().sort_values(ascending=False).head(10)
top_barrio.plot(kind='bar', color='teal', figsize=(10,5))
plt.title('top 10 barrios con mayor precio promedio del m²')
plt.ylabel('precio promedio (USD/m²)')
plt.show()

# optimizacion y preprocesamiento
#Nota= si cuesta instalar las dependencias a sklearn hacer en terminal: 
#pip install -U scikit-learn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# codificar variables categoricas
df_encoded = pd.get_dummies(df, columns=['barrio', 'ambientes', 'estado', 'comuna'], drop_first=True)

# definir variables predictoras (X) y objetivo (y)
X = df_encoded.drop('precio_prom', axis=1)
y = df_encoded['precio_prom']

# division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# escalado de variables
#el scaler va a aprender de los datos de entrenamiento
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("datos preparados para modelado.\n")

from sklearn.linear_model import LinearRegression

#Con LinearRegression creamos el regresor para el modelo

regression = LinearRegression()
regression.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

#Al aplicar .predict(), pasa el regresor como argumento y obtiene la respuesta predicha correspondiente.

y_pred = regression.predict(X_test)
y_pred

# Predicciones sobre el conjunto de test
y_pred = regression.predict(X_test)

# Mostrar primeras predicciones
print(f"Primeras predicciones:\n{y_pred[:5]}\n")

# Comparar con valores reales
df_aux = pd.DataFrame({'Actual': y_test.values, 'Predicción': y_pred})
print("\nComparación de valores reales vs predichos:")
print(df_aux.head())

# Evaluación del modelo por MSE Y R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nError cuadrático medio (MSE): {mse:.2f}\n")
print(f"Coeficiente de determinación R²: {r2:.4f}\n")

#Escenario para comprobacion
nuevo_depto = pd.DataFrame({
    'barrio': ['Agronomia'], #Del barrio agronomia
    'año': [2011],            #en el año 2011
    'trimestre': [3],         #3er trimestre
    'ambientes': [2],         #"2 ambientes"
    'estado': ['Usado'],       #estado usado
    'comuna': [15],            #de la comuna 15

})

#Asigno columnas del modelo al nuevo ejemplo
nuevo_depto = nuevo_depto.reindex(columns=X.columns, fill_value=0)

# Escalar con el mismo scaler usado en entrenamiento
nuevo_depto_scaled = scaler.transform(nuevo_depto)

# Predecir el precio
prediccion_nueva = regression.predict(nuevo_depto_scaled)
print(f"\nPredicción del precio promedio (USD/m²) para el nuevo departamento: {prediccion_nueva[0]:.2f}\n")

#Obtener b_0 y 𝑏_1.
print(f"Intercepción del modelo: {regression.intercept_}\n")
print(f"Coeficientes (regression.coef_), longitud: {len(regression.coef_)}\n")

#Verificar la linealidad con un gráfico de residuos
residuals_test = y_test.values - y_pred

# Predichos vs residuos
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals_test, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valores predichos')
plt.ylabel('Residuos (y_test - y_pred)')
plt.title('Residuos vs Valores predichos (Test set)')
plt.show()

# Histograma de residuos
plt.figure(figsize=(8,4))
sns.histplot(residuals_test, kde=True)
plt.title('Distribución de residuos (Test set)')
plt.show()

#Para comprobar la homocedasticidad se puede examinar el gráfico de residuos generado. 
#Calculo y muestra de residuos por durbin-watson
# Calcula los residuos
residuals = y - regression.predict(X)

# Prueba de Durbin-Watson
from statsmodels.stats.stattools import durbin_watson

dw_test = durbin_watson(residuals_test)
print(f"Estadístico de Durbin-Watson: {dw_test}\n")

# 1.5 y 2.5 se utilizan como umbrales empíricos para interpretar el estadístico de Durbin-Watson.
if dw_test < 1.5:
    print("Posible autocorrelación positiva.\n")
elif dw_test > 2.5:
    print("Posible autocorrelación negativa.\n")
else:
    print("Los errores parecen ser independientes (no hay evidencia de autocorrelación).\n")

#Graficamos el conjunto de entrenamiento y de test
plt.scatter(df['año'], df['precio_prom'], alpha=0.6)
plt.title("Relación entre año y precio promedio del m²")
plt.xlabel("Año")
plt.ylabel("Precio promedio (USD/m²)")
plt.show()

#Validacion:
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# Definimos el modelo base (regresión lineal múltiple)
modelo_cv = LinearRegression()

# Configuramos K-Fold con 10 particiones (K = 10)
# shuffle=True para mezclar los datos antes de dividirlos
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Aplicar la validación cruzada usando R² como métrica
scores = cross_val_score(modelo_cv, X_train, y_train, cv=kfold, scoring='r2')

# Mostrar los resultados
print(f"Resultados R² de cada fold:\n{scores}\n")
print(f"Promedio del coeficiente de determinación R²: {scores.mean():.4f}")
print(f"Desviación estándar de R²: {scores.std():.4f}\n")

# Evaluar con el error cuadrático medio (negativo)
mse_scores = cross_val_score(modelo_cv, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
mse_mean = -mse_scores.mean()
print(f"Promedio del Error Cuadrático Medio (MSE) en validación cruzada: {mse_mean:.2f}\n")

#Regularizacion con Ridge
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)  
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print("R² con Ridge:", r2_score(y_test, y_pred_ridge))
print("MSE con Ridge:", mean_squared_error(y_test, y_pred_ridge))

