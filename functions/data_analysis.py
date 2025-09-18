import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Cargar el archivo CSV
df = pd.read_csv('./data_bases/analysis.csv')

# Eliminar columnas no deseadas
columnas_a_eliminar = ['Día','TI-B-1201','TI-B-1202','TI-B-1203','TI-B-1204','TI-B-1205','TI-B-1206','TI-B-1207','TI-B-1208-A',
                       'TI-B-1208','TI-B-1201-A','Temp. sal. CA-001 TI-B1216','Nivel Diesel DA-027','By-Pass Secado',
                       'Cantidad de Errores en Reporte','Existencia Calcina Total Lixi.','Existencia Calcina Silo Lixi.',
                       'Energia\nCA-301','Presión Cañon No. 1 (1er Turno)','Presión Cañon No. 2 (1er Turno)',
                       'Presión Cañon No. 3 (1er Turno)','Presión Cañon No. 4 (1er Turno)','Presión Cañon No. 5 (1er Turno)',
                       'Presión Cañon No. 6 (1er Turno)','Presión Cañon No. 7 (1er Turno)','Presión Cañon No. 8 (1er Turno)',
                       'Presión Cañon No. 9 (1er Turno)','Presión Cañon No. 10 (1er Turno)']

df = df.drop(columns=columnas_a_eliminar)

# Filtrar datos donde Zn > 0
filtered_data = df[df['Zn'] > 0]
clean_data = filtered_data.copy()

# Reemplazar valores inválidos por NaN
valores_invalidos = ['#DIV/0!', '', ' ', '#N/A', 'NULL', 'NaN', '#VALOR!', '---', '#N/D', '#REF!']
clean_data = clean_data.replace(valores_invalidos, np.nan)

# Convertir a float y eliminar filas con NaN
clean_data = clean_data.astype(float)
clean_data = clean_data.dropna()
# print(clean_data.count())

max_thresholds = {
        'Flujo de aire al Tostador': 60000,
        # 'Agua alim. al Tostador (Lts/hr)': 800,
        'Produccion de calcina': 1000,
        # 'Acido producido': 650,
        'Conc. humedo alimentado (ton/hr)': 50,
        'Cu':3,
        'Ins.':5,
        'Pb':2,
        '%S/S':1.1,
        # 'Conversión de gases': 100
        'Oxigeno consumido (m3)':60000
    }

    # Definir umbrales mínimos
min_thresholds = {
        'Flujo de aire al Tostador': 0,
        'Prom. Cama Tostador': 870,
        'Produccion de calcina': 450
    }

    # Filtrar filas que cumplen con todos los umbrales
for col, max_val in max_thresholds.items():
        if col in clean_data.columns:
            clean_data = clean_data[clean_data[col] <= max_val]

for col, min_val in min_thresholds.items():
        if col in clean_data.columns:
            clean_data = clean_data[clean_data[col] >= min_val]





# Suponiendo que ya tienes clean_data definido y limpio
# Paso 1: Eliminar variables con baja varianza
selector = VarianceThreshold(threshold=0.01)  # puedes ajustar el umbral
reduced_data = selector.fit_transform(clean_data)

# Obtener nombres de las columnas seleccionadas
columnas_seleccionadas = clean_data.columns[selector.get_support(indices=True)]

# Paso 2: Calcular matriz de correlación
correlation_matrix = clean_data[columnas_seleccionadas].corr()

# Paso 3: Eliminar variables altamente correlacionadas (> 0.95)
umbral_correlacion = 0.95
variables_a_eliminar = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > umbral_correlacion:
            colname = correlation_matrix.columns[i]
            variables_a_eliminar.add(colname)

# Variables finales
variables_finales = [col for col in columnas_seleccionadas if col not in variables_a_eliminar]

# # Mostrar resultados
# print("Variables seleccionadas por varianza:")
# print(list(columnas_seleccionadas))

# print("\nVariables eliminadas por alta correlación:")
# print(list(variables_a_eliminar))

# print("\nVariables finales más significativas:")
# print(variables_finales)

clean_data = clean_data[variables_finales]
print(clean_data.count())








# correlation_matrix = clean_data[variables_finales].corr()

# # Crear el mapa de calor
# plt.figure(figsize=(15, 12))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
# plt.title("Mapa de Correlación entre Variables")
# plt.tight_layout()
# plt.show()









# clean_data = clean_data.astype(float)
# # clean_data = clean_data[(clean_data >= 0).all(axis=1)]

# max_thresholds = {
#         'Flujo de aire al Tostador': 60000,
#         # 'Agua alim. al Tostador (Lts/hr)': 800,
#         'Produccion de calcina': 1000,
#         # 'Acido producido': 650,
#         'Conc. humedo alimentado (ton/hr)': 50,
#         'Cu':3,
#         'Ins.':5,
#         'Pb':2,
#         '%S/S':1.1,
#         # 'Conversión de gases': 100
#         'Oxigeno consumido (m3)':60000
#     }

#     # Definir umbrales mínimos
# min_thresholds = {
#         'Flujo de aire al Tostador': 0,
#         'Prom. Cama Tostador': 870,
#         'Produccion de calcina': 450
#     }

#     # Filtrar filas que cumplen con todos los umbrales
# for col, max_val in max_thresholds.items():
#         if col in clean_data.columns:
#             clean_data = clean_data[clean_data[col] <= max_val]

# for col, min_val in min_thresholds.items():
#         if col in clean_data.columns:
#             clean_data = clean_data[clean_data[col] >= min_val]




# # print(clean_data.describe())








