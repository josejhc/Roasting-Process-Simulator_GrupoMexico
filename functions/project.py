# v1

import os
import pandas as pd
import numpy as np
import joblib
from functions.model import model
from functions.cleaning_data import cleaning
from functions.graphs import graphs
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import boxcox

# -------------------------
# Función para aplicar transformaciones por nombre
# -------------------------
def apply_transform(name, series):
    try:
        if name == 'identity':
            return series
        elif name == 'log1p':
            return np.log1p(series.clip(min=0))
        elif name == 'sqrt':
            return np.sqrt(series.clip(min=0))
        elif name == 'square':
            return np.power(series.clip(min=0), 2)
        elif name == 'boxcox':
            return boxcox(series)[0]
        else:
            return series
    except:
        return series

# -------------------------
# Función para invertir transformaciones
# -------------------------
def inverse_transform(name, series):
    try:
        if name == 'identity':
            return series
        elif name == 'log1p':
            return np.expm1(np.clip(series, -20, 20))  # evita overflow
        elif name == 'sqrt':
            return np.power(np.clip(series, 0, 1e10), 2)
        elif name == 'square':
            return np.sqrt(np.clip(series, 0, 1e10))
        else:
            return series
    except:
        return series

# -------------------------
# Función para elegir la mejor transformación
# -------------------------
def best_transform_by_model(X, y):
    candidate_names = ['identity', 'log1p', 'sqrt', 'square']
    if (y > 0).all():
        candidate_names.append('boxcox')

    best_mse = float('inf')
    best_name = 'identity'

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    for name in candidate_names:
        try:
            y_tr = apply_transform(name, y_train)
            y_va = apply_transform(name, y_val)
            # model_tmp = RandomForestRegressor(n_estimators=50, random_state=42)

            model_tmp = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=42,
            max_features="sqrt"
        )



            model_tmp.fit(X_train, y_tr.ravel())
            pred = model_tmp.predict(X_val)
            mse = mean_squared_error(y_va, pred)
            if mse < best_mse:
                best_mse = mse
                best_name = name
        except:
            continue

    return best_name

# -------------------------
# Función principal
# -------------------------
def project(stored_inputs, target_variable):
    os.system("cls")

    df = pd.read_csv('./data_bases/ffinal.csv')
    
    selected_columns = df[[ 
        'Zn', 'Fe', 'Cu', 'Pb', 'Ins.', 'Flujo de aire al Tostador',
        'Humedad conc', 'Conc. humedo alimentado (ton/hr)','Agua alim. al Tostador (Lts/hr)','Oxigeno consumido (m3)',
        'Concentrado Tostado', 'Temp. Garganta', 'Prom. Cama Tostador',
        'Presion caja de viento', 'Produccion de calcina','Días Tostador', 'Temp. Amb. Prom *'
    ]]

    clean_data = cleaning(selected_columns)
    # print(clean_data.count())
    X_columns = [
        'Zn', 'Fe', 'Cu', 'Pb', 'Ins.', 'Flujo de aire al Tostador',
        'Humedad conc', 'Conc. humedo alimentado (ton/hr)','Agua alim. al Tostador (Lts/hr)','Oxigeno consumido (m3)','Días Tostador', 'Temp. Amb. Prom *'
    ]
    Y_columns = [
        'Concentrado Tostado','Temp. Garganta', 'Prom. Cama Tostador', 'Presion caja de viento', 'Produccion de calcina', 
        # '%S/S'
    ]

    X = clean_data[X_columns].values
    Y = clean_data[Y_columns].values

    # -------------------------
    # Transformaciones automáticas con caching
    # -------------------------
    model_dir = "./results"
    os.makedirs(model_dir, exist_ok=True)
    transform_path = os.path.join(model_dir, "transform_dict.pkl")

    transform_dict = {}
    if os.path.exists(transform_path):
        print("Cargando transformaciones guardadas...")
        transform_dict = joblib.load(transform_path)
        for i, col in enumerate(Y_columns):
            name = transform_dict[col]
            Y[:, i] = apply_transform(name, Y[:, i].reshape(-1,1)).ravel()
            print(f"Variable '{col}' transformada con: {name} (cargada)")
    else:
        for i, col in enumerate(Y_columns):
            best_name = best_transform_by_model(X, Y[:, i].reshape(-1,1))
            Y[:, i] = apply_transform(best_name, Y[:, i].reshape(-1,1)).ravel()
            transform_dict[col] = best_name
            print(f"Variable '{col}' transformada con: {best_name}")
        joblib.dump(transform_dict, transform_path)
        print(f"Transformaciones guardadas en {transform_path}")

    # Limpiar NaN o inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    Y = np.nan_to_num(Y, nan=0.0, posinf=1e10, neginf=-1e10)

    # -------------------------
    # Entrenamiento
    # -------------------------
    scaler_X, scaler_Y, X_train, Y_train, Y_val, rf_model, Y_train_pred, Y_val_pred = model(X, Y)

    # Y_val_original = scaler_Y.inverse_transform(Y_val.reshape(-1,1))
    # Y_val__pred_original = scaler_Y.inverse_transform(Y_val_pred.reshape(-1,1))
    print('Creando Gráficos')
    Y_val_original = scaler_Y.inverse_transform(Y_val)
    Y_val_pred_original = scaler_Y.inverse_transform(Y_val_pred)

    for i, col in enumerate(Y_columns):
        Y_val_original[:, i] = inverse_transform(transform_dict[col], Y_val_original[:, i])
        Y_val_pred_original[:, i] = inverse_transform(transform_dict[col], Y_val_pred_original[:, i])





    # -------------------------
    # Evaluación general
    # -------------------------
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    train_r2 = r2_score(Y_train, Y_train_pred)
    val_mse = mean_squared_error(Y_val, Y_val_pred)
    val_r2 = r2_score(Y_val, Y_val_pred)

    print(f"=== Evaluación General ===")
    print(f"Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}")
    print(f"Validation MSE: {val_mse:.4f}, Validation R²: {val_r2:.4f}\n")

    cv_scores = cross_val_score(rf_model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation MSE: {-cv_scores.mean():.4f}\n")

    # print("=== Evaluación por Variable (Validación) ===")
    # for i, col in enumerate(Y_columns):
    #     mse_val = mean_squared_error(Y_val[:, i], Y_val_pred[:, i])
    #     r2_val = r2_score(Y_val[:, i], Y_val_pred[:, i])
    #     mse_train = mean_squared_error(Y_train[:, i], Y_train_pred[:, i])
    #     r2_train = r2_score(Y_train[:, i], Y_train_pred[:, i])
    #     print(f"{col} - Train MSE: {mse_train:.4f}, Train R²: {r2_train:.4f} | Validation MSE: {mse_val:.4f}, Validation R²: {r2_val:.4f}")

    # print("\n=== Validación Cruzada por Variable (Entrenamiento) ===")
    # for i, col in enumerate(Y_columns):
    #     scores = cross_val_score(rf_model, X_train, Y_train[:, i], cv=5, scoring='neg_mean_squared_error')
    #     print(f"{col} - CV MSE: {-scores.mean():.4f}")

    # -------------------------
    # Revertir transformaciones para gráficas
    # -------------------------
    # Crear copia del DataFrame original
    clean_data_original = clean_data.copy()

    # Reemplazar las columnas Y con sus versiones transformadas inversas
    for i, col in enumerate(Y_columns):
        # Usar los datos transformados que están en Y[:, i]
        clean_data_original[col] = inverse_transform(transform_dict[col], Y[:, i])



       

    graphs(X_columns, Y_val_original, Y_val_pred_original, Y_columns, clean_data_original, rf_model, target_variable)

    # -------------------------
    # Predicción para nuevos datos
    # -------------------------
    new_data = np.array(stored_inputs).reshape(1, -1)
    new_data_scaled = scaler_X.transform(new_data)
    prediction = rf_model.predict(new_data_scaled)
    prediction_denorm = scaler_Y.inverse_transform(prediction.reshape(1,-1))

    for i, col in enumerate(Y_columns):
        prediction_denorm[0][i] = inverse_transform(transform_dict[col], prediction_denorm[0][i])

    describe_df = clean_data_original.describe().reset_index()
    stats = describe_df.to_dict(orient="records")

    return prediction_denorm, stats, X_columns, Y_columns

def load_columns():
    df = pd.read_csv('./data_bases/ffinal.csv')

    X_columns = [
        'Zn', 'Fe', 'Cu', 'Pb', 'Ins.', 'Flujo de aire al Tostador',
        'Humedad conc', 'Conc. humedo alimentado (ton/hr)','Agua alim. al Tostador (Lts/hr)','Oxigeno consumido (m3)','Días Tostador', 'Temp. Amb. Prom *'
    ]
    Y_columns = [
        'Concentrado Tostado','Temp. Garganta', 'Prom. Cama Tostador',
        'Presion caja de viento', 'Produccion de calcina', 
        # '%S/S'
    ]

    return X_columns, Y_columns





