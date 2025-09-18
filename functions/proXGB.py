# === Imports ===
import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import boxcox

# === Custom Modules ===
from functions.XGBoost import modelXGB
from functions.cleaning_data import cleaning
from functions.graphsXGB import graphs_xgb

# === Transformation Functions ===
def apply_transform(name, series):
    """Applies a transformation to a data series based on the given name."""
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

def inverse_transform(name, series):
    """Applies the inverse of a transformation to a data series."""
    try:
        if name == 'identity':
            return series
        elif name == 'log1p':
            return np.expm1(np.clip(series, -20, 20))
        elif name == 'sqrt':
            return np.power(np.clip(series, 0, 1e10), 2)
        elif name == 'square':
            return np.sqrt(np.clip(series, 0, 1e10))
        else:
            return series
    except:
        return series

# === Transformation Selection ===
def best_transform_by_model(X, y, column_name=None):
    """Selects the best transformation for a target variable using model performance."""
    candidate_names = ['identity', 'log1p', 'sqrt', 'square']
    
    if column_name == 'delta_P':
        candidate_names = ['identity']  # delta_P should not be transformed
    if (y > 0).all():
        candidate_names.append('boxcox')

    best_mse = float('inf')
    best_name = 'identity'

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    for name in candidate_names:
        try:
            y_tr = apply_transform(name, y_train)
            y_va = apply_transform(name, y_val)
            model_tmp = XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
            model_tmp.fit(X_train, y_tr.ravel())
            pred = model_tmp.predict(X_val)
            mse = mean_squared_error(y_va, pred)
            if mse < best_mse:
                best_mse = mse
                best_name = name
        except:
            continue

    return best_name

# === Load and Clean Data ===
df = pd.read_csv('./data_bases/ffinal.csv')

selected_columns = df[[ 
    'Zn', 'Fe', 'Cu', 'Pb', 'Ins.', 'Flujo de aire al Tostador',
    'Humedad conc', 'Conc. humedo alimentado (ton/hr)', 'Agua alim. al Tostador (Lts/hr)',
    'Oxigeno consumido (m3)', 'Concentrado Tostado', 'Temp. Garganta', 'Prom. Cama Tostador',
    'Presion caja de viento', 'Produccion de calcina', 'Días Tostador', 'Temp. Amb. Prom *',
    'delta_P', 'ciclo_operativo'
]].dropna()

clean_data = cleaning(selected_columns)
ciclo = clean_data["ciclo_operativo"].max()

# === Main Prediction Function ===
def proXGB(stored_inputs, target_variable, weight):
    """Trains the model and generates predictions for multiple days."""
    
    # Define input and output columns
    X_columns = [
        'Zn', 'Fe', 'Cu', 'Pb', 'Ins.', 'Flujo de aire al Tostador',
        'Humedad conc', 'Conc. humedo alimentado (ton/hr)', 'Agua alim. al Tostador (Lts/hr)',
        'Días Tostador', 'Temp. Amb. Prom *', 'Presion caja de viento', 'ciclo_operativo'
    ]
    Y_columns = [
        'Concentrado Tostado', 'Temp. Garganta', 'Prom. Cama Tostador',
        'Produccion de calcina', 'Oxigeno consumido (m3)', 'delta_P'
    ]

    X = clean_data[X_columns].values
    Y = clean_data[Y_columns].values

    model_dir = "./results"
    os.makedirs(model_dir, exist_ok=True)
    transform_path = os.path.join(model_dir, "transform_dict.pkl")

    # === Apply or Load Transformations ===
    transform_dict = {}
    if os.path.exists(transform_path):
        print("Loading saved transformations...")
        transform_dict = joblib.load(transform_path)
        for i, col in enumerate(Y_columns):
            name = transform_dict[col]
            Y[:, i] = apply_transform(name, Y[:, i].reshape(-1, 1)).ravel()
            print(f"Variable '{col}' transformed with: {name} (loaded)")
    else:
        for i, col in enumerate(Y_columns):
            best_name = best_transform_by_model(X, Y[:, i].reshape(-1, 1), column_name=col)
            Y[:, i] = apply_transform(best_name, Y[:, i].reshape(-1, 1)).ravel()
            transform_dict[col] = best_name
            print(f"Variable '{col}' transformed with: {best_name}")
        joblib.dump(transform_dict, transform_path)
        print(f"Transformations saved to {transform_path}")

    # === Normalize Data and Train Model ===
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    Y = np.nan_to_num(Y, nan=0.0, posinf=1e10, neginf=-1e10)

    scaler_X, scaler_Y, X_train, Y_train, Y_val, xgb_model, Y_train_pred, Y_val_pred = modelXGB(X, Y)

    # === Inverse Transform Predictions ===
    Y_val_original = scaler_Y.inverse_transform(Y_val)
    Y_val_pred_original = scaler_Y.inverse_transform(Y_val_pred)

    for i, col in enumerate(Y_columns):
        Y_val_original[:, i] = inverse_transform(transform_dict[col], Y_val_original[:, i])
        Y_val_pred_original[:, i] = inverse_transform(transform_dict[col], Y_val_pred_original[:, i])

    # === Evaluation Metrics ===
    print("\n=== Model Evaluation ===")
    print(f"Train MSE: {mean_squared_error(Y_train, Y_train_pred):.4f}, Train R²: {r2_score(Y_train, Y_train_pred):.4f}")
    print(f"Validation MSE: {mean_squared_error(Y_val, Y_val_pred):.4f}, Validation R²: {r2_score(Y_val, Y_val_pred):.4f}\n")

    # === Simulation Loop ===
    clean_data_original = clean_data.copy()
    stored_inputs = np.append(stored_inputs, ciclo)
    duracion = int(round((weight / stored_inputs[7]) / 24, 0))

    # Initialize prediction containers
    calcina, temp_garganta, temp_cama = [], [], []
    oxigeno, concentrado, presion, delta_presion = [], [], [], []
    dias = list(range(1, duracion + 1))

    for dia in dias:
        new_data = np.array(stored_inputs).reshape(1, -1)
        new_data_scaled = scaler_X.transform(new_data)
        prediction = xgb_model.predict(new_data_scaled)
        prediction_denorm = scaler_Y.inverse_transform(prediction.reshape(1, -1))

        for i, col in enumerate(Y_columns):
            prediction_denorm[0][i] = inverse_transform(transform_dict[col], prediction_denorm[0][i])

        delta_P = prediction_denorm[0][5]
        stored_inputs[11] += delta_P

        # Append predictions
        calcina.append(prediction_denorm[0][3])
        temp_garganta.append(prediction_denorm[0][1])
        temp_cama.append(prediction_denorm[0][2])
        oxigeno.append(prediction_denorm[0][4])
        concentrado.append(prediction_denorm[0][0])
        presion.append(stored_inputs[11])
        delta_presion.append(delta_P)

    # === Generate Graphs ===
    graphs_xgb(
        X_columns, Y_val_original, Y_val_pred_original, Y_columns,
        clean_data_original, xgb_model, target_variable,
        dias, presion, delta_presion, calcina, temp_garganta, temp_cama, oxigeno, concentrado
    )

    # === Return Summary Statistics ===
    describe_df = clean_data_original.describe().reset_index()
    stats = describe_df.to_dict(orient="records")

    return prediction_denorm, stats, stored_inputs

# === Utility Function ===
def load_columns():
    """Returns the input and output column names used in the model."""
    X_columns = [
        'Zn', 'Fe', 'Cu', 'Pb', 'Ins.', 'Flujo de aire al Tostador',
        'Humedad conc', 'Conc. humedo alimentado (ton/hr)', 'Agua alim. al Tostador (Lts/hr)',
        'Días Tostador', 'Temp. Amb. Prom *', 'Presion caja de viento'
    ]
    Y_columns = [
        'Concentrado Tostado', 'Temp. Garganta', 'Prom. Cama Tostador',
        'Produccion de calcina', 'Oxigeno consumido (m3)', 'delta_P'
    ]
    return X_columns, Y_columns

