import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

def modelXGB(X, Y):
    # Escalado
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y)

    # División de datos
    X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

    model_path = "./results/xgb_model.pkl"

    if not os.path.exists(model_path):
        print("Training model...")

        base_model = XGBRegressor(
            n_estimators=1000,
            subsample=0.8,
            max_depth=7,
            learning_rate=0.02,
            objective='reg:squarederror',
            random_state=42,
            verbosity=1,
            colsample_bytree=0.8,
            # reg_alpha=0.1,
            # reg_lambda=1
        )

        xgb_model = MultiOutputRegressor(base_model)
        xgb_model.fit(X_train, Y_train)

        joblib.dump(xgb_model, model_path)
        print("Model saved to", model_path)
        print('')
    else:
        print("Loading existing model...")
        print('')
        xgb_model = joblib.load(model_path)

    # Predicciones
    Y_train_pred = xgb_model.predict(X_train)
    Y_val_pred = xgb_model.predict(X_val)

    return scaler_X, scaler_Y, X_train, Y_train, Y_val, xgb_model, Y_train_pred, Y_val_pred



