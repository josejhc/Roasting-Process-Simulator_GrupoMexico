



# # FUNCTIONAL

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
from sklearn.model_selection import GridSearchCV


def model(X, Y):
    # Normalize input features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Normalize target variables
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y)

    # Split into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

    model_path = "./results/forest_model.pkl"

    # Train and save model if it doesn't exist
    if not os.path.exists(model_path):
        print("Training model...")
        rf_model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=10,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=42,
            max_features="sqrt"
        )
        rf_model.fit(X_train, Y_train)
        joblib.dump(rf_model, model_path)
        print("Model saved to", model_path)
        print('')

    else:
        print("Loading pre-trained model...")
        print('')
        rf_model = joblib.load(model_path)

    # Make predictions
    Y_train_pred = rf_model.predict(X_train)
    Y_val_pred = rf_model.predict(X_val)

    return scaler_X, scaler_Y, X_train, Y_train, Y_val, rf_model, Y_train_pred, Y_val_pred






    #    param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 7],
    #     'min_samples_leaf': [1, 2, 3],
    #     'max_features': ['auto', 'sqrt']
    #     }

        
    #     # grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

    #     grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    #     grid_search.fit(X_train, Y_train)
    #     print('Mejores hiperparámetro: ', grid_search.best_params_)



# import os
# import joblib
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import xgboost as xgb

# def model(X, Y):
#     scaler_X = StandardScaler()
#     X_scaled = scaler_X.fit_transform(X)

#     scaler_Y = StandardScaler()
#     Y_scaled = scaler_Y.fit_transform(Y)

#     X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

#     model_path = "./results/xgb_model.pkl"

#     if not os.path.exists(model_path):
#         print("Training model...")
#         xgb_model = xgb.XGBRegressor(
#             n_estimators=300,
#             max_depth=6,  
#             learning_rate=0.1,
#             objective='reg:squarederror',
#             random_state=42,
#             verbosity=1
#         )
#         xgb_model.fit(X_train, Y_train.ravel())
#         joblib.dump(xgb_model, model_path)
#         print("Model saved to", model_path)
#     else:
#         print("Loading existing model...")
#         xgb_model = joblib.load(model_path)

#     # Retorno fuera del if
#     return scaler_X, scaler_Y, X_train, Y_train, Y_val, xgb_model, xgb_model.predict(X_train), xgb_model.predict(X_val)
