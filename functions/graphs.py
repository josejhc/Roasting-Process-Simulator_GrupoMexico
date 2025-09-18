
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
# (X_columns, Y_val, Y_val_pred, Y_columns, clean_data_original, rf_model, target_variable)
def graphs(X_columns, Y_val, Y_val_pred, Y_columns, clean_data, rf_model, target_variable):
    image_dir = os.path.join(os.path.dirname(__file__), '..', 'roasting_process', 'public', 'images')
    os.makedirs(image_dir, exist_ok=True)

    n = len(Y_columns)
    cols = 3
    rows = math.ceil(n / cols)

    # 1. Actual vs Predicted
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.scatter(Y_val[:, i], Y_val_pred[:, i], alpha=0.7, color='blue')
        plt.plot([Y_val[:, i].min(), Y_val[:, i].max()], [Y_val[:, i].min(), Y_val[:, i].max()], 'k--')
        plt.title(f'{Y_columns[i]}: Actual vs Prediction')
        plt.xlabel('Actual Value')
        plt.ylabel('Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '1_resultados_vs_predicciones.png'))
    plt.close()

    # 2. Correlation of target variable
    # Filtrar solo columnas numéricas
    # numeric_data = clean_data.select_dtypes(include=['number','float'])

    # Calcular correlaciones solo entre variables numéricas
    correlations = clean_data.corr()[target_variable].drop(target_variable)

    # correlations = clean_data.corr()[target_variable].drop(target_variable)
    plt.figure(figsize=(10, 6))
    correlations.sort_values().plot(kind='barh', color='teal')
    plt.title(f'Correlation of other variables with "{target_variable}"')
    plt.xlabel('Correlation Coefficient')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '2_correlacion_variable_objetivo.png'))
    plt.close()

    # 3. Residuals plot
    residuals = Y_val - Y_val_pred
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.scatter(Y_val_pred[:, i], residuals[:, i], alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(Y_columns[i])
        plt.xlabel('Prediction')
        plt.ylabel('Error')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '3_residuos_vs_prediccion.png'))
    plt.close()

    # 4. Histogram of residuals
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(residuals[:, i], kde=True, bins=20)
        plt.title(Y_columns[i])
        plt.xlabel('Error')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '4_histograma_errores.png'))
    plt.close()

    # 5. Correlation matrix
    plt.figure(figsize=(14, 10))
    sns.heatmap(clean_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Variable Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '5_matriz_correlacion.png'))
    plt.close()

    # 6. Feature importance
    importances = rf_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    sorted_X_columns = [X_columns[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_X_columns, sorted_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '6_importancia_caracteristicas.png'))
    plt.close()
