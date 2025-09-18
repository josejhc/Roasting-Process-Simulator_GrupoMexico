import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from matplotlib.ticker import ScalarFormatter


def graphs_xgb(X_columns, Y_val, Y_val_pred, Y_columns, clean_data, xgb_model, target_variable,
               dias_array, presion_array, delta_presion_array, calcina, temp_garganta, temp_cama, oxigeno, concentrado):
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
    correlations = clean_data.corr()[target_variable].drop(target_variable)
    plt.figure(figsize=(10, 6))
    correlations.sort_values().plot(kind='barh', color='teal')
    plt.title(f'Correlation of other variables with \"{target_variable}\"')
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

    # 6. Feature importance from XGBoost estimators
    importances = np.mean([est.feature_importances_ for est in xgb_model.estimators_], axis=0)
    sorted_indices = np.argsort(importances)[::-1]
    sorted_X_columns = [X_columns[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_X_columns, sorted_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance (XGBoost)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '6_importancia_caracteristicas.png'))
    plt.close()

    # 7. Presión y Delta Presión vs Días
    if dias_array is not None and presion_array is not None and delta_presion_array is not None:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(dias_array, presion_array, 'bo-', label='Pressure')
        ax1.set_xlabel('Time(Day)')
        ax1.set_ylabel('Pressure (mm H2O)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.plot(dias_array, delta_presion_array, 'r--', label='Delta Pressure')
        ax2.set_ylabel('Delta Pressure', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title('Pressure (mm H2O) vs Delta Pressure vs Time (Day)')
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, '7_presion_vs_delta_presion.png'))
        plt.close()
    else:
        print("No se proporcionaron los arrays de días, presión y delta presión.")

    

    # 8. Calcina vs dias_array
    fig, ax = plt.subplots()
    ax.plot(dias_array, calcina, linewidth=2.0, label="Calcine Produced (Ton)")
    ax.set_xlabel("Time (Day)")
    ax.set_ylabel("Calcine Produced (Ton)")
    plt.title('Calcine vs Time (Day)')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '8_calcine_plot.png'))
    plt.close()

    # 9. Roaster Throat & Bed Temperature vs dias_array (con doble eje Y)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(dias_array, temp_garganta, 'r-', linewidth=2.0, label="Throat Temperature")
    ax2.plot(dias_array, temp_cama, 'b-', linewidth=2.0, label="Bed Temperature")

    ax1.set_xlabel("Time (Day)")
    ax1.set_ylabel("Throat Temperature (°C)", color='r')
    ax2.set_ylabel("Bed Temperature (°C)", color='b')

    plt.title('Roaster Throat & Bed Temperature (°C) vs Time (Day)')

    # Leyenda combinada
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax1.ticklabel_format(style='plain', axis='y', useOffset=False)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax2.ticklabel_format(style='plain', axis='y', useOffset=False)

    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '9_Throat_Bed_Temperature.png'))
    plt.close()

    # 10. Oxígeno vs dias_array
    fig, ax = plt.subplots()
    ax.plot(dias_array, oxigeno, linewidth=2.0, label="Oxygen Consumed (m3)")
    ax.set_xlabel("Time (Day)")
    ax.set_ylabel("Oxygen Consumed (m3)")
    plt.title('Oxygen Consumed (m3) vs Time (Day)')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '10_Oxygen_plot.png'))
    plt.close()

    # 11. Concentrado vs dias_array
    fig, ax = plt.subplots()
    ax.plot(dias_array, concentrado, linewidth=2.0, label="Roasted Concentrate (Ton)")
    ax.set_xlabel("Time (Day)")
    ax.set_ylabel("Roasted Concentrate (Ton)")
    plt.title('Roasted Concentrate (Ton) vs Time (Day)')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, '11_Concentrate_plot.png'))
    plt.close()
    
    
    # #8. Calcina vs dias_array
    # fig, ax = plt.subplots()
    # ax.plot(dias_array, calcina, linewidth=2.0, label="Calcine")

    # plt.title('Calcine vs Time (Day)')
    # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    # ax.ticklabel_format(style='plain', axis='y', useOffset=False)


    # plt.tight_layout()
    # plt.savefig(os.path.join(image_dir, '8_calcine_plot.png'))
    # plt.close()

    # #9. Temp_garganta vs dias_array
    # fig, ax = plt.subplots()
    # ax.plot(dias_array, temp_garganta, linewidth=2.0, label="Throat")

    # plt.title('Roaster Throat Temperature vs Time (Day)')
    # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    # plt.tight_layout()
    # plt.savefig(os.path.join(image_dir, '9_Throat_Temperature.png'))
    # plt.close()

    # #10. Temp_cama vs dias_array
    # fig, ax = plt.subplots()
    # ax.plot(dias_array, temp_cama, linewidth=2.0, label="Bed")

    # plt.title('Roaster Bed Temperature vs Time (Day)')
    # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    # plt.tight_layout()
    # plt.savefig(os.path.join(image_dir, '10_Bed_Temperature.png'))
    # plt.close()

    # #11. Oxigeno vs dias_array
    # fig, ax = plt.subplots()
    # ax.plot(dias_array, oxigeno, linewidth=2.0, label="Oxygen")

    # plt.title('Oxygen vs Time (Day)')
    # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    # plt.tight_layout()
    # plt.savefig(os.path.join(image_dir, '11_Oxygen_plot.png'))
    # plt.close()


    # #12. Concentrado vs dias_array
    # fig, ax = plt.subplots()
    # ax.plot(dias_array, concentrado, linewidth=2.0, label="Concentrate")

    # plt.title('Concentrate vs Time (Day)')
    # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    # plt.tight_layout()
    # plt.savefig(os.path.join(image_dir, '12_Concentrate_plot.png'))
    # plt.close()





