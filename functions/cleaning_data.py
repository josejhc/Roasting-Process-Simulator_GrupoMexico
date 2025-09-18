import numpy as np

def cleaning(tosta_why):
    """
    Cleans the input DataFrame by applying filters, removing NaNs,
    and enforcing value thresholds.
    """

    # === Step 1: Filter rows where Zn > 0 ===
    non_zero_zn = tosta_why['Zn'] > 0
    filtered_data = tosta_why[non_zero_zn]

    # === Step 2: Drop rows with any NaN values ===
    clean_data = filtered_data.dropna()

    # === Step 3: Convert all columns to float ===
    clean_data = clean_data.astype(float)

    # === Step 4: Remove rows with negative values (except for 'delta_P') ===
    col_with_negatives = 'delta_P'
    cols_without_negatives = clean_data.columns.difference([col_with_negatives])
    clean_data = clean_data[(clean_data[cols_without_negatives] >= 0).all(axis=1)]

    # === Step 5: Apply value thresholds to remove outliers ===

    # Maximum thresholds for specific columns
    max_thresholds = {
        'Flujo de aire al Tostador': 60000,
        'Produccion de calcina': 1000,
        'Conc. humedo alimentado (ton/hr)': 35,
        'Cu': 3,
        'Ins.': 5,
        'Pb': 2,
        '%S/S': 0.7,
        'Oxigeno consumido (m3)': 35000,
        'Días Tostador': 1,
        'Temp. Amb. Prom *': 46,
        'delta_P': 300,
    }

    # Minimum thresholds for specific columns
    min_thresholds = {
        'Flujo de aire al Tostador': 0.01,
        'Prom. Cama Tostador': 870,
        'Produccion de calcina': 450,
        '%S/S': 0.01,
        'Temp. Garganta': 800,
        'Oxigeno consumido (m3)': 0.01,
        'Pb': 0.01,
        'Cu': 0.01,
        'Ins.': 0.01,
    }

    # Apply maximum thresholds
    for col, max_val in max_thresholds.items():
        if col in clean_data.columns:
            clean_data = clean_data[clean_data[col] <= max_val]

    # Apply minimum thresholds
    for col, min_val in min_thresholds.items():
        if col in clean_data.columns:
            clean_data = clean_data[clean_data[col] >= min_val]

    return clean_data
