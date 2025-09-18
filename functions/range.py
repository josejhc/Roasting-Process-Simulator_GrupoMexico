import numpy as np

def range(scaler_X, X_train, X_columns):
    # RANGE
    # Denormalize X_train
    X_train_original = scaler_X.inverse_transform(X_train)

    # Compute the range of input features (denormalized)
    min_values = np.min(X_train_original, axis=0)
    max_values = np.max(X_train_original, axis=0)

    print("\nRange of input features:")
    for i, (min_val, max_val) in enumerate(zip(min_values, max_values)):
        print(f"{X_columns[i]}: min = {min_val:.2f}, max = {max_val:.2f}")
    print('')
