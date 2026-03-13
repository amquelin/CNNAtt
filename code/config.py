from tensorflow.keras.callbacks import EarlyStopping

def fit_params(X_train, P_train, Y_train_scaled):
    return {
        "x": [X_train[:, 0], X_train[:, 1], P_train],  # Pop1, Pop2, Pos
        "y": Y_train_scaled,
        "epochs": 5,
        "batch_size": 32,
        "validation_split": 0.25,
        "shuffle": True,
        "callbacks": [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
    }

def pred_params(X_train, X_test, P_train, P_test):
    return {
        "x_train": [X_train[:, 0], X_train[:, 1], P_train],
        "x_test": [X_test[:, 0], X_test[:, 1], P_test],
    }

