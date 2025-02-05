import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


PARQUET_PATH = r"C:\Users\Dev\OneDrive\Desktop\hbb_production_parquet\ntuple_merged_10_optimized.parquet"

BATCH_SIZE = 1024
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# Feature columns to be used
FEATURE_COLUMNS = [
    'fj_jetNTracks', 'fj_nSV',
    'fj_tau0_trackEtaRel_0', 'fj_tau0_trackEtaRel_1', 'fj_tau0_trackEtaRel_2',
    'fj_tau1_trackEtaRel_0', 'fj_tau1_trackEtaRel_1', 'fj_tau1_trackEtaRel_2',
    'fj_tau_flightDistance2dSig_0', 'fj_tau_flightDistance2dSig_1',
    'fj_tau_vertexDeltaR_0',
    'fj_tau_vertexEnergyRatio_0', 'fj_tau_vertexEnergyRatio_1',
    'fj_tau_vertexMass_0', 'fj_tau_vertexMass_1',
    'fj_trackSip2dSigAboveBottom_0', 'fj_trackSip2dSigAboveBottom_1',
    'fj_trackSip2dSigAboveCharm_0',
    'fj_trackSipdSig_0', 'fj_trackSipdSig_0_0', 'fj_trackSipdSig_0_1',
    'fj_trackSipdSig_1', 'fj_trackSipdSig_1_0', 'fj_trackSipdSig_1_1',
    'fj_trackSipdSig_2', 'fj_trackSipdSig_3',
    'fj_z_ratio'
]

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
def load_and_preprocess_data():
    # Load DataFrame from the Parquet file
    df = pd.read_parquet(PARQUET_PATH)

    # Apply cuts on mass/pt if available
    required_cols = ['fj_sdmass', 'fj_pt']
    if all(col in df.columns for col in required_cols):
        mask = (df['fj_sdmass'] > 40) & (df['fj_sdmass'] < 200) & (df['fj_pt'] > 300) & (df['fj_pt'] < 2000)
        df = df[mask].copy()

    # Create labels if they don't exist
    if 'label' not in df.columns:
        # Create composite labels (modify as needed)
        df['isHbb'] = df['fj_isH'] * df['fj_isBB']
        df['isQCD'] = df['fj_isQCD'] * df['sample_isQCD']
        df = df[(df['isHbb'] + df['isQCD']) == 1].copy()
        df['label'] = df['isHbb'].astype(int)

    # Select features and fill missing values
    df_features = df[FEATURE_COLUMNS].fillna(-999)
    X = df_features.values.astype(np.float32)
    y = tf.keras.utils.to_categorical(df['label'].values, num_classes=2)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['label']
    )

    # Normalize features using training set statistics
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1e-8  # Avoid division by zero

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    return X_train, X_test, y_train, y_test

# =============================================================================
# Prototype Model Definition
# =============================================================================
def build_prototype_model(input_shape):
    """Builds and compiles the prototype model with an improved architecture."""
    model = models.Sequential([  
        layers.Input(shape=input_shape),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            metrics.AUC(name='auc'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )
    return model

# =============================================================================
# Training, Evaluation, and Plotting
# =============================================================================
def train_and_evaluate_prototype():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    input_shape = (X_train.shape[1],)

    # Build the prototype model
    model = build_prototype_model(input_shape)
    print(model.summary())

    # Set up callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_auc', patience=10, mode='max', restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        "Prototype_best.keras", monitor='val_auc', mode='max',
        save_best_only=True, verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Evaluate the model on the test set
    test_metrics = model.evaluate(X_test, y_test, verbose=0)
    print("\nTest Metrics:")
    
    # Explicitly print test metrics with their names
    for metric_name, value in zip(model.metrics_names, test_metrics):
        print(f"{metric_name.capitalize()}: {value:.4f}")

    # Generate ROC curve data
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    print(f"\nTest ROC AUC: {roc_auc:.4f}")

    # Plot training history and ROC curve
    plot_results(history, fpr, tpr, roc_auc)

    # Explicitly print test metrics
    test_loss = test_metrics[0]
    test_accuracy = test_metrics[1]
    test_auc = test_metrics[2]
    test_precision = test_metrics[3]

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")

def plot_results(history, fpr, tpr, roc_auc):
    plt.figure(figsize=(18, 5))

    # Plot ROC Curve
    plt.subplot(1, 3, 1)
    plt.plot(tpr, fpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.semilogy()
    plt.grid(True)
    plt.legend()

    # Plot AUC History (if available)
    if 'auc' in history.history and 'val_auc' in history.history:
        plt.subplot(1, 3, 2)
        plt.plot(history.history['auc'], '--', label='Train AUC')
        plt.plot(history.history['val_auc'], '-', label='Val AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC Training History')
        plt.grid(True)
        plt.legend()

    # Plot Loss History
    plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'], '--', label='Train Loss')
    plt.plot(history.history['val_loss'], '-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Training History')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    train_and_evaluate_prototype()
