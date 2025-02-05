import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
# Set this to the path of your Parquet file (optimized or raw)
PARQUET_PATH = r"C:\Users\Dev\OneDrive\Desktop\hbb_production_parquet\ntuple_merged_10_optimized.parquet"

BATCH_SIZE = 1024
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# Original features from the reference model
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

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
def load_and_preprocess_data():
    # Load the DataFrame from the Parquet file
    df = pd.read_parquet(PARQUET_PATH)
    
    # If the mass/pt columns exist, apply the cuts
    required_cols = ['fj_sdmass', 'fj_pt']
    if all(col in df.columns for col in required_cols):
        mask = (df['fj_sdmass'] > 40) & (df['fj_sdmass'] < 200) & (df['fj_pt'] > 300) & (df['fj_pt'] < 2000)
        df = df[mask].copy()
    
    # (Optional) If the 'label' column does not exist, create it.
    if 'label' not in df.columns:
        # Compute composite labels if needed (update these lines as required)
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
    
    # Normalize features using training statistics
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std == 0] = 1e-8  # Avoid division by zero
    
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    
    return X_train, X_test, y_train, y_test, train_mean, train_std

# ============================================================================
# Model Definitions
# ============================================================================
def build_reference_model(input_shape):
    """Original reference model architecture."""
    # Ensure input_shape is a tuple
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', metrics.AUC(name='auc')]
    )
    return model

def build_prototype_model(input_shape):
    """Prototype model with improved architecture."""
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
        metrics=['accuracy', metrics.AUC(name='auc'),
                 metrics.Precision(name='precision'),
                 metrics.Recall(name='recall')]
    )
    return model

# ============================================================================
# Training and Evaluation
# ============================================================================
def train_and_evaluate(models_to_train):
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data()
    results = {}
    
    for model_name in models_to_train:
        print(f"\nTraining {model_name}...")
        
        # Pass input shape as a tuple
        input_shape = (X_train.shape[1],)
        if model_name == "Reference":
            model = build_reference_model(input_shape)
            monitor_metric = 'val_loss'
            mode = 'auto'
        else:
            model = build_prototype_model(input_shape)
            monitor_metric = 'val_auc'
            mode = 'max'
        
        cb_list = [
            callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=10,
                mode=mode,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                f"{model_name}_best.keras",
                save_best_only=True,
                monitor=monitor_metric,
                mode=mode
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=cb_list,
            verbose=1
        )
        
        test_results = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred[:, 1])
        
        results[model_name] = {
            'history': history.history,
            'test_metrics': test_results,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc(fpr, tpr)
        }
    
    return results

# ============================================================================
# Visualization
# ============================================================================
def plot_comparison(results):
    plt.figure(figsize=(18, 6))
    
    # ROC Curve Comparison
    plt.subplot(1, 3, 1)
    for model_name, data in results.items():
        plt.plot(data['tpr'], data['fpr'], label=f'{model_name} (AUC = {data["auc"]:.3f})')
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.title('ROC Curve Comparison (Log Scale)')
    plt.semilogy()
    plt.ylim(0.001, 1)
    plt.xlim(0, 1)
    plt.grid(True)
    plt.legend()
    
    # AUC History Comparison
    plt.subplot(1, 3, 2)
    for model_name, data in results.items():
        if 'auc' in data['history'] and 'val_auc' in data['history']:
            plt.plot(data['history']['auc'], '--', label=f'{model_name} Train AUC')
            plt.plot(data['history']['val_auc'], '-', label=f'{model_name} Val AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('AUC Training History')
    plt.grid(True)
    plt.legend()
    
    # Loss History Comparison
    plt.subplot(1, 3, 3)
    for model_name, data in results.items():
        plt.plot(data['history']['loss'], '--', label=f'{model_name} Train Loss')
        plt.plot(data['history']['val_loss'], '-', label=f'{model_name} Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Training History')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    results = train_and_evaluate(["Reference", "Prototype"])
    
    print("\nModel Comparison:")
    for model_name, data in results.items():
        print(f"\n{model_name} Model:")
        print(f"- Test Loss: {data['test_metrics'][0]:.4f}")
        print(f"- Accuracy: {data['test_metrics'][1]:.4f}")
        print(f"- AUC: {data['auc']:.4f}")
        # For the prototype, additional metrics might be available:
        if len(data['test_metrics']) > 3:
            print(f"- Precision: {data['test_metrics'][3]:.4f}")
            print(f"- Recall: {data['test_metrics'][4]:.4f}")
    
    plot_comparison(results)
