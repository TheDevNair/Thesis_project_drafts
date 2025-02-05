import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf

# File path for the dataset and the trained model
PARQUET_PATH = r"C:\Users\Dev\OneDrive\Desktop\hbb_production_parquet\ntuple_merged_10_optimized.parquet"
MODEL_PATH = "Prototype_best.keras"

# Feature columns for the distributions
FEATURE_COLUMNS = [
    'fj_jetNTracks', 'fj_nSV', 'fj_tau0_trackEtaRel_0', 'fj_tau0_trackEtaRel_1', 
    'fj_tau0_trackEtaRel_2', 'fj_tau1_trackEtaRel_0', 'fj_tau1_trackEtaRel_1', 
    'fj_tau1_trackEtaRel_2', 'fj_tau_flightDistance2dSig_0', 'fj_tau_flightDistance2dSig_1',
    'fj_tau_vertexDeltaR_0', 'fj_tau_vertexEnergyRatio_0', 'fj_tau_vertexEnergyRatio_1',
    'fj_tau_vertexMass_0', 'fj_tau_vertexMass_1', 'fj_trackSip2dSigAboveBottom_0', 
    'fj_trackSip2dSigAboveBottom_1', 'fj_trackSip2dSigAboveCharm_0', 'fj_trackSipdSig_0',
    'fj_trackSipdSig_0_0', 'fj_trackSipdSig_0_1', 'fj_trackSipdSig_1', 'fj_trackSipdSig_1_0', 
    'fj_trackSipdSig_1_1', 'fj_trackSipdSig_2', 'fj_trackSipdSig_3', 'fj_z_ratio'
]

FEATURES_TO_PLOT = ['fj_jetNTracks', 'fj_tau0_trackEtaRel_0', 'fj_z_ratio']

# Load and preprocess data
def load_and_preprocess_for_classification(parquet_path):
    df = pd.read_parquet(parquet_path)
    df_features = df[FEATURE_COLUMNS].fillna(-999)
    X = df_features.values.astype(np.float32)
    train_mean = X.mean(axis=0)
    train_std = X.std(axis=0)
    train_std[train_std == 0] = 1e-8  # Avoid division by zero
    X = (X - train_mean) / train_std
    return X, df

# Load trained model
def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)

# Classify background and signal
def classify_background_signal(model, X):
    y_pred = model.predict(X)
    return np.argmax(y_pred, axis=1)

# Plot feature distributions
def plot_feature_distributions(df, y_pred_class):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {'background': (0, 0, 1, 0.2), 'signal': 'red'}

    for i, feature in enumerate(FEATURES_TO_PLOT):
        ax = axes[i]
        sns.histplot(df[feature][y_pred_class == 0], bins=50, label="Background", color=colors['background'], stat='count', ax=ax, kde=False, alpha=0.2)
        sns.histplot(df[feature][y_pred_class == 1], bins=50, label="Signal", color=colors['signal'], stat='count', ax=ax, kde=False)

        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{feature} Distribution', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Mean lines and labels
        bg_mean = df[feature][y_pred_class == 0].mean()
        sg_mean = df[feature][y_pred_class == 1].mean()
        ax.axvline(bg_mean, color=colors['background'], linestyle='--', label=f'BG mean ({bg_mean:.2f})')
        ax.axvline(sg_mean, color=colors['signal'], linestyle='--', label=f'SG mean ({sg_mean:.2f})')

        # Add legend to each axis
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    model = load_model()
    X, df = load_and_preprocess_for_classification(PARQUET_PATH)
    y_pred_class = classify_background_signal(model, X)

    # Print background and signal counts
    print(f"Total background events: {np.sum(y_pred_class == 0)}")
    print(f"Total signal events: {np.sum(y_pred_class == 1)}")

    # Plot feature distributions
    plot_feature_distributions(df, y_pred_class)
