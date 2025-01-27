import numpy as np
import tensorflow as tf
import pandas as pd

# Paths
MODEL_PATH = "hbb_dnn_tagger.keras"
NORM_PARAMS_PATH = "hbb_dnn_tagger_norm.npz"
TEST_PARQUET_PATH = r"C:\Users\Dev\OneDrive\Desktop\hbb_production_parquet\ntuple_merged_12_optimized.parquet"

# Load trained model and normalization parameters
model = tf.keras.models.load_model(MODEL_PATH)
norm_params = np.load(NORM_PARAMS_PATH)
train_mean, train_std = norm_params["mean"], norm_params["std"]

# Load and preprocess new data
def load_test_data(filepath):
    """Loads and preprocesses test data."""
    df = pd.read_parquet(filepath)

    # Create labels (assuming 'fj_isBB' is the signal label)
    df['label'] = df['fj_isBB'].astype(np.float32)

    # Select the same features as in training
    feature_columns = [
        'fj_pt', 'fj_eta', 'fj_phi', 'fj_mass', 'fj_sdmass',
        'fj_tau21', 'fj_tau32', 'fj_nbHadrons', 'fj_ncHadrons',
        'pfDeepCSVJetTags_probb', 'pfDeepCSVJetTags_probbb',
        'pfCombinedInclusiveSecondaryVertexV2BJetTags'
    ]

    X = df[feature_columns].fillna(-999).values
    y = df['label'].values

    # Apply normalization using training dataset statistics
    X = (X - train_mean) / (train_std + 1e-8)

    return X, y

# Load test data
X_test, y_test = load_test_data(TEST_PARQUET_PATH)

# Evaluate model
test_results = model.evaluate(X_test, y_test)
print("\nTest Performance on New Data:")
print(f"Accuracy: {test_results[1]:.4f}")
print(f"AUC: {test_results[2]:.4f}")
print(f"Precision: {test_results[3]:.4f}")
print(f"Recall: {test_results[4]:.4f}")

# Predict labels
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)

# Display some sample predictions
print("\nSample Predictions:")
print(predicted_labels[:20].flatten())  # Show first 20 predicted labels
