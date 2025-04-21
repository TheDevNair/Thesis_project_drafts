import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks, metrics, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
PLOT_DIR = "plots"
MODEL_NAME = "hbb_tagger.keras"
SCALER_NAME = 'hbb_tagger_scaler.joblib'
OPTIMAL_THRESHOLD_FILE = 'hbb_tagger_optimal_threshold.txt'
TEST_PREDICTIONS_CSV = 'hbb_tagger_test_predictions.csv'

BATCH_SIZE = 1024
EPOCHS = 100
USE_FOCAL_LOSS = True # Set to False to use standard binary crossentropy

# Define the input features to be used
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
    'fj_z_ratio', 'fj_tau21'
]

# --- Directory Setup ---
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    print(f"Created directory: {PLOT_DIR}")

# --- Focal Loss Definition ---
# Custom loss function to handle class imbalance
@tf.keras.utils.register_keras_serializable()
def focal_loss(alpha=0.5, gamma=2.0):
    """
    Focal Loss function.
    Args:
        alpha (float): Weighting factor for positive class.
        gamma (float): Focusing parameter.
    Returns:
        Callable: Loss function.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        # Calculate cross-entropy
        cross_entropy = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        # Calculate focal loss weight
        weight = y_true * tf.math.pow(1 - y_pred, gamma) * alpha + (1 - y_true) * tf.math.pow(y_pred, gamma) * (1 - alpha)
        # Compute focal loss
        loss = weight * cross_entropy
        return tf.reduce_mean(loss)
    return loss_fn

# --- Data Handling ---
def load_and_preprocess_data():
    """Loads data from Parquet files, preprocesses, scales, and splits it."""
    # *** ADJUST FILE PATHS AS NEEDED ***
    file_list = [
        r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_10_optimized_filtered6.parquet",
        r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_11_optimized_filtered6.parquet",
        r"C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_12_optimized_filtered6.parquet",
    ]
    print("Loading data from:")
    for file in file_list: print(f"  {file}")

    try:
        # Read multiple parquet files and concatenate them
        dfs = [pd.read_parquet(file) for file in file_list]
        df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        print(f"Error loading files: {e}")
        raise

    print(f"Combined dataset shape: {df.shape}")

    # Apply kinematic cuts if columns exist
    if {'fj_sdmass', 'fj_pt'}.issubset(df.columns):
        print("Applying cuts (fj_sdmass > 40 & < 200, fj_pt > 300 & < 2000)...")
        mask = (df['fj_sdmass'] > 40) & (df['fj_sdmass'] < 200) & (df['fj_pt'] > 300) & (df['fj_pt'] < 2000)
        df = df[mask].copy()
        print(f"Shape after cuts: {df.shape}")
    else:
        print("Warning: Cuts skipped (fj_sdmass or fj_pt columns missing).")

    if 'label' not in df.columns: raise ValueError("'label' column missing.")
    if df.empty: raise ValueError("DataFrame empty after cuts/loading.")

    print(f"Class distribution:\n{df['label'].value_counts()}")
    print(f"Class proportions:\n{df['label'].value_counts() / len(df)}")

    # Check for missing values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print("\nMissing values per column (Before handling):")
        print(missing_values)

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = df[FEATURE_COLUMNS].copy()
    y = df['label'].values.astype(np.int32)

    # Impute NaN values with the median of the respective column
    if X.isnull().any().any():
        print("Imputing NaN with column median...")
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        print("NaN imputation finished.")
    else:
        print("No NaN values found to impute.")

    # Plot and save feature correlation matrix
    print("Plotting correlation matrix...")
    correlation = X.corr()
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(correlation, dtype=bool)) # Mask for upper triangle
    sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    corr_plot_path = os.path.join(PLOT_DIR, 'correlation_matrix.png')
    plt.savefig(corr_plot_path)
    plt.close()
    print(f"Correlation plot saved: {corr_plot_path}")

    # Split data into training, validation, and test sets (64%, 16%, 20%)
    print("Splitting data (64% train, 16% validation, 20% test)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X.values, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42 # 0.2 * 0.8 = 0.16
    )
    if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Data split resulted in an empty set.")
    print(f" Train set: {X_train.shape[0]}, Validation set: {X_val.shape[0]}, Test set: {X_test.shape[0]}")

    # Scale features using StandardScaler
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler object
    print(f"Saving scaler to {SCALER_NAME}...")
    joblib.dump(scaler, SCALER_NAME)
    print("Scaler saved.")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X.columns

# --- Model Architecture ---
def build_improved_model(input_shape):
    """Builds the Keras DNN model."""
    print("Building Keras Model...")
    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs) # Input normalization

    # First block
    x1 = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), activation=tf.keras.activations.silu)(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.2)(x1)

    # Second block
    x2 = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), activation=tf.keras.activations.silu)(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)

    # Residual connection
    res = layers.Add()([x1, x2])
    res = layers.Activation(tf.keras.activations.silu)(res)

    # Third block
    x3 = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4), activation=tf.keras.activations.silu)(res)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Dropout(0.2)(x3)

    # Fourth block
    x4 = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4), activation=tf.keras.activations.silu)(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Dropout(0.1)(x4)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x4)
    return models.Model(inputs=inputs, outputs=outputs)

# --- Threshold Tuning ---
def tune_threshold_on_validation(y_val_true, y_val_pred_prob):
    """Finds the optimal classification threshold based on max F1 score on validation set."""
    print("\n--- Tuning Threshold on Validation Set (Max F1-Score) ---")
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = [f1_score(y_val_true, (y_val_pred_prob >= t).astype(int), zero_division=0) for t in thresholds]

    if not f1_scores or np.all(np.isnan(f1_scores)):
        print("Warning: F1 computation failed during threshold tuning. Returning default 0.5.")
        return 0.5, 0.0 # Return default F1 score as well

    best_f1_idx = np.nanargmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    print(f" Best Validation F1: {best_f1:.4f} found at Threshold: {best_threshold:.4f}")

    # Plot metrics vs threshold
    plt.figure(figsize=(10, 6))
    accuracies = [accuracy_score(y_val_true, (y_val_pred_prob >= t).astype(int)) for t in thresholds]
    precisions = [precision_score(y_val_true, (y_val_pred_prob >= t).astype(int), zero_division=0) for t in thresholds]
    recalls = [recall_score(y_val_true, (y_val_pred_prob >= t).astype(int), zero_division=0) for t in thresholds]
    plt.plot(thresholds, accuracies, label='Accuracy (Val)')
    plt.plot(thresholds, precisions, label='Precision (Val)')
    plt.plot(thresholds, recalls, label='Recall (Val)')
    plt.plot(thresholds, f1_scores, label='F1 Score (Val)', lw=2, c='red')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Thr (F1): {best_threshold:.3f}')
    plt.xlabel('Threshold'); plt.ylabel('Score'); plt.title('Metrics vs Threshold (Validation Set)')
    plt.legend(); plt.grid(True)
    thresh_plot_path = os.path.join(PLOT_DIR, 'threshold_tuning.png')
    plt.savefig(thresh_plot_path); plt.close()
    print(f" Threshold tuning plot saved: {thresh_plot_path}")

    return best_threshold, best_f1

# --- Plotting Utilities ---
def plot_discriminator_distribution(y_true, y_pred_prob, threshold):
    """Plots the distribution of the model output score for signal and background."""
    print(f"\n--- Plotting Discriminator Distribution (Test Set) ---")
    plt.figure(figsize=(10, 7))
    sig_scores = y_pred_prob[y_true == 1]
    bkg_scores = y_pred_prob[y_true == 0]

    if len(sig_scores) == 0 or len(bkg_scores) == 0:
        print("Warning: Cannot plot distribution - No samples for one class.")
        plt.close(); return

    bins = np.linspace(0, 1, 51)
    plt.hist(sig_scores, bins=bins, alpha=0.7, density=True, label=f'Signal (Hbb, N={len(sig_scores)})', color='dodgerblue', histtype='step', linewidth=1.5)
    plt.hist(bkg_scores, bins=bins, alpha=0.7, density=True, label=f'Background (N={len(bkg_scores)})', color='salmon', histtype='step', linewidth=1.5)
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5, label=f'Threshold: {threshold:.3f}')

    # Calculate metrics at the given threshold for plot text
    try:
        y_pred_class = (y_pred_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        sig_eff = tp / (tp + fn + 1e-9) # Recall
        bkg_rej = tn / (tn + fp + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        f1 = 2 * (precision * sig_eff) / (precision + sig_eff + 1e-9)

        text_box_props = dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
        plt.text(0.03, 0.97, f'Signal Eff (Recall): {sig_eff:.2%}', transform=plt.gca().transAxes, fontsize=10, va='top', bbox=text_box_props)
        plt.text(0.03, 0.90, f'Background Rej: {bkg_rej:.2%}', transform=plt.gca().transAxes, fontsize=10, va='top', bbox=text_box_props)
        plt.text(0.03, 0.83, f'Precision: {precision:.2%}', transform=plt.gca().transAxes, fontsize=10, va='top', bbox=text_box_props)
        plt.text(0.03, 0.76, f'F1 Score: {f1:.4f}', transform=plt.gca().transAxes, fontsize=10, va='top', bbox=text_box_props)
    except ValueError: # Handles case where confusion_matrix might fail
        print("Warning: Couldn't calculate metrics for plot text.")

    plt.xlabel('DNN Output Score'); plt.ylabel('Normalized Distribution')
    plt.title('Distribution of Discriminator Output (Test Set)'); plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.4); plt.legend(loc='upper center')
    plt.ylim(bottom=max(1e-5, plt.ylim()[0])) # Ensure y-axis doesn't go too close to zero for log scale
    plt.tight_layout()
    dist_plot_path = os.path.join(PLOT_DIR, 'discriminator_distribution.png')
    plt.savefig(dist_plot_path, dpi=300); plt.close()
    print(f"Discriminator distribution plot saved: {dist_plot_path}")

def plot_training_history(history):
    """Plots Loss, AUC, and F1 Score over training epochs."""
    print("Plotting training history (Loss, AUC, F1)...")
    # Calculate F1 scores from Precision and Recall for each epoch
    train_precision = np.array(history.history['precision'])
    train_recall = np.array(history.history['recall'])
    val_precision = np.array(history.history['val_precision'])
    val_recall = np.array(history.history['val_recall'])
    # Add small epsilon to prevent division by zero
    train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-7)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)

    plt.figure(figsize=(18, 5))
    # Loss
    plt.subplot(1, 3, 1); plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Over Epochs'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    # AUC
    plt.subplot(1, 3, 2); plt.plot(history.history['auc'], label='Train'); plt.plot(history.history['val_auc'], label='Validation')
    plt.title('AUC Over Epochs'); plt.xlabel('Epochs'); plt.ylabel('AUC'); plt.legend(); plt.grid(True)
    # F1 Score
    plt.subplot(1, 3, 3); plt.plot(train_f1, label='Train F1'); plt.plot(val_f1, label='Validation F1')
    plt.title('F1 Score Over Epochs'); plt.xlabel('Epochs'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    hist_plot_path = os.path.join(PLOT_DIR, 'training_history.png')
    plt.savefig(hist_plot_path); plt.close()
    print(f"Training history plot saved: {hist_plot_path}")

def plot_roc_curve(y_test, y_test_pred_prob, roc_auc_value):
     """Plots the Receiver Operating Characteristic (ROC) curve."""
     print("Generating ROC curve...")
     fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
     plt.figure(figsize=(10, 8)); plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_value:.4f})')
     plt.plot([0, 1], [0, 1], 'k--', lw=2); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
     plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (Test Set)')
     plt.legend(loc="lower right"); plt.grid(True)
     roc_plot_path = os.path.join(PLOT_DIR, 'roc_curve.png')
     plt.savefig(roc_plot_path); plt.close(); print(f"ROC curve saved: {roc_plot_path}")

def plot_pr_curve(y_test, y_test_pred_prob, pr_auc):
     """Plots the Precision-Recall (PR) curve."""
     print("Generating PR curve...")
     precision, recall, _ = precision_recall_curve(y_test, y_test_pred_prob)
     plt.figure(figsize=(10, 8)); plt.plot(recall, precision, lw=2, label=f'PR (AP = {pr_auc:.4f})')
     plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve (Test Set)')
     plt.legend(loc="lower left"); plt.grid(True); plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
     pr_plot_path = os.path.join(PLOT_DIR, 'precision_recall_curve.png')
     plt.savefig(pr_plot_path); plt.close(); print(f"PR curve saved: {pr_plot_path}")

def plot_confusion_matrix(y_true, y_pred_class, threshold_val, filename_suffix):
     """Plots the confusion matrix."""
     print(f"Generating CM (threshold={threshold_val:.3f})...")
     cm = confusion_matrix(y_true, y_pred_class)
     plt.figure(figsize=(8, 6))
     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
     plt.title(f'Confusion Matrix (Test Set, threshold={threshold_val:.3f})')
     plt.xlabel('Predicted'); plt.ylabel('True')
     cm_path = os.path.join(PLOT_DIR, f'confusion_matrix_{filename_suffix}.png')
     plt.savefig(cm_path); plt.close(); print(f"CM ({filename_suffix}) saved: {cm_path}")

# --- Training and Evaluation Workflow ---
def train_and_evaluate():
    """Main function to load data, build model, train, evaluate, and plot results."""
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_preprocess_data()
    except Exception as e:
        print(f"Failed during data loading/preprocessing: {e}")
        return None

    input_shape = (X_train.shape[1],)
    model = build_improved_model(input_shape)
    print(model.summary())

    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    loss_fn_instance = focal_loss(alpha=0.5, gamma=2.0) if USE_FOCAL_LOSS else 'binary_crossentropy'
    # Determine the name to use for custom loss when loading the model later
    compile_loss_name = 'loss_fn' if USE_FOCAL_LOSS else 'binary_crossentropy'

    model.compile(
        optimizer=optimizer,
        loss=loss_fn_instance,
        metrics=[
            'accuracy',
            metrics.AUC(name='auc'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )

    # Class weights are intentionally not used (`class_weight=None` in fit)

    print("Setting up callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)...")
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max', min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(MODEL_NAME, monitor='val_auc', mode='max', save_best_only=True, verbose=1) # Saves the best model automatically
    ]

    print(f"\n--- Starting Training (Epochs={EPOCHS}, Batch Size={BATCH_SIZE}) ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        class_weight=None, # Explicitly None as per user choice
        verbose=1
    )

    print("Training finished.")
    # EarlyStopping with restore_best_weights=True should mean 'model' holds the best weights.
    # Explicitly loading from the ModelCheckpoint file is safer.
    print(f"Loading best model weights saved to {MODEL_NAME} by ModelCheckpoint...")
    try:
        # Provide custom objects if focal loss was used during compile
        custom_objects = {compile_loss_name: loss_fn_instance} if USE_FOCAL_LOSS else None
        best_model = keras.models.load_model(MODEL_NAME, custom_objects=custom_objects)
        print(f"Successfully loaded best model from {MODEL_NAME}")
        model = best_model # Ensure 'model' variable refers to the best loaded model
    except Exception as e:
        print(f"*** Warning: Failed to explicitly load best model from {MODEL_NAME}. "
              f"Using model state from end of training (check EarlyStopping's restore_best_weights). Error: {e} ***")

    plot_training_history(history)

    # --- Post-Training Evaluation (using the best model) ---
    print("\nPredicting on Validation set for threshold tuning...")
    y_val_pred_prob = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).flatten()
    print("Predicting on Test set for final evaluation...")
    y_test_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()

    optimal_threshold, best_val_f1 = tune_threshold_on_validation(y_val, y_val_pred_prob)

    print("\n--- Evaluating Model on Test Set ---")
    y_test_pred_class_opt = (y_test_pred_prob >= optimal_threshold).astype(np.int32)
    y_test_pred_class_default = (y_test_pred_prob >= 0.5).astype(np.int32)

    # Calculate primary test metrics using sklearn
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob); roc_auc_value = auc(fpr, tpr)
    pr_auc = average_precision_score(y_test, y_test_pred_prob) # Average Precision is PR AUC
    test_accuracy_opt = accuracy_score(y_test, y_test_pred_class_opt)
    test_f1_opt = f1_score(y_test, y_test_pred_class_opt, zero_division=0)
    test_precision_opt = precision_score(y_test, y_test_pred_class_opt, zero_division=0)
    test_recall_opt = recall_score(y_test, y_test_pred_class_opt, zero_division=0)


    print("\nTest Set Performance Metrics:")
    print(f" ROC AUC: {roc_auc_value:.4f}")
    print(f" Average Precision (PR AUC): {pr_auc:.4f}")
    print(f" Optimal Threshold (from Validation Set based on F1): {optimal_threshold:.4f}")
    print(f" Accuracy (@ Optimal Thr): {test_accuracy_opt:.4f}")
    print(f" Precision (@ Optimal Thr): {test_precision_opt:.4f}")
    print(f" Recall (@ Optimal Thr): {test_recall_opt:.4f}")
    print(f" F1 Score (@ Optimal Thr): {test_f1_opt:.4f}")


    # --- Calculate Mistag Rate at Fixed Signal Efficiencies ---
    print("\nCalculating Mistag Rates at Fixed Signal Efficiencies (Test Set)...")
    signal_preds = y_test_pred_prob[y_test == 1]
    background_preds = y_test_pred_prob[y_test == 0]
    n_background = len(background_preds)
    n_signal = len(signal_preds)

    if n_signal > 0 and n_background > 0:
        # Threshold for 50% signal efficiency
        # Find the score where 50% of signal events are above it
        threshold_eff50 = np.percentile(signal_preds, 100 * (1 - 0.50))
        mistag_rate_eff50 = np.sum(background_preds >= threshold_eff50) / n_background
        # Avoid division by zero for rejection factor if mistag rate is zero
        rejection_50 = 1/mistag_rate_eff50 if mistag_rate_eff50 > 0 else np.inf
        print(f" Signal Efficiency: 50.0% -> Threshold: {threshold_eff50:.4f}, Mistag Rate: {mistag_rate_eff50:.4e} (Rejection: {rejection_50:.1f})")

        # Threshold for 70% signal efficiency
        # Find the score where 30% of signal events are below it (70% are above)
        threshold_eff70 = np.percentile(signal_preds, 100 * (1 - 0.70))
        mistag_rate_eff70 = np.sum(background_preds >= threshold_eff70) / n_background
        rejection_70 = 1/mistag_rate_eff70 if mistag_rate_eff70 > 0 else np.inf
        print(f" Signal Efficiency: 70.0% -> Threshold: {threshold_eff70:.4f}, Mistag Rate: {mistag_rate_eff70:.4e} (Rejection: {rejection_70:.1f})")

    elif n_signal == 0:
        print(" Cannot calculate efficiency thresholds: No signal events in test set.")
    else: # n_background == 0
        print(" Cannot calculate mistag rates: No background events in test set.")
    # --- End Mistag Rate Calculation ---


    # Generate Plots
    plot_roc_curve(y_test, y_test_pred_prob, roc_auc_value)
    plot_pr_curve(y_test, y_test_pred_prob, pr_auc)

    print("\nClassification Report (Test Set, threshold=0.5):")
    print(classification_report(y_test, y_test_pred_class_default, digits=4, zero_division=0))
    plot_confusion_matrix(y_test, y_test_pred_class_default, 0.5, "default_thr")

    print(f"\nClassification Report (Test Set, optimal threshold={optimal_threshold:.4f}):")
    print(classification_report(y_test, y_test_pred_class_opt, digits=4, zero_division=0))
    plot_confusion_matrix(y_test, y_test_pred_class_opt, optimal_threshold, "optimal_thr")

    # Save optimal threshold
    print(f"Saving optimal threshold {optimal_threshold:.4f} to {OPTIMAL_THRESHOLD_FILE}...")
    try:
        with open(OPTIMAL_THRESHOLD_FILE, 'w') as f: f.write(str(optimal_threshold))
        print(f"Optimal threshold saved.")
    except Exception as e: print(f"Error saving threshold: {e}")

    # Plot discriminator distribution on Test Set using the optimal threshold
    plot_discriminator_distribution(y_test, y_test_pred_prob, optimal_threshold)

    # Return the final model state (should be the best one) and results
    return model, roc_auc_value, optimal_threshold, feature_names, X_test, y_test, y_test_pred_prob


# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Starting Hbb Tagger Training & Evaluation ---")
    start_time = datetime.datetime.now()
    results = train_and_evaluate()

    if results:
        # Unpack necessary results
        model, roc_auc, optimal_threshold, feature_names, X_test, y_test, y_test_pred_prob = results

        # Model saving is handled by ModelCheckpoint callback during training.
        print(f"\nBest model during training was saved by ModelCheckpoint as: {MODEL_NAME}")

        # Save predictions
        print(f"Saving test predictions to {TEST_PREDICTIONS_CSV}...")
        results_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_prob': y_test_pred_prob,
            'predicted_default': (y_test_pred_prob >= 0.5).astype(np.int32),
            'predicted_optimal': (y_test_pred_prob >= optimal_threshold).astype(np.int32)
        })
        try:
            results_df.to_csv(TEST_PREDICTIONS_CSV, index=False)
            print(f"Predictions saved.")
        except Exception as e:
            print(f"Error saving predictions: {e}")


        print("\n--- Summary ---")
        print(f" Training complete!")
        print(f" Final Test ROC AUC: {roc_auc:.4f}")
        print(f" Optimal classification threshold (from validation set F1): {optimal_threshold:.4f}")
        print(f" Best model saved as: {MODEL_NAME}")
        print(f" Scaler saved as: {SCALER_NAME}")
        print(f" Optimal threshold saved to: {OPTIMAL_THRESHOLD_FILE}")
        print(f" Test set predictions saved to: {TEST_PREDICTIONS_CSV}")
        print(f" Plots saved in directory: {PLOT_DIR}")
        # Note: Mistag rates at fixed efficiencies are printed during evaluation.

    else:
        print("\n--- Training or evaluation failed ---")

    end_time = datetime.datetime.now()
    print(f"\nScript finished in {end_time - start_time}")
    print("====================================================")