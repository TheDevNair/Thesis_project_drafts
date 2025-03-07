import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from matplotlib.colors import LinearSegmentedColormap

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Create custom color palette
signal_color = "#3498db"  # Blue
background_color = "#e74c3c"  # Red
threshold_color = "#2ecc71"  # Green

# Create output directory for plots
os.makedirs('feature_distributions', exist_ok=True)

# Feature columns as specified in your original code
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

def load_data():
    """Load and prepare the dataset for feature distribution analysis"""
    print("Loading dataset...")
    data_path = r"C:\Users\Dev\OneDrive\Desktop\hbb_production_parquet\ntuple_merged_10_optimized.parquet"
    df = pd.read_parquet(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values detected:")
        print(missing_values[missing_values > 0])
        
        # Replace infinite values with NaN and fill with column medians
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill missing values with column medians
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
    
    # Extract features and labels
    X = df[FEATURE_COLUMNS].copy()
    y = df['label'].values
    
    return X, y

def load_model_and_scaler():
    """Load the trained model and feature scaler"""
    print("Loading model and scaler...")
    
    try:
        # Try to load the final model first
        model_path = "final_improved_hbb_tagger.keras"
        if not os.path.exists(model_path):
            # If not found, try the checkpoint model
            model_path = "improved_hbb_tagger.keras"
        
        model = load_model(model_path, compile=False)
        print(f"Model loaded from: {model_path}")
        
        # Load the scaler
        scaler_path = "hbb_tagger_scaler.joblib"
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from: {scaler_path}")
        
        return model, scaler
    
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

def apply_plot_style(ax, title, xlabel, ylabel):
    """Apply consistent styling to plots"""
    # Set title with custom styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Set axis labels
    ax.set_xlabel(xlabel, fontsize=12, fontweight='medium', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='medium', labelpad=10)
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#444444')
    
    # Improve tick appearance
    ax.tick_params(width=0.8, colors='#444444', labelsize=10)
    
    # Add a subtle background color
    ax.set_facecolor('#f9f9f9')

def plot_feature_distributions(X, y, feature_names, predictions, threshold=0.5):
    """
    Plot normalized distributions of features for signal and background samples.
    
    Parameters:
    X (DataFrame): Features data
    y (array): True labels
    feature_names (list): Names of features to plot
    predictions (array): Model predictions (probabilities)
    threshold (float): Classification threshold
    """
    print("Generating feature distribution plots...")
    
    # Create predicted labels using the threshold
    y_pred = (predictions >= threshold).astype(int)
    
    # Plot distributions for each feature
    for i, feature in enumerate(feature_names):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        
        # Get the feature values for true signal/background
        sig_values = X.loc[y == 1, feature]
        bkg_values = X.loc[y == 0, feature]
        
        # Calculate range for x-axis that covers both distributions (with some margin)
        min_val = min(sig_values.min(), bkg_values.min())
        max_val = max(sig_values.max(), bkg_values.max())
        # Add 10% padding on each side
        range_width = max_val - min_val
        x_min = min_val - 0.1 * range_width
        x_max = max_val + 0.1 * range_width
        
        # Create bins ensuring both distributions use the same bins
        bins = np.linspace(x_min, x_max, 50)
        
        # Plot histograms with density normalization (normalized to unity)
        ax.hist(sig_values, bins=bins, alpha=0.7, density=True, 
               label='Signal (Hbb)', color=signal_color, 
               histtype='stepfilled', linewidth=1.5, edgecolor='black')
        ax.hist(bkg_values, bins=bins, alpha=0.5, density=True, 
               label='Background', color=background_color, 
               histtype='stepfilled', linewidth=1.5, edgecolor='black')
        
        # Add statistics to the plot
        sig_mean = sig_values.mean()
        sig_std = sig_values.std()
        bkg_mean = bkg_values.mean()
        bkg_std = bkg_values.std()
        
        # Calculate separation power (a simple metric for feature discriminating power)
        # Using the formula: |μ1 - μ2| / √(σ1² + σ2²)
        separation = abs(sig_mean - bkg_mean) / np.sqrt(sig_std**2 + bkg_std**2)
        
        # Add text box with statistics
        stats_text = (f"Signal: μ={sig_mean:.3f}, σ={sig_std:.3f}\n"
                     f"Background: μ={bkg_mean:.3f}, σ={bkg_std:.3f}\n"
                     f"Separation: {separation:.3f}")
        
        # Create styled text box
        props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                    alpha=0.8, edgecolor='gray', linewidth=0.5)
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', 
               bbox=props, family='monospace')
        
        # Apply consistent styling
        apply_plot_style(ax, f'Distribution of {feature}', feature, 'Normalized Frequency')
        
        # Customize legend
        ax.legend(loc='upper right', fontsize=10, frameon=True, 
                 fancybox=True, framealpha=0.8, edgecolor='gray')
        
        # Add a watermark
        fig.text(0.99, 0.01, 'HBB Tagger Analysis', 
                fontsize=8, color='gray', ha='right', 
                style='italic', alpha=0.7)
        
        # Save the figure with tight layout
        plt.tight_layout()
        plt.savefig(f'feature_distributions/{feature.replace("/", "_")}_distribution.png')
        plt.close()
        
        # Print progress every 5 features
        if (i + 1) % 5 == 0 or i == len(feature_names) - 1:
            print(f"Processed {i + 1}/{len(feature_names)} features")

def plot_correlations(X, y):
    """Plot correlation matrices for signal and background separately"""
    print("Generating correlation matrices...")
    
    # Split data by class
    X_sig = X[y == 1]
    X_bkg = X[y == 0]
    
    # Create correlation matrices
    corr_sig = X_sig.corr()
    corr_bkg = X_bkg.corr()
    
    # Create custom diverging colormap with better visual appeal
    custom_cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Function to create correlation heatmap with improved styling
    def plot_correlation_matrix(corr_data, title, filename):
        plt.figure(figsize=(14, 12), dpi=150)
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        # Plot correlation heatmap
        g = sns.heatmap(
            corr_data, 
            mask=mask, 
            cmap=custom_cmap, 
            center=0, 
            linewidths=0.01,
            cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
            annot=False,
            square=True, 
            vmin=-1, 
            vmax=1
        )
        
        # Improve heatmap appearance
        cbar = g.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        
        # Set title
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add decorative elements
        plt.tight_layout()
        
        # Add footer with timestamp
        plt.figtext(0.01, 0.01, "HBB Tagger Analysis", 
                   fontsize=8, color='gray', ha='left', style='italic')
        
        # Save figure with high quality
        plt.savefig(f'feature_distributions/{filename}', bbox_inches='tight')
        plt.close()
    
    # Plot the three correlation matrices
    plot_correlation_matrix(
        corr_sig, 
        'Feature Correlation Matrix - Signal Events (Hbb)', 
        'signal_correlation_matrix.png'
    )
    
    plot_correlation_matrix(
        corr_bkg, 
        'Feature Correlation Matrix - Background Events', 
        'background_correlation_matrix.png'
    )
    
    plot_correlation_matrix(
        corr_sig - corr_bkg, 
        'Difference in Correlation (Signal - Background)', 
        'correlation_difference.png'
    )

def plot_discriminator_distribution(y_true, y_pred_prob, threshold=0.5):
    """Plot the distribution of model outputs for signal and background"""
    print("Generating discriminator distribution plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    # Separate predictions by true class
    sig_scores = y_pred_prob[y_true == 1]
    bkg_scores = y_pred_prob[y_true == 0]
    
    # Create bins and custom styling for histograms
    bins = np.linspace(0, 1, 50)
    
    # Draw histograms with improved styling
    ax.hist(sig_scores, bins=bins, alpha=0.7, density=True, 
           label='Signal (Hbb)', color=signal_color, 
           histtype='stepfilled', linewidth=1.5, edgecolor='black')
    ax.hist(bkg_scores, bins=bins, alpha=0.5, density=True, 
           label='Background', color=background_color, 
           histtype='stepfilled', linewidth=1.5, edgecolor='black')
    
    # Add a vertical line at the threshold with improved styling
    ax.axvline(x=threshold, color=threshold_color, linestyle='-', 
              linewidth=2.5, label=f'Threshold: {threshold:.2f}', alpha=0.8)
    
    # Calculate and display signal efficiency and background rejection at threshold
    sig_eff = np.sum(sig_scores >= threshold) / len(sig_scores)
    bkg_rej = np.sum(bkg_scores < threshold) / len(bkg_scores)
    
    # Add metrics in a styled text box
    metric_text = (f"Signal efficiency: {sig_eff:.2%}\n"
                  f"Background rejection: {bkg_rej:.2%}\n"
                  f"S/√B improvement: {sig_eff/np.sqrt(1-bkg_rej):.2f}×")
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                alpha=0.8, edgecolor='gray', linewidth=0.5)
    
    ax.text(0.03, 0.97, metric_text, transform=ax.transAxes, 
           fontsize=12, verticalalignment='top', 
           bbox=props, family='sans-serif', fontweight='medium')
    
    # Apply consistent styling
    apply_plot_style(ax, 'DNN Output Score Distribution', 
                    'DNN Output Score', 'Normalized Frequency')
    
    # Add shaded regions to highlight signal and background regions
    ax.axvspan(threshold, 1, alpha=0.1, color=signal_color)
    ax.axvspan(0, threshold, alpha=0.1, color=background_color)
    # Customize legend
    ax.legend(loc='upper center', fontsize=12, frameon=True, 
             fancybox=True, framealpha=0.8, edgecolor='gray', 
             ncol=3, bbox_to_anchor=(0.5, -0.05))
    
    # Add annotations for signal and background regions
    ax.annotate('Background Region', xy=(0.2, 0.15), xycoords='axes fraction',
               fontsize=10, ha='center', color=background_color)
    ax.annotate('Signal Region', xy=(0.8, 0.15), xycoords='axes fraction',
               fontsize=10, ha='center', color=signal_color)
    
    # Add model performance metrics at the bottom
    roc_auc = calculate_roc_auc(y_true, y_pred_prob)
    plt.figtext(0.5, 0.01, f"Model AUC: {roc_auc:.4f}", 
               ha='center', fontsize=10, fontweight='bold')
    
    # Save the figure with high quality
    plt.tight_layout()
    plt.savefig('feature_distributions/discriminator_distribution.png', 
               bbox_inches='tight')
    plt.close()

def calculate_roc_auc(y_true, y_pred_prob):
    """Calculate ROC AUC score"""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred_prob)

def plot_roc_curve(y_true, y_pred_prob):
    """Plot ROC curve with enhanced styling"""
    print("Generating ROC curve plot...")
    
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Plot ROC curve with gradient color
    points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a colormap for the line
    norm = plt.Normalize(0, 1)
    lc = plt.matplotlib.collections.LineCollection(
        segments, cmap='viridis', norm=norm)
    lc.set_array(np.linspace(0, 1, len(fpr)))
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    
    # Add colorbar
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Threshold Value', fontsize=10)
    
    # Add diagonal line for random classifier
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, 
           label='Random Classifier')
    
    # Mark the best threshold point
    # Find the threshold that maximizes tpr - fpr (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
           label=f'Optimal threshold: {optimal_threshold:.2f}')
    
    # Apply consistent styling
    apply_plot_style(ax, f'ROC Curve (AUC = {roc_auc:.4f})', 
                    'False Positive Rate', 'True Positive Rate')
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add annotations for different operating points
    # Low FPR point
    low_fpr_idx = np.argmin(np.abs(fpr - 0.1))
    ax.annotate(f'TPR={tpr[low_fpr_idx]:.2f} @ FPR=0.1', 
               xy=(fpr[low_fpr_idx], tpr[low_fpr_idx]),
               xytext=(fpr[low_fpr_idx]+0.1, tpr[low_fpr_idx]-0.1),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # High TPR point
    high_tpr_idx = np.argmin(np.abs(tpr - 0.9))
    ax.annotate(f'FPR={fpr[high_tpr_idx]:.2f} @ TPR=0.9', 
               xy=(fpr[high_tpr_idx], tpr[high_tpr_idx]),
               xytext=(fpr[high_tpr_idx]-0.2, tpr[high_tpr_idx]-0.1),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add legend with styling
    ax.legend(loc='lower right', fontsize=10, frameon=True, 
             fancybox=True, framealpha=0.8, edgecolor='gray')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('feature_distributions/roc_curve.png', bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_prob):
    """Plot Precision-Recall curve with enhanced styling"""
    print("Generating Precision-Recall curve plot...")
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Plot PR curve with gradient color
    points = np.array([recall, precision]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a colormap for the line
    norm = plt.Normalize(0, 1)
    lc = plt.matplotlib.collections.LineCollection(
        segments, cmap='plasma', norm=norm)
    lc.set_array(np.linspace(0, 1, len(recall)))
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    
    # Add colorbar
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Threshold Value', fontsize=10)
    
    # Add baseline for random classifier
    ax.axhline(y=np.sum(y_true) / len(y_true), color='k', linestyle='--', 
              alpha=0.7, label='Random Classifier')
    
    # Mark the F1 optimal point
    # Calculate F1 score at each threshold
    f1_scores = []
    for i in range(len(precision)-1):
        if precision[i] + recall[i] > 0:  # Avoid division by zero
            f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            f1_scores.append(f1)
        else:
            f1_scores.append(0)
    
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    
    ax.plot(recall[best_f1_idx], precision[best_f1_idx], 'ro', markersize=8,
           label=f'Best F1: {best_f1:.2f}')
    
    # Apply consistent styling
    apply_plot_style(ax, f'Precision-Recall Curve (AP = {avg_precision:.4f})', 
                    'Recall', 'Precision')
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add annotations for specific operating points
    # High precision point
    high_prec_idx = np.argmin(np.abs(precision[:-1] - 0.9))
    ax.annotate(f'Recall={recall[high_prec_idx]:.2f} @ Precision=0.9', 
               xy=(recall[high_prec_idx], precision[high_prec_idx]),
               xytext=(recall[high_prec_idx]-0.2, precision[high_prec_idx]-0.1),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add legend with styling
    ax.legend(loc='upper right', fontsize=10, frameon=True, 
             fancybox=True, framealpha=0.8, edgecolor='gray')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('feature_distributions/precision_recall_curve.png', 
               bbox_inches='tight')
    plt.close()

def plot_feature_importance(X, y, feature_names):
    """Plot feature importance using a gradient boosting classifier"""
    print("Generating feature importance plot...")
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    
    # Create and train a simple gradient boosting model
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X, y)
    
    # Get feature importance
    importance = gb.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    # Prepare data for plotting
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    # Plot feature importance
    bars = ax.barh(range(len(sorted_features)), sorted_importance, 
                  align='center', color=plt.cm.viridis(np.linspace(0, 1, len(sorted_features))))
    
    # Add value labels to the bars
    for i, v in enumerate(sorted_importance):
        ax.text(v + 0.001, i, f'{v:.3f}', 
               va='center', fontsize=8, color='black')
    
    # Apply consistent styling
    apply_plot_style(ax, 'Feature Importance (Gradient Boosting)', 
                    'Importance', 'Features')
    
    # Set y-axis ticks and labels
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    
    # Adjust layout to ensure all feature names are visible
    plt.tight_layout()
    plt.savefig('feature_distributions/feature_importance.png', 
               bbox_inches='tight')
    plt.close()
    
    # Also calculate permutation importance
    # This is more reliable but slower
    perm_importance = permutation_importance(gb, X, y, n_repeats=10, 
                                           random_state=42)
    
    # Sort features by mean importance
    perm_indices = np.argsort(perm_importance.importances_mean)[::-1]
    perm_features = [feature_names[i] for i in perm_indices]
    perm_importance_mean = perm_importance.importances_mean[perm_indices]
    perm_importance_std = perm_importance.importances_std[perm_indices]
    
    # Create figure for permutation importance
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    # Plot permutation importance with error bars
    bars = ax.barh(range(len(perm_features)), perm_importance_mean, 
                  align='center', xerr=perm_importance_std,
                  color=plt.cm.plasma(np.linspace(0, 1, len(perm_features))))
    
    # Add value labels to the bars
    for i, v in enumerate(perm_importance_mean):
        ax.text(v + 0.001, i, f'{v:.3f}', 
               va='center', fontsize=8, color='black')
    
    # Apply consistent styling
    apply_plot_style(ax, 'Permutation Feature Importance', 
                    'Importance', 'Features')
    
    # Set y-axis ticks and labels
    ax.set_yticks(range(len(perm_features)))
    ax.set_yticklabels(perm_features)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('feature_distributions/permutation_importance.png', 
               bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the feature distribution analysis"""
    print("Starting feature distribution analysis...")
    
    # Load data
    X, y = load_data()
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        print("Cannot proceed without model and scaler. Exiting.")
        return
    
    # Apply the scaler to features
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    # Get model predictions
    print("Generating model predictions...")
    predictions = model.predict(X_scaled.values, batch_size=1024, verbose=1).flatten()
    
    # Try to load the optimal threshold from a file, default to 0.5 if not found
    try:
        # You can save this in your main script to a simple text file
        with open('optimal_threshold.txt', 'r') as f:
            optimal_threshold = float(f.read().strip())
        print(f"Loaded optimal threshold: {optimal_threshold}")
    except:
        optimal_threshold = 0.5
        print(f"Using default threshold: {optimal_threshold}")
    
    # Plot feature distributions
    plot_feature_distributions(X, y, X.columns, predictions, optimal_threshold)
    
    # Plot correlation matrices
    plot_correlations(X, y)
    
    # Plot discriminator distribution
    plot_discriminator_distribution(y, predictions, optimal_threshold)
    
    # Plot ROC curve
    plot_roc_curve(y, predictions)
    
    # Plot precision-recall curve
    plot_precision_recall_curve(y, predictions)
    
    # Plot feature importance
    plot_feature_importance(X, y, X.columns)
    
    print("\nAnalysis complete! All plots saved to 'feature_distributions' directory.")

if __name__ == "__main__":
    main()