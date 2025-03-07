import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model

# Set random seeds and plot style for reproducibility and consistent appearance
np.random.seed(42)
tf.random.set_seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Define custom colors for plots
signal_color = "#3498db"      # Blue for signal
background_color = "#e74c3c"  # Red for background
threshold_color = "#2ecc71"   # Green for threshold marker

# Create directory to save plots
os.makedirs('feature_distributions', exist_ok=True)

# List of features used in the analysis
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
    """Load and preprocess the dataset for analysis."""
    print("Loading dataset...")
    data_path = r"C:\Users\Dev\OneDrive\Desktop\hbb_production_parquet\ntuple_merged_10_optimized.parquet"
    df = pd.read_parquet(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Handle missing and infinite values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values detected:")
        print(missing_values[missing_values > 0])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
    
    X = df[FEATURE_COLUMNS].copy()
    y = df['label'].values
    return X, y

def load_model_and_scaler():
    """Load the trained model and feature scaler."""
    print("Loading model and scaler...")
    try:
        model_path = "final_improved_hbb_tagger.keras"
        if not os.path.exists(model_path):
            model_path = "improved_hbb_tagger.keras"
        model = load_model(model_path, compile=False)
        print(f"Model loaded from: {model_path}")
        
        scaler_path = "hbb_tagger_scaler.joblib"
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from: {scaler_path}")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

def apply_plot_style(ax, title, xlabel, ylabel):
    """Apply consistent styling to a plot axis with a thin black boundary."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='medium', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='medium', labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
    ax.tick_params(width=0.8, colors='black', labelsize=10)
    ax.set_facecolor('#f9f9f9')

def plot_feature_distributions(X, y, feature_names, predictions, threshold=0.5):
    """Plot enhanced histograms of each feature for signal and background."""
    print("Generating enhanced feature distribution plots...")
    y_pred = (predictions >= threshold).astype(int)
    
    # Enhanced color palette
    signal_color = "#3498db"      # Blue for signal
    background_color = "#e74c3c"  # Red for background
    threshold_color = "#2ecc71"   # Green for threshold marker
    
    for i, feature in enumerate(feature_names):
        # Create a figure with two subplots - histogram and KDE
        fig, (ax_hist, ax_kde) = plt.subplots(2, 1, figsize=(12, 10), dpi=150, 
                                              gridspec_kw={'height_ratios': [3, 1]})
        
        sig_values = X.loc[y == 1, feature]
        bkg_values = X.loc[y == 0, feature]
        
        min_val = min(sig_values.min(), bkg_values.min())
        max_val = max(sig_values.max(), bkg_values.max())
        range_width = max_val - min_val
        x_min = min_val - 0.1 * range_width
        x_max = max_val + 0.1 * range_width
        bins = np.linspace(x_min, x_max, 50)
        
        # Histogram plot with gradient fill
        n_sig, bins_sig, patches_sig = ax_hist.hist(
            sig_values, bins=bins, alpha=0.7, density=True, 
            label='Signal (Hbb)', color=signal_color, 
            histtype='stepfilled', linewidth=1.5, edgecolor='black'
        )
        
        n_bkg, bins_bkg, patches_bkg = ax_hist.hist(
            bkg_values, bins=bins, alpha=0.5, density=True, 
            label='Background', color=background_color, 
            histtype='stepfilled', linewidth=1.5, edgecolor='black'
        )
        
        # Add gradient effect to histograms
        for patch in patches_sig:
            patch.set_facecolor(plt.cm.Blues(0.7))
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)
            
        for patch in patches_bkg:
            patch.set_facecolor(plt.cm.Reds(0.7))
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)
        
        # Add KDE curves on the second subplot
        sns.kdeplot(sig_values, ax=ax_kde, color=signal_color, linewidth=3, 
                   label='Signal (Hbb)', fill=True, alpha=0.3)
        sns.kdeplot(bkg_values, ax=ax_kde, color=background_color, linewidth=3, 
                   label='Background', fill=True, alpha=0.3)
        
        # Calculate statistics
        sig_mean, sig_std = sig_values.mean(), sig_values.std()
        bkg_mean, bkg_std = bkg_values.mean(), bkg_values.std()
        separation = abs(sig_mean - bkg_mean) / np.sqrt(sig_std**2 + bkg_std**2)
        
        # Add vertical lines for means
        ax_hist.axvline(sig_mean, color=signal_color, linestyle='--', linewidth=2,
                      label=f'Signal Mean: {sig_mean:.3f}')
        ax_hist.axvline(bkg_mean, color=background_color, linestyle='--', linewidth=2,
                      label=f'Background Mean: {bkg_mean:.3f}')
        ax_kde.axvline(sig_mean, color=signal_color, linestyle='--', linewidth=2)
        ax_kde.axvline(bkg_mean, color=background_color, linestyle='--', linewidth=2)
        
        # Add statistical information
        stats_text = (f"Signal: μ={sig_mean:.3f}, σ={sig_std:.3f}\n"
                      f"Background: μ={bkg_mean:.3f}, σ={bkg_std:.3f}\n"
                      f"Separation: {separation:.3f}")
        
        # Create a fancy statistical box
        props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                     alpha=0.9, edgecolor='gray', linewidth=1.5)
        ax_hist.text(0.03, 0.97, stats_text, transform=ax_hist.transAxes, 
                   fontsize=12, verticalalignment='top', bbox=props, 
                   family='monospace', fontweight='bold')
        
        # Add ROC information if significant separation
        if separation > 0.5:
            from sklearn.metrics import roc_auc_score
            try:
                feature_auc = roc_auc_score(y, X[feature])
                auc_text = f"Feature AUC: {feature_auc:.3f}"
                ax_hist.text(0.97, 0.97, auc_text, transform=ax_hist.transAxes,
                           fontsize=12, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                                    alpha=0.9, edgecolor='goldenrod', linewidth=1.5),
                           family='monospace', fontweight='bold')
            except:
                pass
        
        # Style the histogram plot
        ax_hist.set_title(f'Distribution of {feature}', fontsize=16, fontweight='bold', pad=15)
        ax_hist.set_ylabel('Normalized Frequency', fontsize=14, fontweight='medium', labelpad=10)
        ax_hist.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        for spine in ax_hist.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        ax_hist.tick_params(width=1.2, colors='black', labelsize=12)
        ax_hist.set_facecolor('#f9f9f9')
        
        # Style the KDE plot
        ax_kde.set_xlabel(feature, fontsize=14, fontweight='medium', labelpad=10)
        ax_kde.set_ylabel('Density', fontsize=14, fontweight='medium', labelpad=10)
        ax_kde.grid(True, linestyle='--', alpha=0.7, color='lightgray')
        for spine in ax_kde.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        ax_kde.tick_params(width=1.2, colors='black', labelsize=12)
        ax_kde.set_facecolor('#f9f9f9')
        
        # Add shaded regions for significance
        if separation > 1.0:
            # Add shaded regions to highlight separation
            overlap_min = min(sig_mean - sig_std, bkg_mean - bkg_std)
            overlap_max = max(sig_mean + sig_std, bkg_mean + bkg_std)
            ax_hist.axvspan(overlap_min, overlap_max, alpha=0.2, color='purple', 
                          label='Overlap Region')
            ax_kde.axvspan(overlap_min, overlap_max, alpha=0.2, color='purple')
        
        # Add feature ranking annotation if separation is significant
        if separation > 0.8:
            rank_text = "High Discrimination Power"
        elif separation > 0.5:
            rank_text = "Medium Discrimination Power"
        else:
            rank_text = "Low Discrimination Power"
            
        ax_hist.text(0.5, 0.03, rank_text, transform=ax_hist.transAxes,
                   fontsize=12, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                            alpha=0.9, edgecolor='goldenrod', linewidth=1.5),
                   fontweight='bold')
        
        # Create a single legend for both plots
        handles_hist, labels_hist = ax_hist.get_legend_handles_labels()
        ax_hist.legend(handles_hist, labels_hist, loc='upper right', 
                     fontsize=12, frameon=True, fancybox=True, 
                     framealpha=0.8, edgecolor='gray',
                     shadow=True, ncol=2)
        ax_kde.legend().set_visible(False)  # Hide KDE legend to avoid duplication
        
        # Add watermark
        fig.text(0.99, 0.01, 'HBB Tagger Analysis', fontsize=10, 
                color='gray', ha='right', style='italic', alpha=0.7)
        
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.15)  # Reduce space between subplots
        
        # Add a suptitle with fancy styling
        plt.suptitle(f'Feature Analysis: {feature}', 
                     fontsize=18, fontweight='bold', 
                     y=0.98, color='#2c3e50', 
                     bbox=dict(boxstyle='round,pad=0.5', 
                              facecolor='#ecf0f1', 
                              edgecolor='#bdc3c7'))
        
        # Save with higher quality
        plt.savefig(f'feature_distributions/{feature.replace("/", "_")}_distribution.png', 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        if (i + 1) % 5 == 0 or i == len(feature_names) - 1:
            print(f"Processed {i + 1}/{len(feature_names)} features")

def plot_correlations(X, y):
    """Generate correlation heatmaps for signal and background."""
    print("Generating correlation matrices...")
    X_sig = X[y == 1]
    X_bkg = X[y == 0]
    corr_sig = X_sig.corr()
    corr_bkg = X_bkg.corr()
    
    def plot_correlation_matrix(corr_data, title, filename):
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(
            corr_data,
            mask=mask,
            annot=False,
            cmap='coolwarm',
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
        )
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.figtext(0.01, 0.01, "HBB Tagger Analysis", fontsize=8, color='gray', ha='left', style='italic')
        plt.savefig(f'feature_distributions/{filename}', bbox_inches='tight')
        plt.close()
    
    plot_correlation_matrix(corr_sig, 'Feature Correlation Matrix - Signal Events (Hbb)', 'signal_correlation_matrix.png')
    plot_correlation_matrix(corr_bkg, 'Feature Correlation Matrix - Background Events', 'background_correlation_matrix.png')
    plot_correlation_matrix(corr_sig - corr_bkg, 'Difference in Correlation (Signal - Background)', 'correlation_difference.png')
def plot_discriminator_distribution(y_true, y_pred_prob, threshold=0.5):
    """Plot model output score distributions for signal and background."""
    print("Generating discriminator distribution plot...")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    sig_scores = y_pred_prob[y_true == 1]
    bkg_scores = y_pred_prob[y_true == 0]
    bins = np.linspace(0, 1, 50)
    
    ax.hist(sig_scores, bins=bins, alpha=0.7, density=True, 
            label='Signal (Hbb)', color=signal_color, 
            histtype='stepfilled', linewidth=1.5, edgecolor='black')
    ax.hist(bkg_scores, bins=bins, alpha=0.5, density=True, 
            label='Background', color=background_color, 
            histtype='stepfilled', linewidth=1.5, edgecolor='black')
    ax.axvline(x=threshold, color=threshold_color, linestyle='-', linewidth=2.5, 
               label=f'Threshold: {threshold:.2f}', alpha=0.8)
    
    sig_eff = np.sum(sig_scores >= threshold) / len(sig_scores)
    bkg_rej = np.sum(bkg_scores < threshold) / len(bkg_scores)
    metric_text = (f"Signal efficiency: {sig_eff:.2%}\n"
                   f"Background rejection: {bkg_rej:.2%}\n"
                   f"S/√B improvement: {sig_eff/np.sqrt(1-bkg_rej):.2f}×")
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5)
    ax.text(0.03, 0.97, metric_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', 
            bbox=props, family='sans-serif', fontweight='medium')
    
    apply_plot_style(ax, 'DNN Output Score Distribution', 'DNN Output Score', 'Normalized Frequency')
    ax.axvspan(threshold, 1, alpha=0.1, color=signal_color)
    ax.axvspan(0, threshold, alpha=0.1, color=background_color)
    ax.legend(loc='upper center', fontsize=12, frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray', 
              ncol=3, bbox_to_anchor=(0.5, -0.05))
    ax.annotate('Background Region', xy=(0.2, 0.15), xycoords='axes fraction',
                fontsize=10, ha='center', color=background_color)
    ax.annotate('Signal Region', xy=(0.8, 0.15), xycoords='axes fraction',
                fontsize=10, ha='center', color=signal_color)
    
    roc_auc = calculate_roc_auc(y_true, y_pred_prob)
    plt.figtext(0.5, 0.01, f"Model AUC: {roc_auc:.4f}", ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_distributions/discriminator_distribution.png', bbox_inches='tight')
    plt.close()

def calculate_roc_auc(y_true, y_pred_prob):
    """Calculate the ROC AUC score."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred_prob)

def plot_roc_curve(y_true, y_pred_prob):
    """Generate and save the ROC curve plot."""
    print("Generating ROC curve plot...")
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(0, 1)
    lc = plt.matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(np.linspace(0, 1, len(fpr)))
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Threshold Value', fontsize=10)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Classifier')
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, label=f'Optimal threshold: {optimal_threshold:.2f}')
    
    apply_plot_style(ax, f'ROC Curve (AUC = {roc_auc:.4f})', 'False Positive Rate', 'True Positive Rate')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    low_fpr_idx = np.argmin(np.abs(fpr - 0.1))
    ax.annotate(f'TPR={tpr[low_fpr_idx]:.2f} @ FPR=0.1', 
                xy=(fpr[low_fpr_idx], tpr[low_fpr_idx]),
                xytext=(fpr[low_fpr_idx]+0.1, tpr[low_fpr_idx]-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    high_tpr_idx = np.argmin(np.abs(tpr - 0.9))
    ax.annotate(f'FPR={fpr[high_tpr_idx]:.2f} @ TPR=0.9', 
                xy=(fpr[high_tpr_idx], tpr[high_tpr_idx]),
                xytext=(fpr[high_tpr_idx]-0.2, tpr[high_tpr_idx]-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray')
    plt.tight_layout()
    plt.savefig('feature_distributions/roc_curve.png', bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_prob):
    """Generate and save the Precision-Recall curve plot."""
    print("Generating Precision-Recall curve plot...")
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    points = np.array([recall, precision]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(0, 1)
    lc = plt.matplotlib.collections.LineCollection(segments, cmap='plasma', norm=norm)
    lc.set_array(np.linspace(0, 1, len(recall)))
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Threshold Value', fontsize=10)
    
    ax.axhline(y=np.sum(y_true) / len(y_true), color='k', linestyle='--', alpha=0.7, label='Random Classifier')
    
    f1_scores = []
    for i in range(len(precision)-1):
        if precision[i] + recall[i] > 0:
            f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            f1_scores.append(f1)
        else:
            f1_scores.append(0)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    ax.plot(recall[best_f1_idx], precision[best_f1_idx], 'ro', markersize=8,
            label=f'Best F1: {best_f1:.2f}')
    
    apply_plot_style(ax, f'Precision-Recall Curve (AP = {avg_precision:.4f})', 'Recall', 'Precision')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    high_prec_idx = np.argmin(np.abs(precision[:-1] - 0.9))
    ax.annotate(f'Recall={recall[high_prec_idx]:.2f} @ Precision=0.9', 
                xy=(recall[high_prec_idx], precision[high_prec_idx]),
                xytext=(recall[high_prec_idx]-0.2, precision[high_prec_idx]-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray')
    plt.tight_layout()
    plt.savefig('feature_distributions/precision_recall_curve.png', bbox_inches='tight')
    plt.close()

def plot_feature_importance(X, y, feature_names):
    """Plot feature importance using a gradient boosting classifier and permutation importance."""
    print("Generating feature importance plots...")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X, y)
    importance = gb.feature_importances_
    indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    bars = ax.barh(range(len(sorted_features)), sorted_importance, 
                   align='center', color=plt.cm.viridis(np.linspace(0, 1, len(sorted_features))))
    for i, v in enumerate(sorted_importance):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8, color='black')
    apply_plot_style(ax, 'Feature Importance (Gradient Boosting)', 'Importance', 'Features')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    plt.tight_layout()
    plt.savefig('feature_distributions/feature_importance.png', bbox_inches='tight')
    plt.close()
    
    perm_importance = permutation_importance(gb, X, y, n_repeats=10, random_state=42)
    perm_indices = np.argsort(perm_importance.importances_mean)[::-1]
    perm_features = [feature_names[i] for i in perm_indices]
    perm_importance_mean = perm_importance.importances_mean[perm_indices]
    perm_importance_std = perm_importance.importances_std[perm_indices]
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    bars = ax.barh(range(len(perm_features)), perm_importance_mean, 
                   align='center', xerr=perm_importance_std,
                   color=plt.cm.plasma(np.linspace(0, 1, len(perm_features))))
    for i, v in enumerate(perm_importance_mean):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8, color='black')
    apply_plot_style(ax, 'Permutation Feature Importance', 'Importance', 'Features')
    ax.set_yticks(range(len(perm_features)))
    ax.set_yticklabels(perm_features)
    plt.tight_layout()
    plt.savefig('feature_distributions/permutation_importance.png', bbox_inches='tight')
    plt.close()

def main():
    """Run the full feature distribution analysis."""
    print("Starting feature distribution analysis...")
    X, y = load_data()
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        print("Model or scaler not found. Exiting.")
        return
    
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    print("Generating model predictions...")
    predictions = model.predict(X_scaled.values, batch_size=1024, verbose=1).flatten()
    
    try:
        with open('optimal_threshold.txt', 'r') as f:
            optimal_threshold = float(f.read().strip())
        print(f"Loaded optimal threshold: {optimal_threshold}")
    except:
        optimal_threshold = 0.5
        print(f"Using default threshold: {optimal_threshold}")
    
    plot_feature_distributions(X, y, X.columns, predictions, optimal_threshold)
    plot_correlations(X, y)
    plot_discriminator_distribution(y, predictions, optimal_threshold)
    plot_roc_curve(y, predictions)
    plot_precision_recall_curve(y, predictions)
    plot_feature_importance(X, y, X.columns)
    print("\nAnalysis complete! Plots saved to the 'feature_distributions' directory.")

if __name__ == "__main__":
    main()
