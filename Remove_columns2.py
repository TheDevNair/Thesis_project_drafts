import pandas as pd
import os

# Load the original Parquet file
file_path = "C:/Users/Dev/OneDrive/Desktop/hbb_production_parquet/ntuple_merged_11.parquet"
df = pd.read_parquet(file_path)

# Define the list of required features (27 features)
features = [
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

# Create composite labels exactly like in your second code
df['isHbb'] = df['fj_isH'] * df['fj_isBB']
df['isQCD'] = df['fj_isQCD'] * df['sample_isQCD']

# Filter for mutually exclusive labels (exactly one truth)
df = df[(df['isHbb'] + df['isQCD']) == 1].copy()

# Create a binary label: 1 for Hbb and 0 for QCD
df['label'] = df['isHbb'].astype(int)

# Select only the required features and the label column
selected_columns = features + ['label']
df_filtered = df.loc[:, selected_columns].copy()

# Generate the optimized file path
file_dir, file_name = os.path.split(file_path)
file_name_optimized = file_name.replace(".parquet", "_optimized.parquet")
optimized_path = os.path.join(file_dir, file_name_optimized)

# Save the filtered dataset to a new Parquet file
df_filtered.to_parquet(optimized_path, index=False)

# Print confirmation and class balance
print(f"Optimized dataset with only the selected features and label saved to:\n{optimized_path}")
print("\nClass balance:")
print(df_filtered['label'].value_counts(normalize=True))
