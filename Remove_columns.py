#removes 116 columns and keeps 103

import pandas as pd
import pyarrow.parquet as pq

# Define columns to remove (safer version)
columns_to_drop = [
    # Metadata/identifiers
    'event_no', 'jet_no', 'sample_isQCD',
    
    # Constant-value labels
    'label_H_cc', 'label_H_qqqq', 'label_Top_bc', 'label_Top_bcq',
    'label_Top_bq', 'label_Top_bqq', 'label_W_cq', 'label_W_qq',
    'label_Z_bb', 'label_Z_cc', 'label_Z_qq', 'isUndefined',
    
    # Redundant columns
    'npfcands', 'ntracks', 'nsv',  # Keep integer versions
    'rho', 'npv', 'ntrueInt',
    
    # Legacy b-taggers
    'pfJetBProbabilityBJetTags', 'pfJetProbabilityBJetTags',
    'softPFElectronBJetTags',
    
    # Less important DeepCSV components
    'pfDeepCSVJetTags_probc', 'pfDeepCSVJetTags_probcc', 'pfDeepCSVJetTags_probudsg',
    
    # High-dimensional track/pfcand lists
    'pfcand_VTX_ass', 'pfcand_charge', 'pfcand_deltaR', 'pfcand_drminsv',
    'pfcand_drsubjet1', 'pfcand_drsubjet2', 'pfcand_dxy', 'pfcand_dxysig',
    'pfcand_dz', 'pfcand_dzsig', 'pfcand_erel', 'pfcand_etarel',
    'pfcand_fromPV', 'pfcand_hcalFrac', 'pfcand_isChargedHad', 'pfcand_isEl',
    'pfcand_isGamma', 'pfcand_isMu', 'pfcand_isNeutralHad', 'pfcand_lostInnerHits',
    'pfcand_mass', 'pfcand_phirel', 'pfcand_ptrel', 'pfcand_puppiw',
    'trackBTag_DeltaR', 'trackBTag_Eta', 'trackBTag_EtaRel', 'trackBTag_JetDistVal',
    'trackBTag_Momentum', 'trackBTag_PPar', 'trackBTag_PParRatio',
    'trackBTag_PtRatio', 'trackBTag_PtRel', 'trackBTag_Sip2dSig',
    'trackBTag_Sip2dVal', 'trackBTag_Sip3dSig', 'trackBTag_Sip3dVal',
    'track_VTX_ass', 'track_charge', 'track_deltaR', 'track_detadeta',
    'track_dlambdadz', 'track_dphidphi', 'track_dphidxy', 'track_dptdpt',
    'track_drminsv', 'track_drsubjet1', 'track_drsubjet2', 'track_dxy',
    'track_dxydxy', 'track_dxydz', 'track_dxysig', 'track_dz', 'track_dzdz',
    'track_dzsig', 'track_erel', 'track_etarel', 'track_fromPV',
    'track_isChargedHad', 'track_isEl', 'track_isMu', 'track_lostInnerHits',
    'track_mass', 'track_normchi2', 'track_phirel', 'track_pt', 'track_ptrel',
    'track_puppiw', 'track_quality',
    
    # Secondary vertex details
    'sv_chi2', 'sv_costhetasvpv', 'sv_d3d', 'sv_d3derr', 'sv_d3dsig',
    'sv_deltaR', 'sv_dxy', 'sv_dxyerr', 'sv_dxysig', 'sv_erel', 'sv_etarel',
    'sv_mass', 'sv_ndf', 'sv_normchi2', 'sv_ntracks', 'sv_phirel', 'sv_pt',
    'sv_ptrel',
    
    # Legacy labels
    'fj_labelLegacy', 'fj_labelJMAR'
]

def optimize_parquet(input_path, output_path):
    # Read Parquet file
    table = pq.read_table(input_path)
    
    # Get existing columns (to avoid KeyError)
    existing_columns = table.column_names
    columns_to_remove = [col for col in columns_to_drop if col in existing_columns]
    
    # Remove columns
    optimized_table = table.drop(columns_to_remove)
    
    # Write new file
    pq.write_table(optimized_table, output_path)
    print(f"Optimized file created: {output_path}")
    print(f"Removed {len(columns_to_remove)} columns, kept {len(optimized_table.column_names)}")

# Path configuration
input_path = r"C:\Users\Dev\OneDrive\Desktop\hbb_production_parquet\ntuple_merged_12.parquet"
output_path = input_path.replace(".parquet", "_optimized.parquet")

# Run optimization
optimize_parquet(input_path, output_path)