# NOTE: run this script with cwd = TouchLine/ml/  (e.g. `cd ml && python xg_model_architecture.py`)
# so the relative statsbomb_*.csv / *.pth / *.pkl filenames below resolve correctly.

import os
import ast
import warnings
import joblib
import pandas as pd
import numpy as np
from statsbombpy import sb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Suppress the NoAuthWarning from StatsBomb to keep the terminal clean
warnings.filterwarnings("ignore", message="credentials were not supplied")

# ---------------------------------------------------------
# 1. DATA EXTRACTION & ENGINEERING (MULTI-TOURNAMENT SCALE)
# ---------------------------------------------------------
print("--- STATSBOMB NEURAL NETWORK INITIALIZED ---")

# Check if we already downloaded the massive dataset
if os.path.exists('statsbomb_shots_data.csv'):
    print("Found saved dataset! Loading directly from CSV...")
    shots = pd.read_csv('statsbomb_shots_data.csv')
    print(f"Total Shots Loaded: {len(shots)}")
    
else:
    print("Fetching multi-tournament match list (This will take 10-20 minutes)...")

    # List of (competition_id, season_id) from StatsBomb Free Data
    tournaments = [
        (43, 106), # World Cup 2022
        (43, 3),   # World Cup 2018
        (55, 43),  # Euro 2020
        (16, 27),  # Champions League 2020/2021
        (11, 90)   # La Liga 2020/2021 (Messi Data)
    ]

    all_shots_list = []

    # Loop through each tournament
    for comp_id, season_id in tournaments:
        print(f"\n-> Fetching Matches for Competition {comp_id}, Season {season_id}...")
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            match_ids = matches['match_id'].tolist()
            print(f"   Found {len(match_ids)} matches. Extracting shots...")
            
            # Loop through every match in this tournament
            for i, match_id in enumerate(match_ids, 1):
                try:
                    events = sb.events(match_id=match_id)
                    if 'type' in events.columns:
                        match_shots = events[events['type'] == 'Shot'].copy()
                        all_shots_list.append(match_shots)
                except Exception as e:
                    pass # Silently skip specific matches that fail to download
                    
        except Exception as e:
            print(f"   Could not load tournament {comp_id}/{season_id}: {e}")

    # Combine everything and de-fragment memory
    print("\nConcatenating all matches into a single dataset...")
    shots = pd.concat(all_shots_list, ignore_index=True).copy()
    
    # Save to CSV so we never have to wait for this download again!
    print("Saving dataset to 'statsbomb_shots_data.csv'...")
    shots.to_csv('statsbomb_shots_data.csv', index=False)
    print(f"Total Shots Extracted: {len(shots)}")

print("\nProcessing Features...")

# --- FEATURE ENGINEERING: NUMERICAL ---
# Function to safely parse coordinates whether they come from the API (list) or CSV (string)
def parse_location(loc):
    if isinstance(loc, list): return loc
    if pd.isna(loc): return [np.nan, np.nan]
    try: return ast.literal_eval(loc)
    except: return [np.nan, np.nan]

shots['parsed_location'] = shots['location'].apply(parse_location)
shots['x'] = shots['parsed_location'].apply(lambda loc: loc[0] if len(loc) == 2 else np.nan)
shots['y'] = shots['parsed_location'].apply(lambda loc: loc[1] if len(loc) == 2 else np.nan)

# Calculate Geometry
shots['distance_to_goal'] = np.sqrt((120 - shots['x'])**2 + (40 - shots['y'])**2)
shots['angle_to_goal'] = np.abs(np.arctan2(40 - shots['y'], 120 - shots['x']))

# Handle missing 'under_pressure' flags safely
shots['under_pressure'] = shots['under_pressure'].fillna(0).astype(int)

# Define Target (1 for Goal, 0 for Miss/Save/Block)
shots['is_goal'] = (shots['shot_outcome'] == 'Goal').astype(int)

# --- FEATURE ENGINEERING: CATEGORICAL (One-Hot Encoding) ---
body_parts = pd.get_dummies(shots['shot_body_part'], prefix='bp', dtype=int)
play_patterns = pd.get_dummies(shots['play_pattern'], prefix='pattern', dtype=int)

# Attach binary columns back to our main DataFrame
shots = pd.concat([shots, body_parts, play_patterns], axis=1)

# --- FINAL FEATURE SELECTION ---
base_features = ['x', 'y', 'distance_to_goal', 'angle_to_goal', 'under_pressure']
features = base_features + list(body_parts.columns) + list(play_patterns.columns)

# Drop rows with missing spatial coordinates and isolate target
X_df = shots[features].dropna()
y_df = shots.loc[X_df.index, 'is_goal']

X = X_df.values
y = y_df.values

print(f"Feature Engineering Complete! Training Network on {len(X)} shots with {X.shape[1]} features.\n")

# ---------------------------------------------------------
# 2. PYTORCH DATASET DEFINITION
# ---------------------------------------------------------
class XGDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        # targets need to be shape (N, 1) for BCEWithLogitsLoss
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------
# 3. NEURAL NETWORK ARCHITECTURE
# ---------------------------------------------------------
class XGNet(nn.Module):
    def __init__(self, input_dim):
        super(XGNet, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)        
        self.dropout1 = nn.Dropout(0.3)      

        self.layer2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.3)

        self.output = nn.Linear(16, 1)       

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.output(x) 
        return x

# ---------------------------------------------------------
# 4. STRATIFIED K-FOLD CROSS VALIDATION SETUP
# ---------------------------------------------------------
K_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_results = []

print(f"Starting {K_FOLDS}-Fold Stratified Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"\n--- Fold {fold + 1} ---")
    
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Create DataLoaders
    # Create DataLoaders
    train_loader = DataLoader(XGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(XGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model dynamically
    model = XGNet(input_dim=X_train.shape[1])
    
    # Handle Class Imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32) if num_pos > 0 else torch.tensor([1.0])
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 
    
    # ------------- TRAINING LOOP -------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
    # ------------- VALIDATION LOOP -------------
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            
            probs = torch.sigmoid(predictions).numpy()
            all_preds.extend(probs)
            all_targets.extend(batch_y.numpy())
            
    val_loss /= len(val_loader.dataset)
    
    if len(np.unique(all_targets)) == 1:
        print("Warning: Only one class in validation set. Defaulting AUC to 0.5")
        auc = 0.5 
    else:
        auc = roc_auc_score(all_targets, all_preds)
        
    print(f"Final Val Loss: {val_loss:.4f} | Val ROC-AUC: {auc:.4f}")
    fold_results.append(auc)

print("\n==================================")
print(f"FINAL AVERAGE ROC-AUC SCORE: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

# =========================================================
# 5. SAVE THE MODEL AND SCALER
# =========================================================
print("\nSaving the xG Model and Scaler to disk...")

# Save the PyTorch model weights
torch.save(model.state_dict(), 'xg_model.pth')

# Save the Scikit-Learn scaler
joblib.dump(scaler, 'xg_scaler.pkl')

print("Success! 'xg_model.pth' and 'xg_scaler.pkl' are saved and ready for the Match Engine.")