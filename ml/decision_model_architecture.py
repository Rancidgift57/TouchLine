# NOTE: run this script with cwd = TouchLine/ml/  (e.g. `cd ml && python xg_model_architecture.py`)
# so the relative statsbomb_*.csv / *.pth / *.pkl filenames below resolve correctly.

import os
import ast
import warnings
import pandas as pd
import joblib
import numpy as np
from statsbombpy import sb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Suppress the NoAuthWarning from StatsBomb
warnings.filterwarnings("ignore", message="credentials were not supplied")

# ---------------------------------------------------------
# 1. DATA EXTRACTION (PASS, DRIBBLE, SHOT)
# ---------------------------------------------------------
print("--- UPGRADED DECISION MODEL INITIALIZED ---")

if os.path.exists('statsbomb_decision_data.csv'):
    print("Found saved dataset! Loading directly from CSV...")
    actions = pd.read_csv('statsbomb_decision_data.csv')
else:
    print("Fetching Match List (Restricted to World Cup 2022 to save RAM)...")
    tournaments = [(43, 106)] 
    all_actions_list = []

    for comp_id, season_id in tournaments:
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            match_ids = matches['match_id'].tolist()
            print(f"Found {len(match_ids)} matches. Extracting actions...")
            
            for i, match_id in enumerate(match_ids, 1):
                try:
                    events = sb.events(match_id=match_id)
                    if 'type' in events.columns:
                        match_actions = events[events['type'].isin(['Pass', 'Dribble', 'Shot'])].copy()
                        all_actions_list.append(match_actions)
                except Exception:
                    pass 
        except Exception as e:
            print(f"Error: {e}")

    print("\nConcatenating dataset...")
    actions = pd.concat(all_actions_list, ignore_index=True).copy()
    
    print("Saving dataset to 'statsbomb_decision_data.csv'...")
    actions.to_csv('statsbomb_decision_data.csv', index=False)

print(f"Total Actions Loaded: {len(actions)}")

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING (NOW WITH CONTEXT)
# ---------------------------------------------------------
print("Processing Advanced Features...")

def parse_location(loc):
    if isinstance(loc, list): return loc
    if pd.isna(loc): return [np.nan, np.nan]
    try: return ast.literal_eval(loc)
    except: return [np.nan, np.nan]

actions['parsed_location'] = actions['location'].apply(parse_location)
actions['x'] = actions['parsed_location'].apply(lambda loc: loc[0] if len(loc) == 2 else np.nan)
actions['y'] = actions['parsed_location'].apply(lambda loc: loc[1] if len(loc) == 2 else np.nan)

# Base Geometry
actions['distance_to_goal'] = np.sqrt((120 - actions['x'])**2 + (40 - actions['y'])**2)
actions['angle_to_goal'] = np.abs(np.arctan2(40 - actions['y'], 120 - actions['x']))
actions['under_pressure'] = actions['under_pressure'].fillna(0).astype(int)

# NEW 1: Pitch Zones (Defensive, Middle, Attacking Thirds)
actions['zone_def'] = (actions['x'] <= 40).astype(int)
actions['zone_mid'] = ((actions['x'] > 40) & (actions['x'] <= 80)).astype(int)
actions['zone_att'] = (actions['x'] > 80).astype(int)

# NEW 2: Play Patterns (One-Hot Encoded)
play_patterns = pd.get_dummies(actions['play_pattern'], prefix='pattern', dtype=int)
actions = pd.concat([actions, play_patterns], axis=1)

# Multi-Class Target Encoding (Pass=0, Dribble=1, Shot=2)
action_mapping = {'Pass': 0, 'Dribble': 1, 'Shot': 2}
actions['action_target'] = actions['type'].map(action_mapping)

# Combine all features
base_features = ['x', 'y', 'distance_to_goal', 'angle_to_goal', 'under_pressure', 'zone_def', 'zone_mid', 'zone_att']
features = base_features + list(play_patterns.columns)

X_df = actions[features + ['action_target']].dropna()

X = X_df[features].values
y = X_df['action_target'].values.astype(int)

print(f"Training Network on {len(X)} actions with {X.shape[1]} features.\n")

# ---------------------------------------------------------
# 3. PYTORCH DATASET & ARCHITECTURE 
# ---------------------------------------------------------
class DecisionDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.long) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DecisionNet(nn.Module):
    def __init__(self, input_dim):
        super(DecisionNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)        
        self.dropout1 = nn.Dropout(0.3)      

        self.layer2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.3)

        self.output = nn.Linear(16, 3)       

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
# 4. TRAINING & EVALUATION
# ---------------------------------------------------------
K_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_accuracies = []

print(f"Starting {K_FOLDS}-Fold Stratified Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"\n--- Fold {fold + 1} ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    train_loader = DataLoader(DecisionDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(DecisionDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    model = DecisionNet(input_dim=X_train.shape[1])
    
    # NEW 3: Smoothed Class Weights
    # We calculate the balanced weights, but take the square root to dampen extreme numbers.
    # This prevents the network from over-predicting rare events (Shots/Dribbles).
    raw_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    smoothed_weights = np.sqrt(raw_weights)
    weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            preds = torch.argmax(predictions, dim=1).numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    print(f"Val Accuracy: {acc * 100:.2f}%")
    
    if fold == K_FOLDS - 1:
        print("\n--- Final Fold Breakdown ---")
        print(classification_report(all_targets, all_preds, target_names=['Pass (0)', 'Dribble (1)', 'Shot (2)']))
        
    fold_accuracies.append(acc)

print("\n==================================")
print(f"FINAL AVERAGE ACCURACY: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")

# =========================================================
# 5. SAVE THE MODEL AND SCALER
# =========================================================
print("\nSaving the xG Model and Scaler to disk...")

# Save the PyTorch model weights
torch.save(model.state_dict(), 'decision_model.pth')

# Save the Scikit-Learn scaler
joblib.dump(scaler, 'decision_scaler.pkl')

print("Success! 'decision_model.pth' and 'decision_scaler.pkl' are saved and ready for the Match Engine.")