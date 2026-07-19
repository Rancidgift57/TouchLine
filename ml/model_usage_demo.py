# NOTE: run this script with cwd = TouchLine/ml/  (e.g. `cd ml && python xg_model_architecture.py`)
# so the relative statsbomb_*.csv / *.pth / *.pkl filenames below resolve correctly.

import time
import random
import joblib
import numpy as np
import torch
import torch.nn as nn

print("Booting up AI Match Engine...")

# =========================================================
# 1. DEFINE NEURAL NETWORK ARCHITECTURES
# =========================================================
# PyTorch needs the "blueprints" to pour the saved weights into.

class DecisionNet(nn.Module):
    def __init__(self, input_dim):
        super(DecisionNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)        
        self.dropout1 = nn.Dropout(0.3)      
        self.layer2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(16, 3) # 3 outputs (Pass/Dribble/Shot)
        
    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.layer1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.layer2(x))))
        return self.output(x)

class XGNet(nn.Module):
    def __init__(self, input_dim):
        super(XGNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)        
        self.dropout1 = nn.Dropout(0.3)      
        self.layer2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(16, 1) # 1 output (Goal Probability)
        
    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.layer1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.layer2(x))))
        return self.output(x)

# =========================================================
# 2. FEATURE COLUMN MAPS (Crucial for Scikit-Learn Scaler)
# =========================================================
# These must exactly match the columns your models were trained on.
DECISION_FEATURES = ['x', 'y', 'distance_to_goal', 'angle_to_goal', 'under_pressure', 
                     'zone_def', 'zone_mid', 'zone_att', 'pattern_From Corner', 
                     'pattern_From Counter', 'pattern_From Free Kick', 'pattern_From Goal Kick', 
                     'pattern_From Keeper', 'pattern_From Kick Off', 'pattern_From Throw In', 
                     'pattern_Other', 'pattern_Regular Play']

# Assumes 4 body parts were in your dataset: Head, Left Foot, Right Foot, Other
XG_FEATURES = ['x', 'y', 'distance_to_goal', 'angle_to_goal', 'under_pressure', 
               'bp_Head', 'bp_Left Foot', 'bp_Other', 'bp_Right Foot',
               'pattern_From Corner', 'pattern_From Counter', 'pattern_From Free Kick', 
               'pattern_From Goal Kick', 'pattern_From Keeper', 'pattern_From Kick Off', 
               'pattern_From Throw In', 'pattern_Other', 'pattern_Regular Play']

# =========================================================
# 3. LIVE MATCH ENGINE
# =========================================================
class Player:
    def __init__(self, name, finishing, passing, dribbling):
        self.name = name
        self.attributes = {'finishing': finishing, 'passing': passing, 'dribbling': dribbling}
        self.fatigue = 0
        self.match_log = []

class LiveMatchEngine:
    def __init__(self):
        self.home_score = 0
        self.away_score = 0
        
        # Load Decision Model (17 features)
        self.decision_model = DecisionNet(input_dim=len(DECISION_FEATURES))
        self.decision_model.load_state_dict(torch.load('decision_model.pth'))
        self.decision_model.eval() # Lock weights
        self.decision_scaler = joblib.load('decision_scaler.pkl')
        
        # Load xG Model (18 features)
        self.xg_model = XGNet(input_dim=len(XG_FEATURES))
        self.xg_model.load_state_dict(torch.load('xg_model.pth'))
        self.xg_model.eval() # Lock weights
        self.xg_scaler = joblib.load('xg_scaler.pkl')

    def generate_random_scenario(self):
        """Generates a random tactical situation on the pitch."""
        x = round(random.uniform(20, 115), 1)
        y = round(random.uniform(5, 75), 1)
        pressure = random.choice([True, False])
        patterns = ["Regular Play", "From Counter", "From Corner"]
        pattern = random.choice(patterns)
        
        if pattern == "From Corner":
            x, y = 120.0, 80.0
        elif pattern == "From Counter":
            pressure = False
            
        return x, y, pressure, pattern

    def _get_ai_decision(self, x, y, pressure, pattern):
        """Feeds the scenario into the PyTorch Decision Model."""
        dist = np.sqrt((120 - x)**2 + (40 - y)**2)
        angle = np.abs(np.arctan2(40 - y, 120 - x))
        
        input_dict = {col: 0 for col in DECISION_FEATURES}
        input_dict['x'] = x; input_dict['y'] = y
        input_dict['distance_to_goal'] = dist; input_dict['angle_to_goal'] = angle
        input_dict['under_pressure'] = int(pressure)
        input_dict['zone_def'] = 1 if x <= 40 else 0
        input_dict['zone_mid'] = 1 if 40 < x <= 80 else 0
        input_dict['zone_att'] = 1 if x > 80 else 0
        
        if f"pattern_{pattern}" in input_dict:
            input_dict[f"pattern_{pattern}"] = 1
            
        vec = np.array([[input_dict[col] for col in DECISION_FEATURES]])
        scaled_vec = self.decision_scaler.transform(vec)
        tensor = torch.tensor(scaled_vec, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.decision_model(tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            
        actions = ['Pass', 'Dribble', 'Shot']
        return np.random.choice(actions, p=probs) # Rolls dice based on AI probabilities

    def _get_xg(self, x, y, pressure, pattern):
        """Feeds the scenario into the PyTorch xG Model."""
        dist = np.sqrt((120 - x)**2 + (40 - y)**2)
        angle = np.abs(np.arctan2(40 - y, 120 - x))
        
        input_dict = {col: 0 for col in XG_FEATURES}
        input_dict['x'] = x; input_dict['y'] = y
        input_dict['distance_to_goal'] = dist; input_dict['angle_to_goal'] = angle
        input_dict['under_pressure'] = int(pressure)
        
        # Assume a standard Right Foot shot for the simulation
        input_dict['bp_Right Foot'] = 1 
        if f"pattern_{pattern}" in input_dict:
            input_dict[f"pattern_{pattern}"] = 1
            
        vec = np.array([[input_dict[col] for col in XG_FEATURES]])
        scaled_vec = self.xg_scaler.transform(vec)
        tensor = torch.tensor(scaled_vec, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.xg_model(tensor)
            prob = torch.sigmoid(logits).item() # Sigmoid for binary goal/no-goal
            
        return round(prob, 3)

    def process_minute(self, minute, player):
        if random.random() > 0.15: # 15% chance of highlight event
            print(f"[{minute:02d}'] ...")
            time.sleep(0.1)
            return

        x, y, pressure, pattern = self.generate_random_scenario()
        player.fatigue = min(player.fatigue + 2, 100)
        
        print("\n" + "="*60)
        print(f"⏱️ MINUTE {minute:02d} | ATTACKING SCENARIO")
        print(f"📍 X:{x} Y:{y} | 🛡️ Pressured: {pressure} | 📋 {pattern}")
        print("-" * 60)
        time.sleep(0.5)
        
        decision = self._get_ai_decision(x, y, pressure, pattern)
        print(f"🧠 AI DECISION: {player.name} chooses to [ {decision.upper()} ]")
        time.sleep(0.5)
        
        if decision == "Shot":
            xg = self._get_xg(x, y, pressure, pattern)
            print(f"📊 PyTorch xG: {xg}")
            time.sleep(0.5)
            
            # Combine real PyTorch xG with player's FIFA attribute
            modifier = (player.attributes['finishing'] / 100) * 0.1
            roll = random.random()
            
            if roll <= (xg + modifier):
                self.home_score += 1
                print(f"⚽ GOALLLLLL! Spectacular finish!")
                player.match_log.append("Goal")
            else:
                print(f"❌ MISSED. Keeper saves or it goes wide.")
                player.match_log.append("Miss")
                
        elif decision == "Pass":
            roll = random.random()
            if roll > 0.15 + (player.fatigue/200): 
                print(f"👟 SUCCESSFUL PASS. Attack continues.")
                player.match_log.append("Pass")
            else:
                print(f"⚠️ INTERCEPTED. Possession lost.")
                
        elif decision == "Dribble":
            roll = random.random()
            if roll > 0.40: 
                print(f"💨 SUCCESSFUL DRIBBLE. {player.name} beats his man!")
                player.match_log.append("Dribble")
            else:
                print(f"🛑 TACKLED. Defense recovers.")
                
        print(f"📈 SCORE: Home {self.home_score} - {self.away_score} Away")
        print("="*60 + "\n")
        time.sleep(1.0)

# =========================================================
# 4. START THE MATCH
# =========================================================
if __name__ == "__main__":
    try:
        engine = LiveMatchEngine()
        messi = Player("Lionel Messi", finishing=95, passing=92, dribbling=98)
        
        print("\n🏟️ MATCH KICKOFF 🏟️")
        print("AI Models successfully loaded. Simulating 90 minutes...\n")
        time.sleep(1)
        
        for minute in range(1, 91):
            engine.process_minute(minute, messi)
            
        print("\n🏁 FULL TIME WHISTLE 🏁")
        print(f"Final Score: Home {engine.home_score} - {engine.away_score} Away")
        print(f"{messi.name} Log: {messi.match_log.count('Goal')} Goals, {messi.match_log.count('Pass')} Passes.")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Missing a saved file: {e}")
        print("Make sure 'decision_model.pth', 'decision_scaler.pkl', 'xg_model.pth', and 'xg_scaler.pkl' are in this folder!")
    except Exception as e:
        print(f"\n[ERROR] Model mismatch: {e}")
        print("Your saved models might have a different number of features than the ones defined in DECISION_FEATURES or XG_FEATURES above.")