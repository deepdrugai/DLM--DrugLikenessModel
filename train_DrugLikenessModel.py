"""
Filename: train_DrugLikenessModel.py
Description: Full training pipeline for the Drug-Likeness Self-Normalizing Neural Network (SNN).
"""
import numpy as np
import pandas as pd
import json
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import warnings

warnings.filterwarnings('ignore')

class DrugLikenessModel(nn.Module):
    def __init__(self, input_dim=1238):
        super(DrugLikenessModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.BatchNorm1d(2048), nn.SELU(),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.SELU(), nn.AlphaDropout(0.2),
            nn.Linear(1024, 512), nn.SELU(), nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, x): 
        return self.net(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Training on {device}...")

    # 1. Load Training Data
    # NOTE: Ensure data paths match your local repository structure
    feat = pd.read_csv("data/processed/massive_augmented_features.csv")
    lab = pd.read_csv("data/raw/massive_master_dataset_augmented.csv").rename(columns={'canonical_smiles':'SMILES'})
    data = pd.merge(feat, lab[['SMILES', 'label', 'chembl_id']], on='SMILES').drop_duplicates('SMILES')

    # Drop unstable features to prevent gradient overflow
    data = data.drop(columns=[c for c in ['Ipc', 'Kappa3', 'HallKierAlpha'] if c in data.columns])
    
    # Correct complexity bias via oversampling
    decoys = data[data['chembl_id'].str.contains('MANUAL', na=False)]
    data = pd.concat([data] + [decoys]*100).sample(frac=1, random_state=42).reset_index(drop=True)

    cols = [c for c in data.columns if c not in ['SMILES', 'label', 'chembl_id']]
    data[cols] = data[cols].replace([np.inf, -np.inf], np.nan).fillna(data[cols].median())
    
    # Save feature names
    with open("feature_names.json", "w") as f:
        json.dump(cols, f)

    X = data[cols].values
    y = data['label'].values
    
    # 2. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    cont_idx = [i for i, f in enumerate(cols) if not f.startswith("bit_")]
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
    
    X_train_scaled = X_train.copy().astype(np.float32)
    X_train_scaled[:, cont_idx] = scaler.fit_transform(X_train[:, cont_idx])
    joblib.dump(scaler, "scaler.pkl")

    # 3. Model Initialization
    model = DrugLikenessModel(input_dim=len(cols)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    
    train_dataset = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # 4. Training Loop
    print("Training SNN for 300 Epochs...")
    model.train()
    for epoch in range(300):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/300] Loss: {loss.item():.4f}")

    # 5. Save Final Model
    torch.save(model.state_dict(), "DrugLikenessModel.pth")
    print("Training Complete. Model saved as DrugLikenessModel.pth")
