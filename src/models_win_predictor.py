"""
Win probability models for League of Legends draft prediction.

  AlwaysBlueBaseline      — trivial majority-class baseline
  LogisticDraftModel      — L2-regularised logistic regression (SAGA solver)
  XGBoostDraftModel       — gradient-boosted trees
  NeuralNetworkDraftModel — three-layer feedforward network (PyTorch)
"""
import numpy as np
import xgboost as xgb
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, log_loss, roc_auc_score,
                             classification_report)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_metrics(name, results):
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"AUC-ROC:   {results['auc']:.4f}")
    print(f"Log Loss:  {results['log_loss']:.4f}")

def _metrics(y_test, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc':      roc_auc_score(y_test, y_proba),
        'log_loss': log_loss(y_test, y_proba),
    }


# ── Always-Blue Baseline ──────────────────────────────────────────────────────

class AlwaysBlueBaseline:
    """Predicts Blue wins for every game — sets the accuracy floor."""
    def __init__(self):
        self.blue_rate = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.blue_rate = float(y_train.mean())
        print(f"AlwaysBlue baseline — Blue win rate: {self.blue_rate:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X), dtype=np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.blue_rate or 0.53, dtype=np.float32)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = 0.5
        results = {'accuracy': accuracy_score(y_test, y_pred),
                   'auc': auc, 'log_loss': log_loss(y_test, y_proba)}
        _print_metrics("Always-Blue Baseline", results)
        print(f"  (accuracy = Blue win rate in test set, AUC = 0.5 by construction)")
        print(classification_report(y_test, y_pred,
                                    target_names=['Red Win', 'Blue Win'],
                                    zero_division=0))
        return results


# ── Logistic Regression ───────────────────────────────────────────────────────

class LogisticDraftModel:
    """L2-regularised logistic regression with SAGA solver."""
    def __init__(self, C: float = 0.1, max_iter: int = 1000):
        self.model = LogisticRegression(
            solver='saga', C=C, max_iter=max_iter, random_state=42, verbose=0)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print("Training Logistic Regression...")
        self.model.fit(X_train, y_train)
        print("Done.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        results = _metrics(y_test, y_pred, y_proba)
        _print_metrics("Logistic Regression", results)
        print(classification_report(y_test, y_pred,
                                    target_names=['Red Win', 'Blue Win']))
        return results

    def save(self, path: str):
        with open(path, 'wb') as f: pickle.dump(self.model, f)
        print(f"Saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f: self.model = pickle.load(f)
        print(f"Loaded from {path}")


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostDraftModel:
    """Gradient-boosted trees win probability model."""
    def __init__(self, max_depth=6, learning_rate=0.05, n_estimators=500,
                 subsample=0.8, colsample_bytree=0.8):
        self.params = dict(
            objective='binary:logistic', max_depth=max_depth,
            learning_rate=learning_rate, n_estimators=n_estimators,
            subsample=subsample, colsample_bytree=colsample_bytree,
            tree_method='hist', eval_metric='logloss',
            early_stopping_rounds=30, random_state=42,
        )
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray):
        print("Training XGBoost...")
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print(f"Best iteration: {self.model.best_iteration}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        results = _metrics(y_test, y_pred, y_proba)
        _print_metrics("XGBoost", results)
        print(classification_report(y_test, y_pred,
                                    target_names=['Red Win', 'Blue Win']))
        return results

    def save(self, path: str):
        with open(path, 'wb') as f: pickle.dump(self.model, f)
        print(f"Saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f: self.model = pickle.load(f)
        print(f"Loaded from {path}")


# ── Neural Network (PyTorch) ──────────────────────────────────────────────────

class _FNN(nn.Module):
    """Three-layer feedforward network: 512 → 256 → 128 → 1 (sigmoid)."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class NeuralNetworkDraftModel:
    """Three-layer feedforward network (512→256→128) with BatchNorm and dropout."""
    def __init__(self):
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 128,
              lr: float = 0.001, patience: int = 10):
        print(f"Building Neural Network... (device: {DEVICE})")
        self.model = _FNN(X_train.shape[1]).to(DEVICE)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-6)
        criterion = nn.BCELoss()

        # DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32))
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        best_val_loss = float('inf')
        best_state    = None
        epochs_no_imp = 0

        print("Training Neural Network...")
        for epoch in range(1, epochs + 1):
            # Training
            self.model.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(Xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(Xb)
            train_loss /= len(train_ds)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    val_loss += criterion(self.model(Xb), yb).item() * len(Xb)
            val_loss /= len(val_ds)

            scheduler.step(val_loss)
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in
                                 self.model.state_dict().items()}
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.to(DEVICE)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            return self.model(t).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        results = _metrics(y_test, y_pred, y_proba)
        _print_metrics("Neural Network", results)
        print(classification_report(y_test, y_pred,
                                    target_names=['Red Win', 'Blue Win']))
        return results

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Saved to {path}")

    def load(self, path: str):
        # input_dim unknown at load time — caller must pass it or we infer from weights
        state = torch.load(path, map_location=DEVICE)
        input_dim = state['net.0.weight'].shape[1]
        self.model = _FNN(input_dim).to(DEVICE)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"Loaded from {path}")