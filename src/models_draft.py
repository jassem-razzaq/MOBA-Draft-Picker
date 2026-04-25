"""
Winner-guided draft prediction models.

  WinnerDraftMLPModel  — bag-of-champions MLP with side embedding (PyTorch)
  WinnerDraftLSTMModel — stacked LSTM with side embedding tiled across sequence (PyTorch)
  WinnerDraftXGBModel  — per-position XGBoost (one classifier per draft step)

All three share the same interface:
  train(X, pos, sides, y, X_val, pos_val, sides_val, y_val, ...)
  predict_proba(X, positions, sides) → (n, vocab_size)
  predict(X, positions, sides)       → (n,)
  evaluate(X, pos, sides, y)
  save(path) / load(path)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import pickle
from typing import Optional
from sklearn.metrics import accuracy_score, top_k_accuracy_score


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Shared helpers ────────────────────────────────────────────────────────────

def _eval_topk(name, y_test, y_pred, y_proba, vocab_size, sides_test=None):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Top-1  Accuracy: {accuracy:.4f}")
    metrics = {'accuracy': accuracy}
    for k in [3, 5, 10]:
        top_k = top_k_accuracy_score(
            y_test, y_proba, k=k, labels=np.arange(vocab_size))
        print(f"Top-{k:<2} Accuracy: {top_k:.4f}")
        metrics[f'top_{k}'] = top_k
    if sides_test is not None:
        for side_val, side_name in [(0, 'Blue'), (1, 'Red')]:
            mask = sides_test.flatten() == side_val
            if mask.sum():
                acc = accuracy_score(y_test[mask], y_pred[mask])
                print(f"  {side_name} accuracy: {acc:.4f}  ({mask.sum()} examples)")
    return metrics


def _train_loop(model, train_loader, val_loader, epochs, patience, label,
                has_weights=False):
    """Shared training loop with early stopping for PyTorch draft models."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(reduction='none')

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train    = 0
        for batch in train_loader:
            batch = [b.to(DEVICE) for b in batch]
            if has_weights:
                seq, pos, side, y, w = batch
            else:
                seq, pos, side, y   = batch
                w = None
            optimizer.zero_grad()
            logits = model(seq, pos, side)
            losses = criterion(logits, y)
            loss   = (losses * w).mean() if w is not None else losses.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            n_train    += len(y)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                seq, pos, side, y = [b.to(DEVICE) for b in batch[:4]]
                logits = model(seq, pos, side)
                val_loss += criterion(logits, y).mean().item() * len(y)
                n_val += len(y)
        val_loss /= n_val
        scheduler.step(val_loss)
        print(f"  [{label}] Epoch {epoch:3d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)


def _make_loader(X, pos, sides, y, batch_size, shuffle, sample_weight=None):
    tensors = [
        torch.tensor(X,     dtype=torch.long),
        torch.tensor(pos.flatten(),   dtype=torch.long),
        torch.tensor(sides.flatten(), dtype=torch.long),
        torch.tensor(y,     dtype=torch.long),
    ]
    if sample_weight is not None:
        tensors.append(torch.tensor(sample_weight, dtype=torch.float32))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


# ── Winner MLP (PyTorch) ──────────────────────────────────────────────────────

class _MLPNet(nn.Module):
    """
    Bag-of-champions: champion embeddings are masked, summed and meaned,
    then concatenated with position and side embeddings before MLP layers.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dims,
                 dropout_rate, n_draft_positions):
        super().__init__()
        self.champ_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_emb   = nn.Embedding(n_draft_positions, embedding_dim)
        self.side_emb  = nn.Embedding(2, embedding_dim)

        in_dim = embedding_dim * 4  # sum + mean + pos + side
        layers = []
        for dim in hidden_dims:
            layers += [nn.Linear(in_dim, dim), nn.ReLU(),
                       nn.BatchNorm1d(dim), nn.Dropout(dropout_rate)]
            in_dim = dim
        layers.append(nn.Linear(in_dim, vocab_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, seq, pos, side):
        emb  = self.champ_emb(seq)                         # (B, L, E)
        mask = (seq != 0).float().unsqueeze(-1)             # (B, L, 1)
        masked = emb * mask
        s = masked.sum(dim=1)                               # (B, E)
        n = mask.sum(dim=1).clamp(min=1.0)
        m = s / n                                           # (B, E)
        p = self.pos_emb(pos)                               # (B, E)
        d = self.side_emb(side)                             # (B, E)
        x = torch.cat([s, m, p, d], dim=1)
        return self.mlp(x)


class WinnerDraftMLPModel:
    """
    Bag-of-champions MLP. Champion embeddings are masked, summed and meaned,
    then concatenated with position and side embeddings before the MLP layers.
    Order-invariant within the sequence — treats draft as a set, not a sequence.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 64,
                 hidden_dims: tuple = (256, 256, 128), dropout_rate: float = 0.3,
                 n_draft_positions: int = 20, seq_len: int = 20):
        self.vocab_size        = vocab_size
        self.embedding_dim     = embedding_dim
        self.hidden_dims       = hidden_dims
        self.dropout_rate      = dropout_rate
        self.n_draft_positions = n_draft_positions
        self.net               = None

    def train(self, X_train, pos_train, sides_train, y_train,
              X_val, pos_val, sides_val, y_val,
              sample_weight=None, epochs=30, batch_size=256):
        self.net = _MLPNet(self.vocab_size, self.embedding_dim, self.hidden_dims,
                           self.dropout_rate, self.n_draft_positions).to(DEVICE)
        train_loader = _make_loader(X_train, pos_train, sides_train, y_train,
                                    batch_size, shuffle=True, sample_weight=sample_weight)
        val_loader   = _make_loader(X_val,   pos_val,   sides_val,   y_val,
                                    batch_size, shuffle=False)
        _train_loop(self.net, train_loader, val_loader, epochs, patience=5,
                    label='MLP', has_weights=(sample_weight is not None))

    def predict_proba(self, X, positions, sides):
        self.net.eval()
        with torch.no_grad():
            logits = self.net(
                torch.tensor(X, dtype=torch.long).to(DEVICE),
                torch.tensor(positions.flatten(), dtype=torch.long).to(DEVICE),
                torch.tensor(sides.flatten(), dtype=torch.long).to(DEVICE),
            )
        return F.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, X, positions, sides):
        return np.argmax(self.predict_proba(X, positions, sides), axis=1)

    def evaluate(self, X_test, pos_test, sides_test, y_test):
        y_pred  = self.predict(X_test, pos_test, sides_test)
        y_proba = self.predict_proba(X_test, pos_test, sides_test)
        return _eval_topk("Winner MLP", y_test, y_pred, y_proba,
                          self.vocab_size, sides_test)

    def save(self, path: str):
        torch.save({'state': self.net.state_dict(),
                    'vocab_size': self.vocab_size,
                    'embedding_dim': self.embedding_dim,
                    'hidden_dims': self.hidden_dims,
                    'dropout_rate': self.dropout_rate,
                    'n_draft_positions': self.n_draft_positions}, path)
        print(f"Saved to {path}")

    def load(self, path: str):
        d = torch.load(path, map_location=DEVICE)
        self.vocab_size        = d['vocab_size']
        self.embedding_dim     = d['embedding_dim']
        self.hidden_dims       = d['hidden_dims']
        self.dropout_rate      = d['dropout_rate']
        self.n_draft_positions = d['n_draft_positions']
        self.net = _MLPNet(self.vocab_size, self.embedding_dim, self.hidden_dims,
                           self.dropout_rate, self.n_draft_positions).to(DEVICE)
        self.net.load_state_dict(d['state'])
        self.net.eval()
        print(f"Loaded from {path}")


# ── Winner LSTM (PyTorch) ─────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    """
    Two stacked LSTMs. Side and position embeddings are tiled across the full
    sequence so the LSTM sees side context at every time step.
    """
    def __init__(self, vocab_size, embedding_dim, lstm_units,
                 n_draft_positions, seq_len):
        super().__init__()
        self.seq_len    = seq_len
        self.champ_emb  = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_emb    = nn.Embedding(n_draft_positions, 16)
        self.side_emb   = nn.Embedding(2, 16)

        lstm_in = embedding_dim + 16 + 16
        self.lstm1   = nn.LSTM(lstm_in,   lstm_units, batch_first=True)
        self.drop1   = nn.Dropout(0.3)
        self.lstm2   = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.drop2   = nn.Dropout(0.3)
        self.head    = nn.Sequential(
            nn.Linear(lstm_units, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, vocab_size))

    def forward(self, seq, pos, side):
        B, L = seq.shape
        champ = self.champ_emb(seq)                          # (B, L, E)
        # tile pos and side across the sequence
        p = self.pos_emb(pos).unsqueeze(1).expand(B, L, 16) # (B, L, 16)
        s = self.side_emb(side).unsqueeze(1).expand(B, L, 16)
        x = torch.cat([champ, p, s], dim=-1)                 # (B, L, E+32)
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x[:, -1, :])                         # last step
        return self.head(x)


class WinnerDraftLSTMModel:
    """
    Two stacked LSTMs. Side and position embeddings are tiled across the full
    sequence so the LSTM sees side context at every time step.
    """
    def __init__(self, vocab_size: int, seq_len: int = 20,
                 embedding_dim: int = 64, lstm_units: int = 128,
                 n_draft_positions: int = 20):
        self.vocab_size        = vocab_size
        self.seq_len           = seq_len
        self.embedding_dim     = embedding_dim
        self.lstm_units        = lstm_units
        self.n_draft_positions = n_draft_positions
        self.net               = None

    def train(self, X_train, pos_train, sides_train, y_train,
              X_val, pos_val, sides_val, y_val,
              sample_weight=None, epochs=30, batch_size=256):
        self.net = _LSTMNet(self.vocab_size, self.embedding_dim, self.lstm_units,
                            self.n_draft_positions, self.seq_len).to(DEVICE)
        train_loader = _make_loader(X_train, pos_train, sides_train, y_train,
                                    batch_size, shuffle=True, sample_weight=sample_weight)
        val_loader   = _make_loader(X_val,   pos_val,   sides_val,   y_val,
                                    batch_size, shuffle=False)
        _train_loop(self.net, train_loader, val_loader, epochs, patience=5,
                    label='LSTM', has_weights=(sample_weight is not None))

    def predict_proba(self, X, positions, sides):
        self.net.eval()
        with torch.no_grad():
            logits = self.net(
                torch.tensor(X, dtype=torch.long).to(DEVICE),
                torch.tensor(positions.flatten(), dtype=torch.long).to(DEVICE),
                torch.tensor(sides.flatten(), dtype=torch.long).to(DEVICE),
            )
        return F.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, X, positions, sides):
        return np.argmax(self.predict_proba(X, positions, sides), axis=1)

    def evaluate(self, X_test, pos_test, sides_test, y_test):
        y_pred  = self.predict(X_test, pos_test, sides_test)
        y_proba = self.predict_proba(X_test, pos_test, sides_test)
        return _eval_topk("Winner LSTM", y_test, y_pred, y_proba,
                          self.vocab_size, sides_test)

    def save(self, path: str):
        torch.save({'state': self.net.state_dict(),
                    'vocab_size': self.vocab_size,
                    'seq_len': self.seq_len,
                    'embedding_dim': self.embedding_dim,
                    'lstm_units': self.lstm_units,
                    'n_draft_positions': self.n_draft_positions}, path)
        print(f"Saved to {path}")

    def load(self, path: str):
        d = torch.load(path, map_location=DEVICE)
        self.vocab_size        = d['vocab_size']
        self.seq_len           = d['seq_len']
        self.embedding_dim     = d['embedding_dim']
        self.lstm_units        = d['lstm_units']
        self.n_draft_positions = d['n_draft_positions']
        self.net = _LSTMNet(self.vocab_size, self.embedding_dim, self.lstm_units,
                            self.n_draft_positions, self.seq_len).to(DEVICE)
        self.net.load_state_dict(d['state'])
        self.net.eval()
        print(f"Loaded from {path}")


# ── Winner XGBoost ────────────────────────────────────────────────────────────

class WinnerDraftXGBModel:
    """
    One XGBoost multiclass classifier per draft position (20 total).
    Input: binary presence vector + normalised position + side indicator.
    Labels are remapped to contiguous 0..n-1 per position to satisfy XGBoost.
    """
    def __init__(self, vocab_size: int, n_draft_positions: int = 20,
                 max_depth: int = 6, learning_rate: float = 0.1,
                 n_estimators: int = 300, subsample: float = 0.8,
                 colsample_bytree: float = 0.8):
        self.vocab_size        = vocab_size
        self.n_draft_positions = n_draft_positions
        self.max_depth         = max_depth
        self.learning_rate     = learning_rate
        self.n_estimators      = n_estimators
        self.subsample         = subsample
        self.colsample_bytree  = colsample_bytree
        self.models         = {}
        self.label_maps     = {}
        self.inv_label_maps = {}

    def _encode(self, X, positions, sides):
        n         = len(X)
        pos_flat  = positions.flatten().astype(np.float32)
        side_flat = sides.flatten().astype(np.float32)
        presence  = np.zeros((n, self.vocab_size), dtype=np.float32)
        for i, seq in enumerate(X):
            for token in seq:
                if token > 0:
                    presence[i, token] = 1.0
        pos_norm = (pos_flat / (self.n_draft_positions - 1)).reshape(-1, 1)
        return np.hstack([presence, pos_norm, side_flat.reshape(-1, 1)])

    def train(self, X_train, pos_train, sides_train, y_train,
              X_val, pos_val, sides_val, y_val,
              sample_weight: Optional[np.ndarray] = None,
              epochs: int = 300, batch_size: int = 256):
        pos_flat = pos_train.flatten()
        print(f"Training Winner XGBoost across {len(np.unique(pos_flat))} positions...")
        for t in sorted(np.unique(pos_flat)):
            mask     = pos_flat == t
            X_t      = self._encode(X_train[mask], pos_train.flatten()[mask],
                                    sides_train.flatten()[mask])
            y_t      = y_train[mask]
            sw_t     = sample_weight[mask] if sample_weight is not None else None
            val_mask = pos_val.flatten() == t
            X_vt     = self._encode(X_val[val_mask], pos_val.flatten()[val_mask],
                                    sides_val.flatten()[val_mask])
            y_vt     = y_val[val_mask]

            unique_labels  = np.sort(np.unique(y_t))
            label_map      = {lbl: i for i, lbl in enumerate(unique_labels)}
            inv_label_map  = {i: lbl for lbl, i in label_map.items()}
            self.label_maps[t]     = label_map
            self.inv_label_maps[t] = inv_label_map

            y_t_mapped  = np.array([label_map[l]       for l in y_t],  dtype=np.int32)
            y_vt_mapped = np.array([label_map.get(l, 0) for l in y_vt], dtype=np.int32)

            model = xgb.XGBClassifier(
                objective='multi:softprob', num_class=len(unique_labels),
                max_depth=self.max_depth, learning_rate=self.learning_rate,
                n_estimators=self.n_estimators, subsample=self.subsample,
                colsample_bytree=self.colsample_bytree, tree_method='hist',
                eval_metric='mlogloss', early_stopping_rounds=20,
                random_state=42, verbosity=0)
            eval_set = [(X_vt, y_vt_mapped)] if len(X_vt) > 0 else None
            model.fit(X_t, y_t_mapped, sample_weight=sw_t,
                      eval_set=eval_set, verbose=False)
            self.models[t] = model
            print(f"  Position {t:2d}: {mask.sum():5d} examples  "
                  f"{len(unique_labels)} classes  best_iter={model.best_iteration}")
        print(f"Trained {len(self.models)} position models.")

    def predict_proba(self, X, positions, sides):
        output   = np.zeros((len(X), self.vocab_size), dtype=np.float32)
        pos_flat = positions.flatten()
        for t in np.unique(pos_flat):
            mask = pos_flat == t
            if t not in self.models:
                output[mask] = 1.0 / self.vocab_size
                continue
            feats       = self._encode(X[mask], pos_flat[mask], sides.flatten()[mask])
            local_probs = self.models[t].predict_proba(feats)
            for local_idx, token_id in self.inv_label_maps[t].items():
                output[mask, token_id] = local_probs[:, local_idx]
        return output

    def predict(self, X, positions, sides):
        return np.argmax(self.predict_proba(X, positions, sides), axis=1)

    def evaluate(self, X_test, pos_test, sides_test, y_test):
        y_pred  = self.predict(X_test, pos_test, sides_test)
        y_proba = self.predict_proba(X_test, pos_test, sides_test)
        return _eval_topk("Winner XGBoost", y_test, y_pred, y_proba,
                          self.vocab_size, sides_test)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'models': self.models, 'label_maps': self.label_maps,
                         'inv_label_maps': self.inv_label_maps,
                         'vocab_size': self.vocab_size,
                         'n_draft_positions': self.n_draft_positions}, f)
        print(f"Saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f: data = pickle.load(f)
        self.models            = data['models']
        self.label_maps        = data['label_maps']
        self.inv_label_maps    = data['inv_label_maps']
        self.vocab_size        = data['vocab_size']
        self.n_draft_positions = data['n_draft_positions']
        print(f"Loaded from {path}  ({len(self.models)} position models)")