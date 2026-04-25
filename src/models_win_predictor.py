"""
Win probability models for League of Legends draft prediction.

  AlwaysBlueBaseline    — trivial majority-class baseline
  LogisticDraftModel    — L2-regularised logistic regression (SAGA solver)
  XGBoostDraftModel     — gradient-boosted trees
  NeuralNetworkDraftModel — three-layer feedforward network
"""
import numpy as np
import xgboost as xgb
import pickle
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, log_loss, roc_auc_score,
                             classification_report)


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


# ── Neural Network ────────────────────────────────────────────────────────────

class NeuralNetworkDraftModel:
    """Three-layer feedforward network (512→256→128) with BatchNorm and dropout."""
    def __init__(self):
        self.model = None

    def build_model(self, input_dim: int):
        return models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(), layers.Dropout(0.3),
            layers.Dense(256, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(), layers.Dropout(0.3),
            layers.Dense(128, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(), layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid'),
        ])

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 128):
        print("Building Neural Network...")
        self.model = self.build_model(X_train.shape[1])
        self.model.compile(optimizer=keras.optimizers.Adam(0.001),
                           loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
        self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                              restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                  patience=5, min_lr=1e-6),
            ],
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.model.predict(X, verbose=0).flatten() >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        results = _metrics(y_test, y_pred, y_proba)
        _print_metrics("Neural Network", results)
        print(classification_report(y_test, y_pred,
                                    target_names=['Red Win', 'Blue Win']))
        return results

    def save(self, path: str):
        self.model.save(path)
        print(f"Saved to {path}")

    def load(self, path: str):
        self.model = keras.models.load_model(path)
        print(f"Loaded from {path}")