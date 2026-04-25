"""
Winner-guided draft prediction models.

  WinnerDraftMLPModel  — bag-of-champions MLP with side embedding
  WinnerDraftLSTMModel — stacked LSTM with side embedding tiled across sequence
  WinnerDraftXGBModel  — per-position XGBoost (one classifier per draft step)

All three share the same interface:
  train(X, pos, sides, y, X_val, pos_val, sides_val, y_val, ...)
  predict_proba(X, positions, sides) → (n, vocab_size)
  predict(X, positions, sides)       → (n,)
  evaluate(X, pos, sides, y)
  save(path) / load(path)
"""
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
import xgboost as xgb
import pickle
from typing import Optional
from sklearn.metrics import accuracy_score, top_k_accuracy_score


# Shared helpers

def _callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ]

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


# ── Winner MLP ────────────────────────────────────────────────────────────────

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
        self.seq_len           = seq_len
        self.model             = None

    def build_model(self):
        seq_input  = layers.Input(shape=(self.seq_len,), name='sequence_input')
        pos_input  = layers.Input(shape=(1,),            name='position_input')
        side_input = layers.Input(shape=(1,),            name='side_input')

        champ_embs = layers.Embedding(
            self.vocab_size, self.embedding_dim, name='champion_embedding')(seq_input)
        mask       = keras.ops.expand_dims(
            keras.ops.cast(keras.ops.not_equal(seq_input, 0), 'float32'), -1)
        masked     = champ_embs * mask
        champ_sum  = keras.ops.sum(masked, axis=1)
        champ_mean = champ_sum / keras.ops.maximum(keras.ops.sum(mask, axis=1), 1.0)
        champ_agg  = layers.Concatenate(name='champ_agg')([champ_sum, champ_mean])

        pos_emb  = layers.Flatten()(layers.Embedding(
            self.n_draft_positions, self.embedding_dim, name='pos_emb')(pos_input))
        side_emb = layers.Flatten()(layers.Embedding(
            2, self.embedding_dim, name='side_emb')(side_input))

        x = layers.Concatenate(name='feature_concat')([champ_agg, pos_emb, side_emb])
        for i, dim in enumerate(self.hidden_dims):
            x = layers.Dense(dim, activation='relu', name=f'hidden_{i}')(x)
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'drop_{i}')(x)
        output = layers.Dense(self.vocab_size, activation='softmax', name='output')(x)

        self.model = models.Model(inputs=[seq_input, pos_input, side_input], outputs=output)

    def train(self, X_train, pos_train, sides_train, y_train,
              X_val, pos_val, sides_val, y_val,
              sample_weight=None, epochs=30, batch_size=256):
        self.build_model()
        self.model.compile(optimizer=keras.optimizers.Adam(0.001),
                           loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        return self.model.fit(
            [X_train, pos_train, sides_train], y_train,
            validation_data=([X_val, pos_val, sides_val], y_val),
            sample_weight=sample_weight, epochs=epochs,
            batch_size=batch_size, callbacks=_callbacks(), verbose=1)

    def predict_proba(self, X, positions, sides):
        return self.model(
            [tf.constant(X, tf.int32), tf.constant(positions, tf.int32),
             tf.constant(sides, tf.int32)], training=False).numpy()

    def predict(self, X, positions, sides):
        return np.argmax(self.predict_proba(X, positions, sides), axis=1)

    def evaluate(self, X_test, pos_test, sides_test, y_test):
        y_pred  = self.predict(X_test, pos_test, sides_test)
        y_proba = self.predict_proba(X_test, pos_test, sides_test)
        return _eval_topk("Winner MLP", y_test, y_pred, y_proba,
                          self.vocab_size, sides_test)

    def save(self, path: str):
        self.model.save(path); print(f"Saved to {path}")

    def load(self, path: str):
        self.model = keras.models.load_model(path); print(f"Loaded from {path}")


# ── Winner LSTM ───────────────────────────────────────────────────────────────

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
        self.model             = None

    def build_model(self):
        seq_input  = layers.Input(shape=(self.seq_len,), name='sequence_input')
        pos_input  = layers.Input(shape=(1,),            name='position_input')
        side_input = layers.Input(shape=(1,),            name='side_input')

        champ_emb = layers.Embedding(
            self.vocab_size, self.embedding_dim,
            mask_zero=True, name='champion_embedding')(seq_input)

        # Tile position and side embeddings across sequence length
        pos_emb  = layers.RepeatVector(self.seq_len)(layers.Flatten()(
            layers.Embedding(self.n_draft_positions, 16, name='pos_emb')(pos_input)))
        side_emb = layers.RepeatVector(self.seq_len)(layers.Flatten()(
            layers.Embedding(2, 16, name='side_emb')(side_input)))

        x = layers.Concatenate()([champ_emb, pos_emb, side_emb])
        x = layers.Dropout(0.3)(layers.LSTM(self.lstm_units, return_sequences=True)(x))
        x = layers.Dropout(0.3)(layers.LSTM(self.lstm_units)(x))
        x = layers.Dropout(0.2)(layers.Dense(256, activation='relu')(x))
        output = layers.Dense(self.vocab_size, activation='softmax', name='output')(x)

        self.model = models.Model(inputs=[seq_input, pos_input, side_input], outputs=output)

    def train(self, X_train, pos_train, sides_train, y_train,
              X_val, pos_val, sides_val, y_val,
              sample_weight=None, epochs=30, batch_size=256):
        self.build_model()
        self.model.compile(optimizer=keras.optimizers.Adam(0.001),
                           loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        return self.model.fit(
            [X_train, pos_train, sides_train], y_train,
            validation_data=([X_val, pos_val, sides_val], y_val),
            sample_weight=sample_weight, epochs=epochs,
            batch_size=batch_size, callbacks=_callbacks(), verbose=1)

    def predict_proba(self, X, positions, sides):
        return self.model(
            [tf.constant(X, tf.int32), tf.constant(positions, tf.int32),
             tf.constant(sides, tf.int32)], training=False).numpy()

    def predict(self, X, positions, sides):
        return np.argmax(self.predict_proba(X, positions, sides), axis=1)

    def evaluate(self, X_test, pos_test, sides_test, y_test):
        y_pred  = self.predict(X_test, pos_test, sides_test)
        y_proba = self.predict_proba(X_test, pos_test, sides_test)
        return _eval_topk("Winner LSTM", y_test, y_pred, y_proba,
                          self.vocab_size, sides_test)

    def save(self, path: str):
        self.model.save(path); print(f"Saved to {path}")

    def load(self, path: str):
        self.model = keras.models.load_model(path); print(f"Loaded from {path}")


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

            y_t_mapped  = np.array([label_map[l]      for l in y_t],  dtype=np.int32)
            y_vt_mapped = np.array([label_map.get(l,0) for l in y_vt], dtype=np.int32)

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