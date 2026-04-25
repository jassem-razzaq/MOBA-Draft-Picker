"""
Train all three winner-guided draft models (MLP, LSTM, XGBoost).
Only winning-side picks are used for training, with inverse-frequency
sample weighting to correct for Blue winning ~52.7% of games.
"""
import numpy as np
import sys
import os
sys.path.append('src')

from draft_data_preprocessing import WinnerSequenceProcessor, BLUE_SIDE, RED_SIDE
from models_draft import WinnerDraftMLPModel, WinnerDraftLSTMModel, WinnerDraftXGBModel
from sklearn.model_selection import train_test_split
import pickle


def compute_side_weights(sides_flat: np.ndarray) -> np.ndarray:
    n_blue = (sides_flat == BLUE_SIDE).sum()
    n_red  = (sides_flat == RED_SIDE).sum()
    total  = n_blue + n_red
    print(f"  Blue: {n_blue}  Red: {n_red}  "
          f"(weights — Blue: {total/(2*n_blue):.4f}  Red: {total/(2*n_red):.4f})")
    return np.where(sides_flat == BLUE_SIDE,
                    total / (2 * n_blue),
                    total / (2 * n_red)).astype(np.float32)


def train_all_winner_models():
    print("=" * 60)
    print("Winner-Guided Draft Model Training (MLP + LSTM + XGBoost)")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    processor = WinnerSequenceProcessor("src/2024.csv")
    df = processor.load_data()
    processor.fit_encoder(df)
    winner_sequences = processor.build_winner_sequences(df)

    metadata_path = "data/draft_metadata.pkl"
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            vocab_size = pickle.load(f)['vocab_size']
        print(f"Using existing metadata (vocab_size={vocab_size})")
    else:
        processor.save_metadata(metadata_path)
        vocab_size = processor.vocab_size

    print("\n[2/5] Splitting data...")
    indices = np.arange(len(winner_sequences))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    _, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    train_seqs = [winner_sequences[i] for i in train_idx]
    val_seqs   = [winner_sequences[i] for i in temp_idx[:len(temp_idx)//2]]
    test_seqs  = [winner_sequences[i] for i in test_idx]
    print(f"Train: {len(train_seqs)} | Val: {len(val_seqs)} | Test: {len(test_seqs)}")

    X_train, pos_train, sides_train, y_train = processor.create_winner_examples(train_seqs)
    X_val,   pos_val,   sides_val,   y_val   = processor.create_winner_examples(val_seqs)
    X_test,  pos_test,  sides_test,  y_test  = processor.create_winner_examples(test_seqs)

    def to2d(a): return a.reshape(-1, 1)
    pos_train_2d,   pos_val_2d,   pos_test_2d   = to2d(pos_train),   to2d(pos_val),   to2d(pos_test)
    sides_train_2d, sides_val_2d, sides_test_2d = to2d(sides_train), to2d(sides_val), to2d(sides_test)

    print("\n[3/5] Computing sample weights...")
    weights = compute_side_weights(sides_train)

    results = {}

    print("\n[4/5a] Training Winner MLP...")
    mlp = WinnerDraftMLPModel(vocab_size=vocab_size, embedding_dim=64,
                              hidden_dims=(256, 256, 128), dropout_rate=0.3)
    mlp.train(X_train, pos_train_2d, sides_train_2d, y_train,
              X_val,   pos_val_2d,   sides_val_2d,   y_val,
              sample_weight=weights, epochs=30, batch_size=256)
    results['mlp'] = mlp.evaluate(X_test, pos_test_2d, sides_test_2d, y_test)
    mlp.save("models/draft_mlp_winner.keras")

    print("\n[4/5b] Training Winner LSTM...")
    lstm = WinnerDraftLSTMModel(vocab_size=vocab_size, embedding_dim=64, lstm_units=128)
    lstm.train(X_train, pos_train_2d, sides_train_2d, y_train,
               X_val,   pos_val_2d,   sides_val_2d,   y_val,
               sample_weight=weights, epochs=30, batch_size=256)
    results['lstm'] = lstm.evaluate(X_test, pos_test_2d, sides_test_2d, y_test)
    lstm.save("models/draft_lstm_winner.keras")

    print("\n[5/5] Training Winner XGBoost...")
    xgb_model = WinnerDraftXGBModel(vocab_size=vocab_size, max_depth=6,
                                    learning_rate=0.1, n_estimators=300)
    xgb_model.train(X_train, pos_train, sides_train, y_train,
                    X_val,   pos_val,   sides_val,   y_val,
                    sample_weight=weights)
    results['xgboost'] = xgb_model.evaluate(X_test, pos_test, sides_test, y_test)
    xgb_model.save("models/draft_xgb_winner.pkl")

    print(f"\n{'='*60}\nWinner Model Accuracy — held-out test games\n{'='*60}")
    print(f"{'Model':<12} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10} {'Top-10':<10}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<12} {r['accuracy']:<10.4f} {r['top_3']:<10.4f} "
              f"{r['top_5']:<10.4f} {r['top_10']:<10.4f}")

    with open("models/winner_all_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nModels saved: draft_mlp_winner.keras  draft_lstm_winner.keras  draft_xgb_winner.pkl")


if __name__ == "__main__":
    train_all_winner_models()