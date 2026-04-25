"""
Train win probability models for LoL draft prediction.
5-fold cross-validation then final 70/15/15 training.
"""
import numpy as np
import sys
sys.path.append('src')

from data_preprocessing import DraftDataProcessor
from models_win_predictor import (AlwaysBlueBaseline, LogisticDraftModel,
                                  XGBoostDraftModel, NeuralNetworkDraftModel)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import pickle


def cross_validate(processor, game_df, n_folds=5):
    print(f"\n{'='*60}\nRunning {n_folds}-Fold Cross-Validation\n{'='*60}")
    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_all = game_df['blue_win'].values
    cv    = {m: {'accuracy': [], 'auc': [], 'log_loss': []}
             for m in ['logistic', 'xgboost', 'neural_network']}

    for fold, (train_idx, test_idx) in enumerate(skf.split(game_df, y_all), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        train_df = game_df.iloc[train_idx]
        test_df  = game_df.iloc[test_idx]
        train_sub, val_df = train_test_split(
            train_df, test_size=0.15, random_state=42, stratify=train_df['blue_win'])

        X_tr,      y_tr      = processor.encode_features(train_sub)
        X_val,     y_val     = processor.encode_features(val_df)
        X_te,      y_te      = processor.encode_features(test_df)
        X_tr_full, y_tr_full = processor.encode_features(train_df)

        for name, model, Xf, yf in [
            ('logistic',       LogisticDraftModel(),      X_tr_full, y_tr_full),
            ('xgboost',        XGBoostDraftModel(),       X_tr,      y_tr),
            ('neural_network', NeuralNetworkDraftModel(), X_tr,      y_tr),
        ]:
            if name == 'xgboost':
                model.train(Xf, yf, X_val, y_val)
            elif name == 'neural_network':
                model.train(Xf, yf, X_val, y_val, epochs=50, batch_size=128)
            else:
                model.train(Xf, yf)
            y_pred  = model.predict(X_te)
            y_proba = model.predict_proba(X_te)
            y_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            cv[name]['accuracy'].append(accuracy_score(y_te, y_pred))
            cv[name]['auc'].append(roc_auc_score(y_te, y_proba))
            cv[name]['log_loss'].append(log_loss(y_te, y_proba))

        print(f"  LR: {cv['logistic']['accuracy'][-1]:.4f}  "
              f"XGB: {cv['xgboost']['accuracy'][-1]:.4f}  "
              f"NN: {cv['neural_network']['accuracy'][-1]:.4f}")

    print(f"\n{'='*60}\nCross-Validation Results (mean ± std)\n{'='*60}")
    print(f"{'Model':<18} {'Accuracy':<22} {'AUC':<22} {'Log Loss'}")
    print("-" * 80)
    for name, m in cv.items():
        acc, auc, ll = map(np.array, [m['accuracy'], m['auc'], m['log_loss']])
        print(f"{name:<18} {acc.mean():.4f} ± {acc.std():.4f}    "
              f"{auc.mean():.4f} ± {auc.std():.4f}    "
              f"{ll.mean():.4f} ± {ll.std():.4f}")
    return cv


def train_final_models(processor, game_df, metadata):
    print(f"\n{'='*60}\nTraining Final Models  (70/15/15)\n{'='*60}")
    train_df, temp_df = train_test_split(
        game_df, test_size=0.3, random_state=42, stratify=game_df['blue_win'])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['blue_win'])
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    X_train, y_train = processor.encode_features(train_df)
    X_val,   y_val   = processor.encode_features(val_df)
    X_test,  y_test  = processor.encode_features(test_df)

    metadata['n_features'] = X_train.shape[1]
    with open("data/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    results = {}

    print("\n--- Logistic Regression ---")
    lr = LogisticDraftModel()
    lr.train(X_train, y_train)
    results['logistic'] = lr.evaluate(X_test, y_test)
    lr.save("models/logistic_model.pkl")

    print("\n--- XGBoost ---")
    xgb = XGBoostDraftModel()
    xgb.train(X_train, y_train, X_val, y_val)
    results['xgboost'] = xgb.evaluate(X_test, y_test)
    xgb.save("models/xgboost_model.pkl")

    print("\n--- Neural Network ---")
    nn = NeuralNetworkDraftModel()
    nn.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=128)
    results['neural_network'] = nn.evaluate(X_test, y_test)
    nn.save("models/neural_network_model.keras")

    print("\n--- Always-Blue Baseline ---")
    baseline = AlwaysBlueBaseline()
    baseline.train(X_train, y_train)
    results['always_blue_baseline'] = baseline.evaluate(X_test, y_test)

    print(f"\n{'='*60}\nFinal Results (holdout test set)\n{'='*60}")
    print(f"{'Model':<22} {'Accuracy':<12} {'AUC':<12} {'Log Loss'}")
    print("-" * 58)
    for name, m in results.items():
        print(f"{name:<22} {m['accuracy']:<12.4f} {m['auc']:<12.4f} {m['log_loss']:.4f}")

    with open("models/results_summary.pkl", "wb") as f:
        pickle.dump(results, f)
    return results


def main():
    print("=" * 60)
    print("LoL Win Probability — Model Training")
    print("=" * 60)

    print("\n[1/3] Preprocessing...")
    processor = DraftDataProcessor("src/Split2_2024.csv")
    game_df, metadata = processor.process_full_pipeline()

    print("\n[2/3] Cross-validation...")
    cross_validate(processor, game_df)

    print("\n[3/3] Final training...")
    train_final_models(processor, game_df, metadata)

    print(f"\n{'='*60}\nAll done!\n{'='*60}")


if __name__ == "__main__":
    main()