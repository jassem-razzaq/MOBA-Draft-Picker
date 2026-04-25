"""
Adversarial evaluation for the winner-guided draft models.

The single model plays both sides simultaneously:
  - On Blue-side turns it receives side=0 and picks as Blue
  - On Red-side turns  it receives side=1 and picks as Red

Each side is trying to replicate winning behaviour from its own
perspective. The LR model scores the finished 5v5 at the end.

Comparisons:
  1. Winner model Blue vs Winner model Red  — head-to-head
  2. Winner model Blue vs Random Red        — Blue quality in isolation
  3. Random Blue vs Winner model Red        — Red quality in isolation
  4. Random vs Random                       — baseline
"""
import numpy as np
import pickle
import sys
import os
sys.path.append('src')

from models_draft import WinnerDraftMLPModel, WinnerDraftLSTMModel, WinnerDraftXGBModel
from models_win_predictor import LogisticDraftModel
from draft_data_preprocessing import DRAFT_ORDER, BLUE_SIDE, RED_SIDE


class WinnerAdversarialEvaluator:
    def __init__(self, lr_model_path, wp_metadata_path, draft_metadata_path):
        self.lr_model = LogisticDraftModel()
        self.lr_model.load(lr_model_path)

        with open(wp_metadata_path, 'rb') as f:
            wp = pickle.load(f)
        self.n_wp_champs   = len(wp['champion_encoder'].classes_)
        self.n_wp_patches  = len(wp['patch_encoder'].classes_)
        self._wp_idx       = {c: i for i, c in enumerate(wp['champion_encoder'].classes_)}
        self._wp_set       = set(wp['champion_encoder'].classes_)
        self._wp_champ_enc = wp['champion_encoder']
        self._patch_enc    = wp['patch_encoder']
        # Read include_slots from metadata so feature encoding matches
        # how the LR model was trained
        self.include_slots = wp.get('include_slots', True)
        print(f"  include_slots={self.include_slots}  "
              f"n_champs={self.n_wp_champs}  n_patches={self.n_wp_patches}")

        with open(draft_metadata_path, 'rb') as f:
            dm = pickle.load(f)
        self.draft_champ_encoder = dm['champion_encoder']
        self.vocab_size      = dm['vocab_size']
        self.pad_token       = dm['pad_token']
        self.start_token     = dm['start_token']
        self.champion_offset = dm['champion_offset']
        self._all_champs     = list(dm['champion_encoder'].classes_)

        print("Winner adversarial evaluator initialized")

    def _legal(self, used_names):
        return [c for c in self._all_champs
                if c not in used_names and c in self._wp_set]

    def _winner_model_choose(self, model, input_seq, t, side_id,
                              used_ids, used_names, rng):
        """Call the winner model with the correct side indicator."""
        pos      = np.array([[t]],       dtype=np.int32)
        side_arr = np.array([[side_id]], dtype=np.int32)
        proba    = model.predict_proba(
            input_seq.reshape(1, -1), pos, side_arr)[0].copy()
        proba[self.pad_token]   = 0
        proba[self.start_token] = 0
        for uid in used_ids:
            proba[uid] = 0
        total = proba.sum()
        if total > 0:
            proba /= total
        top_k = np.argsort(proba)[-5:]
        top_p = proba[top_k]
        if top_p.sum() > 0:
            top_p /= top_p.sum()
            return int(rng.choice(top_k, p=top_p))
        legal = self._legal(used_names)
        if legal:
            return self._all_champs.index(rng.choice(legal)) + self.champion_offset
        return self.pad_token

    def _random_choose(self, used_names, rng):
        legal = self._legal(used_names)
        if not legal:
            return self.pad_token
        return self._all_champs.index(rng.choice(legal)) + self.champion_offset

    def _score(self, blue_picks, red_picks):
        if len(blue_picks) != 5 or len(red_picks) != 5:
            return 0.5

        n_c = self.n_wp_champs
        n_p = self.n_wp_patches
        slot_cols = 10 if self.include_slots else 0
        n = (slot_cols + 2) * n_c + n_p
        x = np.zeros(n, dtype=np.float32)
        offset = 0

        if self.include_slots:
            # Blue slot one-hots (5 slots)
            for champ in blue_picks:
                if champ in self._wp_idx:
                    x[offset + self._wp_idx[champ]] = 1
                offset += n_c
            # Red slot one-hots (5 slots)
            for champ in red_picks:
                if champ in self._wp_idx:
                    x[offset + self._wp_idx[champ]] = 1
                offset += n_c

        # Blue presence
        for champ in blue_picks:
            if champ in self._wp_idx:
                x[offset + self._wp_idx[champ]] = 1
        offset += n_c

        # Red presence
        for champ in red_picks:
            if champ in self._wp_idx:
                x[offset + self._wp_idx[champ]] = 1

        return float(self.lr_model.predict_proba(x.reshape(1, -1))[0, 1])

    def _extract_picks(self, seq):
        blue, red = [], []
        for t, (side, action, _) in enumerate(DRAFT_ORDER):
            if action != 'pick':
                continue
            token = int(seq[t])
            if token < self.champion_offset:
                continue
            name = self.draft_champ_encoder.classes_[token - self.champion_offset]
            (blue if side == 'Blue' else red).append(name)
        return blue, red

    def simulate(self, blue_agent, red_agent, rng):
        """
        blue_agent / red_agent: dict with keys:
            'model' : model object or None for random
            'type'  : 'winner' | 'plain' | 'random'
        """
        seq       = np.full(20, self.pad_token, dtype=np.int32)
        input_seq = np.full(20, self.pad_token, dtype=np.int32)
        input_seq[0] = self.start_token
        used_ids   = set()
        used_names = set()

        for t in range(20):
            draft_side = DRAFT_ORDER[t][0]  # 'Blue' or 'Red'
            agent      = blue_agent if draft_side == 'Blue' else red_agent
            side_id    = BLUE_SIDE  if draft_side == 'Blue' else RED_SIDE

            if agent['type'] == 'winner':
                chosen = self._winner_model_choose(
                    agent['model'], input_seq, t, side_id,
                    used_ids, used_names, rng)
            else:
                chosen = self._random_choose(used_names, rng)

            seq[t] = chosen
            used_ids.add(chosen)
            if chosen >= self.champion_offset:
                used_names.add(
                    self.draft_champ_encoder.classes_[chosen - self.champion_offset])
            if t < 19:
                input_seq[t + 1] = chosen

        blue, red = self._extract_picks(seq)
        return self._score(blue, red)

    def evaluate_matchup(self, blue_agent, red_agent, label,
                         n_games=500, random_state=42):
        print(f"\n{'='*60}")
        print(f"{label}  ({n_games} games)")
        print(f"{'='*60}")
        rng   = np.random.RandomState(random_state)
        probs = []
        for i in range(n_games):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Game {i + 1}/{n_games}...")
            probs.append(self.simulate(blue_agent, red_agent, rng))
        probs = np.array(probs)
        r = {
            'label':            label,
            'mean_blue_wp':     float(probs.mean()),
            'blue_favored_pct': float((probs > 0.5).mean() * 100),
            'std':              float(probs.std()),
        }
        print(f"  Mean Blue WP:  {r['mean_blue_wp']:.4f}")
        print(f"  Blue favoured: {r['blue_favored_pct']:.1f}%")
        print(f"  Std:           {r['std']:.4f}")
        return r


def load_winner_model(model_type, vocab_size):
    """Load a winner model by type string."""
    if model_type == 'mlp':
        m = WinnerDraftMLPModel(vocab_size=vocab_size)
        m.load("models/draft_mlp_winner.pt")
    elif model_type == 'lstm':
        m = WinnerDraftLSTMModel(vocab_size=vocab_size)
        m.load("models/draft_lstm_winner.pt")
    elif model_type == 'xgb':
        m = WinnerDraftXGBModel(vocab_size=vocab_size)
        m.load("models/draft_xgb_winner.pkl")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return m


def main():
    print("=" * 60)
    print("Adversarial Evaluation — All Winner-Guided Models")
    print("=" * 60)

    with open("data/draft_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    vocab_size = metadata['vocab_size']

    evaluator = WinnerAdversarialEvaluator(
        lr_model_path       = "models/logistic_model.pkl",
        wp_metadata_path    = "data/metadata.pkl",
        draft_metadata_path = "data/draft_metadata.pkl",
    )

    random_agent = {'model': None, 'type': 'random'}
    results = {}

    # Run evaluations for each available winner model
    for model_type, label in [
        ('mlp',  'MLP'),
        ('lstm', 'LSTM'),
        ('xgb',  'XGBoost'),
    ]:
        path_map = {
            'mlp':  "models/draft_mlp_winner.pt",
            'lstm': "models/draft_lstm_winner.pt",
            'xgb':  "models/draft_xgb_winner.pkl",
        }
        if not os.path.exists(path_map[model_type]):
            print(f"\nSkipping {label} — model not found at {path_map[model_type]}")
            continue

        print(f"\n--- Evaluating Winner {label} ---")
        model = load_winner_model(model_type, vocab_size)
        blue_agent = {'model': model, 'type': 'winner'}
        red_agent  = {'model': model, 'type': 'winner'}

        results[f'{model_type}_vs_{model_type}'] = evaluator.evaluate_matchup(
            blue_agent, red_agent,
            label=f"Winner {label} (Blue) vs Winner {label} (Red)",
        )
        results[f'{model_type}_blue_vs_random'] = evaluator.evaluate_matchup(
            blue_agent, random_agent,
            label=f"Winner {label} (Blue) vs Random Red",
        )
        results[f'random_vs_{model_type}_red'] = evaluator.evaluate_matchup(
            random_agent, red_agent,
            label=f"Random Blue vs Winner {label} (Red)",
        )

    # Random baseline once
    results['random_vs_random'] = evaluator.evaluate_matchup(
        random_agent, random_agent, label="Random vs Random")

    # Cross-model matchups if all three exist
    all_exist = all(os.path.exists(p) for p in [
        "models/draft_mlp_winner.pt",
        "models/draft_lstm_winner.pt",
        "models/draft_xgb_winner.pkl",
    ])
    if all_exist:
        print("\n--- Cross-model matchups ---")
        mlp_model  = load_winner_model('mlp',  vocab_size)
        lstm_model = load_winner_model('lstm', vocab_size)
        xgb_model  = load_winner_model('xgb',  vocab_size)

        for (btype, bmodel, rtype, rmodel) in [
            ('MLP',     mlp_model,  'LSTM',    lstm_model),
            ('MLP',     mlp_model,  'XGBoost', xgb_model),
            ('LSTM',    lstm_model, 'XGBoost', xgb_model),
        ]:
            key = f'{btype.lower()}_blue_vs_{rtype.lower()}_red'
            results[key] = evaluator.evaluate_matchup(
                {'model': bmodel, 'type': 'winner'},
                {'model': rmodel, 'type': 'winner'},
                label=f"Winner {btype} (Blue) vs Winner {rtype} (Red)",
            )

    # Summary
    print(f"\n{'='*70}")
    print("Summary — Blue win probability (>0.5 = Blue advantage)")
    print(f"{'='*70}")
    print(f"{'Matchup':<45} {'Mean WP':<10} {'Blue Fav%':<12} {'Std'}")
    print("-" * 73)
    for r in results.values():
        print(f"{r['label']:<45} {r['mean_blue_wp']:<10.4f} "
              f"{r['blue_favored_pct']:<12.1f} {r['std']:.4f}")

    with open("models/adversarial_winner_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nResults saved to models/adversarial_winner_results.pkl")


if __name__ == "__main__":
    main()