"""
Data preprocessing for League of Legends win probability prediction.

Feature design
--------------
(default, ~2040 features)
    - Blue slot one-hots : 5 x (n_champions,) — one vector per pick slot
    - Red  slot one-hots : 5 x (n_champions,)
    - Blue presence      : (n_champions,) — 1 for each Blue champion
    - Red  presence      : (n_champions,) — 1 for each Red  champion
    - Patch              : (n_patches,)   — one-hot
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import LabelEncoder
import pickle


class DraftDataProcessor:
    def __init__(self, csv_path: str):
        self.csv_path         = csv_path
        self.champion_encoder = LabelEncoder()
        self.patch_encoder    = LabelEncoder()

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, low_memory=False)
        return df[df['participantid'].isin([100, 200])].copy()

    def build_game_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        games = []
        for game_id, group in df.groupby('gameid'):
            if len(group) != 2:
                continue
            blue = group[group['side'] == 'Blue']
            red  = group[group['side'] == 'Red']
            if len(blue) != 1 or len(red) != 1:
                continue
            blue = blue.iloc[0]
            red  = red.iloc[0]

            row = {'gameid': game_id}
            for i in range(1, 6):
                row[f'blue_pick{i}'] = blue[f'pick{i}']
                row[f'red_pick{i}']  = red[f'pick{i}']
            row['patch']    = blue['patch'] if pd.notna(blue['patch']) else red['patch']
            row['blue_win'] = int(blue['result'])
            games.append(row)

        return pd.DataFrame(games)

    def fit_encoders(self, game_df: pd.DataFrame):
        all_champions = set()
        for col in [f'{side}_pick{i}' for side in ['blue', 'red'] for i in range(1, 6)]:
            all_champions.update(game_df[col].dropna().unique())
        self.champion_encoder.fit(sorted(all_champions))
        self.patch_encoder.fit(sorted(game_df['patch'].dropna().unique()))

    def encode_features(self, game_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode each game into a feature vector.
            [blue_slot1 | ... | blue_slot5 | red_slot1 | ... | red_slot5
             | blue_presence | red_presence | patch]
            Shape: (n_games, 12*n_champs + n_patches)
        """
        n_champs  = len(self.champion_encoder.classes_)
        n_patches = len(self.patch_encoder.classes_)
        n_games   = len(game_df)

        slot_cols = 10
        n_features = (slot_cols + 2) * n_champs + n_patches
        X = np.zeros((n_games, n_features), dtype=np.float32)

        champ_idx = {c: i for i, c in enumerate(self.champion_encoder.classes_)}
        patch_idx = {p: i for i, p in enumerate(self.patch_encoder.classes_)}

        for i, (_, row) in enumerate(game_df.iterrows()):
            offset = 0

            # Blue slot one-hots
            for j in range(1, 6):
                champ = row[f'blue_pick{j}']
                if pd.notna(champ) and champ in champ_idx:
                    X[i, offset + champ_idx[champ]] = 1
                offset += n_champs

            # Red slot one-hots
            for j in range(1, 6):
                champ = row[f'red_pick{j}']
                if pd.notna(champ) and champ in champ_idx:
                    X[i, offset + champ_idx[champ]] = 1
                offset += n_champs

            # Blue presence
            for j in range(1, 6):
                champ = row[f'blue_pick{j}']
                if pd.notna(champ) and champ in champ_idx:
                    X[i, offset + champ_idx[champ]] = 1
            offset += n_champs

            # Red presence
            for j in range(1, 6):
                champ = row[f'red_pick{j}']
                if pd.notna(champ) and champ in champ_idx:
                    X[i, offset + champ_idx[champ]] = 1
            offset += n_champs

            # Patch one-hot
            patch = row['patch']
            if pd.notna(patch) and patch in patch_idx:
                X[i, offset + patch_idx[patch]] = 1

        y = game_df['blue_win'].values.astype(np.int32)
        return X, y

    def process_full_pipeline(self) -> Tuple[pd.DataFrame, Dict]:
        print("Loading data...")
        df = self.load_data()
        print(f"  {len(df)} team rows from {df['gameid'].nunique()} games")

        print("Building game rows...")
        game_df = self.build_game_rows(df)
        print(f"  {len(game_df)} game rows")

        self.fit_encoders(game_df)

        n_champs  = len(self.champion_encoder.classes_)
        n_patches = len(self.patch_encoder.classes_)
        print(f"  Champions: {n_champs}  Patches: {n_patches}")
        print(f"  Feature size (with slots):    {12*n_champs + n_patches}")
        print(f"  Feature size (without slots): {2*n_champs  + n_patches}")

        metadata = {
            'champion_encoder': self.champion_encoder,
            'patch_encoder':    self.patch_encoder,
            'n_champions':      n_champs,
            'n_patches':        n_patches,
        }
        return game_df, metadata

    def save_metadata(self, metadata: Dict, path: str):
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {path}")