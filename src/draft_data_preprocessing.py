"""
Data preprocessing for draft sequence prediction.

DraftSequenceProcessor   — builds full 20-step sequences for all games
WinnerSequenceProcessor  — subclass: filters to winning side only,
                           adds side indicator for winner-guided training
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder
import pickle

DRAFT_ORDER = [
    ('Blue', 'ban', 1), ('Red', 'ban', 1),
    ('Blue', 'ban', 2), ('Red', 'ban', 2),
    ('Blue', 'ban', 3), ('Red', 'ban', 3),
    ('Blue', 'pick', 1), ('Red', 'pick', 1),
    ('Red', 'pick', 2), ('Blue', 'pick', 2),
    ('Red', 'ban', 4), ('Blue', 'ban', 4),
    ('Red', 'ban', 5), ('Blue', 'ban', 5),
    ('Red', 'pick', 3), ('Blue', 'pick', 3),
    ('Blue', 'pick', 4), ('Red', 'pick', 4),
    ('Red', 'pick', 5), ('Blue', 'pick', 5),
]

BLUE_SIDE = 0
RED_SIDE  = 1


class DraftSequenceProcessor:
    def __init__(self, csv_path: str):
        self.csv_path         = csv_path
        self.champion_encoder = LabelEncoder()
        self.n_champions      = 0
        self.pad_token        = 0
        self.start_token      = 1
        self.champion_offset  = 2

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, low_memory=False)
        return df[df['participantid'].isin([100, 200])].copy()

    def fit_encoder(self, df: pd.DataFrame):
        all_champions = set()
        for col in [f'ban{i}' for i in range(1, 6)] + [f'pick{i}' for i in range(1, 6)]:
            all_champions.update(df[col].dropna().unique())
        self.champion_encoder.fit(sorted(all_champions))
        self.n_champions = len(self.champion_encoder.classes_)
        self.vocab_size  = self.n_champions + self.champion_offset
        print(f"Vocab: {self.n_champions} champions + 2 special = {self.vocab_size}")

    def encode_champion(self, name: str) -> int:
        if pd.isna(name) or name not in self.champion_encoder.classes_:
            return self.pad_token
        return int(self.champion_encoder.transform([name])[0]) + self.champion_offset

    def _parse_game(self, group) -> Tuple[bool, object, object]:
        """Return (valid, blue_row, red_row) for a game group."""
        if len(group) != 2:
            return False, None, None
        blue = group[group['side'] == 'Blue']
        red  = group[group['side'] == 'Red']
        if len(blue) != 1 or len(red) != 1:
            return False, None, None
        return True, blue.iloc[0], red.iloc[0]

    def _build_sequence(self, blue, red):
        """Build token sequence from blue/red rows. Returns (seq, valid)."""
        seq = []
        for side, action, idx in DRAFT_ORDER:
            token = self.encode_champion((blue if side == 'Blue' else red)[f'{action}{idx}'])
            if token == self.pad_token:
                return [], False
            seq.append(token)
        return seq, True

    def build_draft_sequences(self, df: pd.DataFrame) -> List[np.ndarray]:
        sequences, skipped = [], 0
        for _, group in df.groupby('gameid'):
            valid, blue, red = self._parse_game(group)
            if not valid:
                skipped += 1; continue
            seq, ok = self._build_sequence(blue, red)
            if ok:
                sequences.append(np.array(seq, dtype=np.int32))
            else:
                skipped += 1
        print(f"Built {len(sequences)} sequences (skipped {skipped})")
        return sequences

    def save_metadata(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'champion_encoder': self.champion_encoder,
                'n_champions':      self.n_champions,
                'vocab_size':       self.vocab_size,
                'pad_token':        self.pad_token,
                'start_token':      self.start_token,
                'champion_offset':  self.champion_offset,
                'draft_order':      DRAFT_ORDER,
            }, f)
        print(f"Metadata saved to {path}")


class WinnerSequenceProcessor(DraftSequenceProcessor):
    """
    Subclass of DraftSequenceProcessor.
    Builds winner-only sequences paired with the winning side indicator,
    and creates training examples only for the winning side's turns.
    """

    def build_winner_sequences(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, int]]:
        result, skipped = [], 0
        for _, group in df.groupby('gameid'):
            valid, blue, red = self._parse_game(group)
            if not valid:
                skipped += 1; continue
            winning_side = BLUE_SIDE if int(blue['result']) == 1 else RED_SIDE
            seq, ok = self._build_sequence(blue, red)
            if ok:
                result.append((np.array(seq, dtype=np.int32), winning_side))
            else:
                skipped += 1
        blue_wins = sum(1 for _, s in result if s == BLUE_SIDE)
        print(f"Built {len(result)} winner sequences (skipped {skipped})")
        print(f"  Blue wins: {blue_wins}  Red wins: {len(result) - blue_wins}")
        return result

    def create_winner_examples(self, winner_sequences: List[Tuple[np.ndarray, int]]):
        X_list, pos_list, side_list, y_list = [], [], [], []
        for seq, winning_side in winner_sequences:
            winning_side_str = 'Blue' if winning_side == BLUE_SIDE else 'Red'
            for t, (step_side, _, _) in enumerate(DRAFT_ORDER):
                if step_side != winning_side_str:
                    continue
                input_seq        = np.full(20, self.pad_token, dtype=np.int32)
                input_seq[0]     = self.start_token
                input_seq[1:t+1] = seq[:t]
                X_list.append(input_seq)
                pos_list.append(t)
                side_list.append(winning_side)
                y_list.append(seq[t])
        X         = np.array(X_list,    dtype=np.int32)
        positions = np.array(pos_list,  dtype=np.int32)
        sides     = np.array(side_list, dtype=np.int32)
        y         = np.array(y_list,    dtype=np.int32)
        print(f"Created {len(X)} winner examples  "
              f"(Blue: {(sides==BLUE_SIDE).sum()}  Red: {(sides==RED_SIDE).sum()})")
        return X, positions, sides, y

    def save_metadata(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'champion_encoder': self.champion_encoder,
                'n_champions':      self.n_champions,
                'vocab_size':       self.vocab_size,
                'pad_token':        self.pad_token,
                'start_token':      self.start_token,
                'champion_offset':  self.champion_offset,
                'draft_order':      DRAFT_ORDER,
                'blue_side':        BLUE_SIDE,
                'red_side':         RED_SIDE,
            }, f)
        print(f"Metadata saved to {path}")