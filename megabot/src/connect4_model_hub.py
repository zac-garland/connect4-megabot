from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


# `src/connect4_model_hub.py` -> megabot/ (models, data) -> repo root (sub-bots/)
_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = MODELS_DIR / "manifest.csv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str
    source_url: str
    local_fallback: str | None
    original_format: str
    role: str
    notes: str


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        name="archie_resnet",
        source="GitHub sub-bots/archie",
        source_url="https://raw.githubusercontent.com/zac-garland/connect4-megabot/main/sub-bots/archie/resnet20_final_updated.h5",
        local_fallback="connect4-megabot/sub-bots/archie/resnet20_final_updated.h5",
        original_format="h5",
        role="opponent_pool",
        notes="ResNet-style CNN.",
    ),
    ModelSpec(
        name="archie_transformer",
        source="GitHub sub-bots/archie",
        source_url="https://raw.githubusercontent.com/zac-garland/connect4-megabot/main/sub-bots/archie/best_transformer.keras",
        local_fallback="connect4-megabot/sub-bots/archie/best_transformer.keras",
        original_format="keras",
        role="opponent_pool",
        notes="Transformer with custom positional layers.",
    ),
    ModelSpec(
        name="dean_cnn",
        source="GitHub sub-bots/dean",
        source_url="https://raw.githubusercontent.com/zac-garland/connect4-megabot/main/sub-bots/dean/connect-4/my_connect4_cnn.h5",
        local_fallback="connect4-megabot/sub-bots/dean/connect-4/my_connect4_cnn.h5",
        original_format="h5",
        role="m1_best_model",
        notes="Current strongest benchmark bot. Use as M1.",
    ),
    ModelSpec(
        name="dean_transformer",
        source="GitHub sub-bots/dean",
        source_url="https://raw.githubusercontent.com/zac-garland/connect4-megabot/main/sub-bots/dean/connect-4/my_connect4_transformer.keras",
        local_fallback="connect4-megabot/sub-bots/dean/connect-4/my_connect4_transformer.keras",
        original_format="keras",
        role="opponent_pool",
        notes="Dean transformer checkpoint.",
    ),
    ModelSpec(
        name="zac_cnn_best",
        source="GitHub sub-bots/zac",
        source_url="https://raw.githubusercontent.com/zac-garland/connect4-megabot/main/sub-bots/zac/connect-4/cnn_improved_best.keras",
        local_fallback="connect4-megabot/sub-bots/zac/connect-4/cnn_improved_best.keras",
        original_format="keras",
        role="opponent_pool",
        notes="Best validation checkpoint.",
    ),
    ModelSpec(
        name="zac_cnn_final",
        source="GitHub sub-bots/zac",
        source_url="https://raw.githubusercontent.com/zac-garland/connect4-megabot/main/sub-bots/zac/connect-4/cnn_improved_final.keras",
        local_fallback="connect4-megabot/sub-bots/zac/connect-4/cnn_improved_final.keras",
        original_format="keras",
        role="opponent_pool",
        notes="Final saved CNN model.",
    ),
    ModelSpec(
        name="connor_cnn",
        source="GitHub sub-bots/connor",
        source_url="https://raw.githubusercontent.com/zac-garland/connect4-megabot/main/sub-bots/connor/connect4_cnn_model_full_v29.h5",
        local_fallback="connect4-megabot/sub-bots/connor/connect4_cnn_model_full_v29.h5",
        original_format="h5",
        role="opponent_pool",
        notes="Connor CNN model activated after GitHub update.",
    ),
]


@dataclass
class LoadedBot:
    name: str
    model: Any
    model_path: Path
    source: str
    input_shape: tuple[Any, ...]
    role: str
    notes: str

    def _masked_scores(self, board: np.ndarray, player: int) -> np.ndarray:
        encoded = encode_board_for_model(board, player, self.input_shape)
        raw = self.model(encoded, training=False)
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        scores = np.asarray(raw, dtype=np.float64).reshape(-1)
        legal = legal_moves(board)
        masked = np.full(scores.shape, -1e18, dtype=np.float64)
        masked[legal] = scores[legal]
        return masked

    def select_move(self, board: np.ndarray, player: int) -> int:
        return int(np.argmax(self._masked_scores(board, player)))

    def sample_move(self, board: np.ndarray, player: int, temperature: float = 1.0) -> int:
        masked = self._masked_scores(board, player)
        legal = legal_moves(board)
        legal_scores = masked[legal]
        scaled = legal_scores / max(temperature, 1e-6)
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs = probs / probs.sum()
        return int(np.random.choice(legal, p=probs))


def _require_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras import layers

    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass
    return tf, layers


def custom_objects() -> dict[str, Any]:
    tf, layers = _require_tensorflow()

    class PositionalIndex(layers.Layer):
        def call(self, x):
            bs = tf.shape(x)[0]
            n = tf.shape(x)[1]
            idx = tf.range(n)
            return tf.tile(idx[None, :], [bs, 1])

    class ClassTokenIndex(layers.Layer):
        def call(self, x):
            bs = tf.shape(x)[0]
            return tf.zeros((bs, 1), dtype=tf.int32)

    class AddCLSToken(layers.Layer):
        def __init__(self, d_model, **kwargs):
            super().__init__(**kwargs)
            self.d_model = d_model

        def build(self, input_shape):
            self.cls = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.d_model),
                initializer="random_normal",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, x):
            batch_size = tf.shape(x)[0]
            cls_tokens = tf.tile(self.cls, [batch_size, 1, 1])
            return tf.concat([cls_tokens, x], axis=1)

        def get_config(self):
            config = super().get_config()
            config.update({"d_model": self.d_model})
            return config

    class TransformerBlock(layers.Layer):
        def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_ff = d_ff
            self.dropout_rate = dropout
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
            self.mha = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                dropout=dropout,
            )
            self.ffn = tf.keras.Sequential(
                [
                    layers.Dense(d_ff, activation=tf.nn.gelu),
                    layers.Dropout(dropout),
                    layers.Dense(d_model),
                ]
            )
            self.dropout1 = layers.Dropout(dropout)
            self.dropout2 = layers.Dropout(dropout)

        def call(self, x, training=None):
            normed = self.norm1(x)
            attn_output = self.mha(normed, normed, training=training)
            attn_output = self.dropout1(attn_output, training=training)
            x = x + attn_output
            normed = self.norm2(x)
            ffn_output = self.ffn(normed, training=training)
            ffn_output = self.dropout2(ffn_output, training=training)
            return x + ffn_output

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "d_model": self.d_model,
                    "num_heads": self.num_heads,
                    "d_ff": self.d_ff,
                    "dropout": self.dropout_rate,
                }
            )
            return config

    class FocalLoss(tf.keras.losses.Loss):
        def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss"):
            super().__init__(name=name)
            self.gamma = gamma
            self.alpha = alpha

        def call(self, y_true, y_pred):
            y_pred_prob = tf.nn.softmax(y_pred, axis=-1)
            y_true = tf.cast(y_true, tf.int32)
            batch_size = tf.shape(y_true)[0]
            indices = tf.stack([tf.range(batch_size), y_true], axis=1)
            p_t = tf.gather_nd(y_pred_prob, indices)
            focal_weight = tf.pow(1.0 - p_t, self.gamma)
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_true,
                logits=y_pred,
            )
            return tf.reduce_mean(self.alpha * focal_weight * ce)

        def get_config(self):
            config = super().get_config()
            config.update({"gamma": self.gamma, "alpha": self.alpha})
            return config

    return {
        "PositionalIndex": PositionalIndex,
        "ClassTokenIndex": ClassTokenIndex,
        "AddCLSToken": AddCLSToken,
        "TransformerBlock": TransformerBlock,
        "FocalLoss": FocalLoss,
    }


def _load_model(path: Path):
    tf, _ = _require_tensorflow()
    return tf.keras.models.load_model(
        path,
        compile=False,
        safe_mode=False,
        custom_objects=custom_objects(),
    )


def _download_or_copy(spec: ModelSpec, destination: Path, prefer_download: bool = True) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if prefer_download:
        try:
            urlretrieve(spec.source_url, destination)
            return "downloaded"
        except Exception:
            pass

    if spec.local_fallback:
        rel = spec.local_fallback
        if rel.startswith("connect4-megabot/"):
            rel = rel[len("connect4-megabot/"):]
        fallback_path = REPO_ROOT / rel
        if fallback_path.exists():
            shutil.copy2(fallback_path, destination)
            return "copied_local_fallback"

    raise FileNotFoundError(
        f"Could not fetch model '{spec.name}' from GitHub or local fallback."
    )


def import_and_standardize_models(
    models_dir: Path | None = None,
    prefer_download: bool = True,
    overwrite: bool = True,
) -> pd.DataFrame:
    models_dir = Path(models_dir or MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = models_dir / "_tmp_imports"
    temp_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        temp_path = temp_dir / f"{spec.name}.{spec.original_format}"
        fetch_mode = _download_or_copy(spec, temp_path, prefer_download=prefer_download)
        standardized_path = models_dir / f"{spec.name}.keras"

        if overwrite or not standardized_path.exists():
            model = _load_model(temp_path)
            model.save(standardized_path)
            input_shape = tuple(model.input_shape)
        else:
            model = _load_model(standardized_path)
            input_shape = tuple(model.input_shape)

        records.append(
            {
                "name": spec.name,
                "source": spec.source,
                "source_url": spec.source_url,
                "role": spec.role,
                "notes": spec.notes,
                "fetch_mode": fetch_mode,
                "original_format": spec.original_format,
                "standardized_format": "keras",
                "standardized_path": str(standardized_path.resolve()),
                "input_shape": json.dumps(list(input_shape)),
            }
        )

    shutil.rmtree(temp_dir, ignore_errors=True)
    manifest = pd.DataFrame(records).sort_values("name").reset_index(drop=True)
    manifest.to_csv(models_dir / "manifest.csv", index=False)
    return manifest


def load_manifest(models_dir: Path | None = None) -> pd.DataFrame:
    models_dir = Path(models_dir or MODELS_DIR)
    manifest_path = models_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Model manifest not found at {manifest_path}. Run the import notebook first."
        )
    return pd.read_csv(manifest_path)


def load_standardized_bots(models_dir: Path | None = None) -> list[LoadedBot]:
    manifest = load_manifest(models_dir)
    bots: list[LoadedBot] = []
    for row in manifest.to_dict(orient="records"):
        model_path = Path(row["standardized_path"])
        model = _load_model(model_path)
        bots.append(
            LoadedBot(
                name=row["name"],
                model=model,
                model_path=model_path,
                source=row["source"],
                input_shape=tuple(model.input_shape),
                role=row["role"],
                notes=row["notes"],
            )
        )
    return bots


def legal_moves(board: np.ndarray) -> list[int]:
    return [col for col in range(7) if board[0, col] == 0]


def apply_move(board: np.ndarray, col: int, player: int) -> np.ndarray:
    next_board = board.copy()
    for row in range(5, -1, -1):
        if next_board[row, col] == 0:
            next_board[row, col] = player
            return next_board
    raise ValueError(f"Column {col} is full.")


def check_winner(board: np.ndarray) -> int | None:
    for row in range(6):
        for col in range(7):
            player = board[row, col]
            if player == 0:
                continue
            if col <= 3 and np.all(board[row, col : col + 4] == player):
                return int(player)
            if row <= 2 and np.all(board[row : row + 4, col] == player):
                return int(player)
            if row <= 2 and col <= 3 and all(board[row + i, col + i] == player for i in range(4)):
                return int(player)
            if row >= 3 and col <= 3 and all(board[row - i, col + i] == player for i in range(4)):
                return int(player)
    return None


def encode_board_for_model(
    board: np.ndarray,
    player: int,
    input_shape: tuple[Any, ...],
) -> np.ndarray:
    perspective = (board * player).astype(np.float32)
    dims = tuple(input_shape[1:])

    if dims == (6, 7, 2):
        channels = np.zeros((1, 6, 7, 2), dtype=np.float32)
        channels[0, :, :, 0] = perspective == 1
        channels[0, :, :, 1] = perspective == -1
        return channels

    if dims == (6, 7, 1):
        return perspective.reshape(1, 6, 7, 1)

    if dims == (42, 2):
        flat = perspective.reshape(42)
        tokens = np.zeros((1, 42, 2), dtype=np.float32)
        tokens[0, :, 0] = flat == 1
        tokens[0, :, 1] = flat == -1
        return tokens

    if dims == (42,):
        return perspective.reshape(1, 42)

    raise ValueError(f"Unsupported model input shape: {input_shape}")


def play_single_game(
    first_bot: LoadedBot,
    second_bot: LoadedBot,
    rng: random.Random,
    opening_random_moves: int = 2,
) -> dict[str, Any]:
    board = np.zeros((6, 7), dtype=np.int8)
    owners = {1: first_bot, -1: second_bot}
    current_player = 1
    move_count = 0

    while True:
        legal = legal_moves(board)
        if not legal:
            return {"winner": None, "moves": move_count, "final_board": board}

        if move_count < opening_random_moves:
            col = rng.choice(legal)
        else:
            col = owners[current_player].select_move(board, current_player)
            if col not in legal:
                col = legal[0]

        board = apply_move(board, col, current_player)
        move_count += 1

        winner = check_winner(board)
        if winner is not None:
            return {
                "winner": owners[winner].name,
                "moves": move_count,
                "final_board": board,
            }

        current_player *= -1


def play_head_to_head(
    bot_a: LoadedBot,
    bot_b: LoadedBot,
    games: int = 100,
    opening_random_moves: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    bot_a_wins = 0
    bot_b_wins = 0
    draws = 0
    move_counts: list[int] = []

    for game_idx in range(games):
        if game_idx % 2 == 0:
            result = play_single_game(bot_a, bot_b, rng, opening_random_moves)
        else:
            result = play_single_game(bot_b, bot_a, rng, opening_random_moves)

        winner = result["winner"]
        move_counts.append(result["moves"])
        if winner == bot_a.name:
            bot_a_wins += 1
        elif winner == bot_b.name:
            bot_b_wins += 1
        else:
            draws += 1

    return {
        "bot_a": bot_a.name,
        "bot_b": bot_b.name,
        "games": games,
        "opening_random_moves": opening_random_moves,
        "bot_a_wins": bot_a_wins,
        "bot_b_wins": bot_b_wins,
        "draws": draws,
        "bot_a_win_rate": bot_a_wins / games,
        "bot_b_win_rate": bot_b_wins / games,
        "draw_rate": draws / games,
        "bot_a_score_rate": (bot_a_wins + 0.5 * draws) / games,
        "bot_b_score_rate": (bot_b_wins + 0.5 * draws) / games,
        "avg_moves": float(np.mean(move_counts)),
    }


def run_round_robin(
    bots: list[LoadedBot],
    games_per_pair: int = 100,
    opening_random_moves: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for pair_idx, (bot_a, bot_b) in enumerate(combinations(bots, 2)):
        rows.append(
            play_head_to_head(
                bot_a,
                bot_b,
                games=games_per_pair,
                opening_random_moves=opening_random_moves,
                seed=seed + pair_idx,
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["bot_a_score_rate", "bot_a_win_rate", "bot_a", "bot_b"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def leaderboard_from_results(results: pd.DataFrame, bots: list[LoadedBot]) -> pd.DataFrame:
    totals = {
        bot.name: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "score": 0.0,
        }
        for bot in bots
    }

    for row in results.to_dict(orient="records"):
        a = totals[row["bot_a"]]
        b = totals[row["bot_b"]]
        a["games"] += row["games"]
        b["games"] += row["games"]
        a["wins"] += row["bot_a_wins"]
        b["wins"] += row["bot_b_wins"]
        a["losses"] += row["bot_b_wins"]
        b["losses"] += row["bot_a_wins"]
        a["draws"] += row["draws"]
        b["draws"] += row["draws"]
        a["score"] += row["bot_a_wins"] + 0.5 * row["draws"]
        b["score"] += row["bot_b_wins"] + 0.5 * row["draws"]

    leaderboard = pd.DataFrame.from_dict(totals, orient="index").reset_index(names="bot")
    leaderboard["win_rate"] = leaderboard["wins"] / leaderboard["games"]
    leaderboard["score_rate"] = leaderboard["score"] / leaderboard["games"]
    return leaderboard.sort_values(["score_rate", "win_rate"], ascending=False).reset_index(drop=True)


def pairwise_score_matrix(results: pd.DataFrame, bots: list[LoadedBot]) -> pd.DataFrame:
    names = [bot.name for bot in bots]
    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for name in names:
        matrix.loc[name, name] = 1.0

    for row in results.to_dict(orient="records"):
        matrix.loc[row["bot_a"], row["bot_b"]] = row["bot_a_score_rate"]
        matrix.loc[row["bot_b"], row["bot_a"]] = row["bot_b_score_rate"]
    return matrix


def save_round_robin_outputs(
    results: pd.DataFrame,
    bots: list[LoadedBot],
    data_dir: Path | None = None,
) -> tuple[Path, Path, Path, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir or DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    results_path = data_dir / "round_robin_results.csv"
    leaderboard_path = data_dir / "round_robin_leaderboard.csv"
    matrix_path = data_dir / "round_robin_score_matrix.csv"

    leaderboard = leaderboard_from_results(results, bots)
    matrix = pairwise_score_matrix(results, bots)

    results.to_csv(results_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)
    matrix.to_csv(matrix_path)
    return results_path, leaderboard_path, matrix_path, leaderboard, matrix


def describe_available_models(models_dir: Path | None = None) -> pd.DataFrame:
    manifest = load_manifest(models_dir)
    return manifest[["name", "role", "source", "original_format", "standardized_format", "notes"]].copy()

