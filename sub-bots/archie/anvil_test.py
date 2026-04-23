import anvil.server
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

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


# ============================================================================
# CONFIGURATION
# ============================================================================
ANVIL_UPLINK_KEY = "server_ITF53PIKVGRTZUZKAA55DNTI-WCM4NA4ADARZXVWG"
CNN_MODEL_PATH = "resnet20_final_updated.h5"
TRANSFORMER_MODEL_PATH = "best_transformer.keras"

# ============================================================================
# LOAD MODELS
# ============================================================================
try:
    print("Loading CNN model...")
    cnn_model = tf.keras.models.load_model(
        CNN_MODEL_PATH,
        compile=False
    )

    if cnn_model == None:
        print(f"CNN model failed to load")
    else:
        print(f"CNN model loaded successfully")

    print("Loading Transformer model...")
    transformer_model = tf.keras.models.load_model(
        TRANSFORMER_MODEL_PATH,
        safe_mode=False,
        compile=False,
        custom_objects={
            "PositionalIndex": PositionalIndex,
            "ClassTokenIndex": ClassTokenIndex
        }
    )

    if transformer_model == None:
        print(f"Transformer model failed to load")
    else:
        print(f"Transformer model loaded successfully")

except Exception as e:
    print(f"Error encountered while loading: {e}")
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_valid_moves(board_array):
    board_array = np.array(board_array)
    return [c for c in range(7) if board_array[0][c] == 0]


def predict_move(model, board_array):
    """
    Anvil sends (and model expects):
        +1 = AI
        -1 = Human
         0 = Empty

    Model input shape: (6, 7, 2)
    """

    board_array = np.array(board_array)

    # Safety checks
    assert board_array.shape == (6, 7), "Board must be 6x7"
    assert set(np.unique(board_array)).issubset({-1, 0, 1}), "Invalid board encoding"

    # Build (6, 7, 2) input
    channel_ai = (board_array == 1).astype(np.float32)
    channel_human = (board_array == -1).astype(np.float32)

    board_input = np.stack([channel_ai, channel_human], axis=-1)
    board_input = np.expand_dims(board_input, axis=0)  # (1, 6, 7, 2)

    probs = model.predict(board_input, verbose=0)[0]

    valid_moves = get_valid_moves(board_array)
    if not valid_moves:
        return None

    # Mask illegal moves
    masked_probs = np.full_like(probs, -1e10)
    for c in valid_moves:
        masked_probs[c] = probs[c]

    return int(np.argmax(masked_probs))


# ============================================================================
# ANVIL ENTRY POINT
# ============================================================================

@anvil.server.callable
def get_model_move(selected_model, board):
    """
    Args:
        selected_model: "cnn" or "transformer"
        board (from Anvil):
            +1 = AI
            -1 = Human
             0 = Empty

    Returns:
        column index (0–6)
    """

    try:
        if selected_model == "cnn":
            model = cnn_model

        elif selected_model == "transformer":
            model = transformer_model if transformer_model else cnn_model

        else:
            raise ValueError(f"Unknown model type: {selected_model}")

        col = predict_move(model, board)
        print(f"[{selected_model.upper()}] selected column: {col}")
        return col

    except Exception as e:
        print(f"Error in get_model_move: {e}")
        valid_moves = get_valid_moves(board)
        return valid_moves[0] if valid_moves else 3


# ============================================================================
# SERVER BOOTSTRAP
# ============================================================================

if __name__ == "__main__":
    print("\nCONNECT 4 AI SERVER")
    print("===================")
    print(f"CNN Loaded: {cnn_model is not None}")
    print(f"Transformer Loaded: {transformer_model is not None}")
    print("===================")

    anvil.server.connect(ANVIL_UPLINK_KEY)
    print("Connected to Anvil")
    anvil.server.wait_forever()
