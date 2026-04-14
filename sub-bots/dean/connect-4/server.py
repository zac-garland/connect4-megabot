import anvil.server
import numpy as np
import tensorflow as tf

class PositionalIndex(tf.keras.layers.Layer):
    def call(self, x):
        bs = tf.shape(x)[0]
        number_of_vectors = tf.shape(x)[1]
        indices = tf.range(number_of_vectors)
        indices = tf.expand_dims(indices, 0)
        return tf.tile(indices, [bs, 1])

class ClassTokenIndex(tf.keras.layers.Layer):
    def call(self, x):
        bs = tf.shape(x)[0]
        number_of_vectors = 1
        indices = tf.range(number_of_vectors)
        indices = tf.expand_dims(indices, 0)
        return tf.tile(indices, [bs, 1])

anvil.server.connect("server_SHQWLBKU5BBSGQTFRNRPOQU4-EIUS5H3WO4DL3Z3N")

custom_objects = {
    "PositionalIndex": PositionalIndex,
    "ClassTokenIndex": ClassTokenIndex,
}

cnn_model = tf.keras.models.load_model("my_connect4_cnn.h5")
transformer_model = tf.keras.models.load_model("my_connect4_transformer.keras", custom_objects=custom_objects)

@anvil.server.callable
def get_move(board, model_type):
    board_array = np.array(board)

    if model_type == "cnn":
        board_input = board_array.reshape(1, 6, 7, 1)
        prediction = cnn_model.predict(board_input, verbose=0)
    else:
        tokens = np.zeros((1, 42, 2), dtype=np.float32)
        flat = board_array.reshape(42)
        tokens[0, :, 0] = (flat == 1).astype(np.float32)
        tokens[0, :, 1] = (flat == -1).astype(np.float32)
        prediction = transformer_model.predict(tokens, verbose=0)

    move_probs = prediction[0]

    for col in range(7):
        if board_array[0][col] != 0:
            move_probs[col] = -1

    best_move = int(np.argmax(move_probs))
    return best_move

anvil.server.wait_forever()
