from keras import Sequential
from keras.layers import Dense, Input

from persistence_reg import NeuralPersistence


def build_model(num_layers, layer_width, batch_size=32, jit_compile="auto"):
    model = Sequential()
    model.add(Input(batch_shape=(batch_size, 2), dtype="float32"), rebuild=False)
    model.add(
        Dense(
            layer_width,
            activation="relu",
            kernel_regularizer=NeuralPersistence(2, layer_width),
        ),
        rebuild=False,
    )
    for _ in range(num_layers - 1):
        model.add(
            Dense(
                layer_width,
                activation="relu",
                kernel_regularizer=NeuralPersistence(layer_width, layer_width),
            ),
            rebuild=False,
        )
    model.add(
        Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=NeuralPersistence(layer_width, 1),
        )
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
        jit_compile=jit_compile,
    )
    return model
