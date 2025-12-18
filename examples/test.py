from keras import Sequential
from keras.layers import Dense, Input

from persistence_reg import NeuralPersistence


def build_model(num_layers, layer_width, batch_size=32, jit_compile="auto"):
    model = Sequential()
    model.add(Input(batch_shape=(batch_size, 2), dtype="float32"), rebuild=False)
    for _ in range(num_layers):
        model.add(
            Dense(
                layer_width,
                activation="relu",
                kernel_regularizer=NeuralPersistence(),
            ),
            rebuild=False,
        )
    model.add(
        Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=NeuralPersistence(),
        )
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
        jit_compile=jit_compile,
    )
    return model

if __name__ == "__main__":
    from sklearn.datasets import make_moons

    x, y = make_moons()
    val_data = make_moons()

    model = build_model(5, 5)

    model.fit(x, y, validation_data=val_data)
