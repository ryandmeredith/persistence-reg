from keras import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.datasets import make_circles


def build_model(
    num_layers,
    layer_width,
    jit_compile="auto",
    kernel_regularizer=None,
    learning_rate=0.001,
    batch_size=8,
    epochs=100,
    data_size=100,
    val_split=0.2,
):
    model = Sequential()
    model.add(Input(batch_shape=(batch_size, 2)), rebuild=False)
    for _ in range(num_layers):
        model.add(
            Dense(
                layer_width,
                activation="relu",
                kernel_regularizer=kernel_regularizer,
            ),
            rebuild=False,
        )
    model.add(
        Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=kernel_regularizer,
        )
    )

    model.compile(
        optimizer=Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        jit_compile=jit_compile,
    )

    x, y = make_circles(data_size, noise=0.05, random_state=42)

    history = model.fit(x, y, batch_size, epochs, validation_split=val_split)

    return model, history
