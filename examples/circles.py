from keras import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.datasets import make_circles


def build_model(
    num_layers=10,
    layer_width=500,
    jit_compile="auto",
    kernel_regularizer=None,
    learning_rate=1e-5,
    batch_size=8,
    epochs=500,
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


if __name__ == "__main__":
    from numpy import savez

    from persistence_reg import NeuralPersistence

    from .utils import plot_history

    model, history = build_model()
    plot_history(history)
    model.save("no_reg.keras")
    savez("no_reg", **history.history)

    model, history = build_model(kernel_regularizer=NeuralPersistence(scale=0.1))
    plot_history(history)
    model.save("reg.keras")
    savez("reg", **history.history)
