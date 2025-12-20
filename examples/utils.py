from matplotlib.pyplot import legend, plot, show, xlabel, ylabel


def plot_history(history):
    plot(history.history["loss"])
    plot(history.history["val_loss"])
    xlabel("Epochs")
    ylabel("Loss")
    legend(["Training Loss", "Validation Loss"])
    show()
