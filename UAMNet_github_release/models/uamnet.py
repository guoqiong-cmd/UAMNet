import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MCDropout(layers.Dropout):
    """Dropout layer kept active during inference for MC dropout."""

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


def nll_loss(y_true, y_pred):
    mu = y_pred[:, 0:1]
    log_var = tf.clip_by_value(y_pred[:, 1:2], -4.0, 2.0)
    precision = tf.exp(-log_var)
    return tf.reduce_mean(0.5 * precision * tf.square(y_true - mu) + 0.5 * log_var)


def reg_mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred[:, 0:1])


def build_uamnet(input_dim, n_classes):
    inp = keras.Input(shape=(input_dim, 1))

    x = layers.Conv1D(32, 11, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 7, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    shared = layers.Dense(256, activation="relu")(x)
    shared = layers.Dropout(0.1)(shared)

    cls = layers.Dense(128, activation="relu")(shared)
    cls = layers.Dense(n_classes, activation="softmax", name="cls")(cls)

    reg = layers.Dense(256, activation="relu")(shared)
    reg = layers.BatchNormalization()(reg)
    reg = MCDropout(0.3)(reg)
    reg = layers.Dense(128, activation="relu")(reg)
    reg = layers.Dense(2, activation="linear", name="reg")(reg)

    return keras.Model(inp, [cls, reg], name="UAMNet")


CUSTOM_OBJECTS = {
    "MCDropout": MCDropout,
    "nll_loss": nll_loss,
    "reg_mae": reg_mae,
}
