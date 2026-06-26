import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.uamnet import build_uamnet, nll_loss, reg_mae


def compile_model(model, steps_per_epoch, epochs, use_nll):
    if use_nll:
        lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=2e-4,
            decay_steps=epochs * steps_per_epoch,
            alpha=1e-6,
        )
        reg_loss = nll_loss
    else:
        lr = 5e-4
        reg_loss = "mse"

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss={"cls": "sparse_categorical_crossentropy", "reg": reg_loss},
        loss_weights={"cls": 0.1, "reg": 1.0},
        metrics={"cls": "accuracy", "reg": reg_mae},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/uamnet_split_data.npz")
    parser.add_argument("--out", default="weights/retrained")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    data = np.load(args.data, allow_pickle=True)

    X_tr = data["X_tr"][..., np.newaxis]
    X_va = data["X_va"][..., np.newaxis]
    yc_tr, yc_va = data["yc_tr"], data["yc_va"]
    yr_tr_n, yr_va_n = data["yr_tr_n"], data["yr_va_n"]

    model = build_uamnet(int(data["input_dim"]), int(data["n_classes"]))
    steps = X_tr.shape[0] // args.batch + 1

    print("Stage 1: MSE warm-up")
    compile_model(model, steps, args.epochs, use_nll=False)
    warmup = model.fit(
        X_tr, {"cls": yc_tr, "reg": yr_tr_n},
        validation_data=(X_va, {"cls": yc_va, "reg": yr_va_n}),
        epochs=50,
        batch_size=args.batch,
        verbose=1,
    )

    print("Stage 2: NLL fine-tuning")
    best_path = os.path.join(args.out, "UAMNet_best.h5")
    compile_model(model, steps, args.epochs, use_nll=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_reg_loss", mode="min", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_reg_loss", patience=40, restore_best_weights=True
        ),
    ]
    nll = model.fit(
        X_tr, {"cls": yc_tr, "reg": yr_tr_n},
        validation_data=(X_va, {"cls": yc_va, "reg": yr_va_n}),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1,
    )

    history = {
        "warmup": warmup.history,
        "nll": nll.history,
    }
    with open(os.path.join(args.out, "training_history.json"), "w") as f:
        json.dump(
            {k: {m: [float(x) for x in v] for m, v in h.items()} for k, h in history.items()},
            f,
            indent=2,
        )

    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()
