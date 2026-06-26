import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.uamnet import CUSTOM_OBJECTS


def mc_predict(model, X, yr_max, n_mc, batch, logvar_clip=None):
    cls_samples, mu_samples, var_samples = [], [], []
    X = X[..., np.newaxis]
    if batch <= 0 or batch > len(X):
        batch = len(X)

    for _ in range(n_mc):
        cls_batches, mu_batches, var_batches = [], [], []
        for start in range(0, len(X), batch):
            stop = start + batch
            cls, reg = model(X[start:stop], training=True)
            cls_batches.append(cls.numpy())
            mu_batches.append(reg[:, 0].numpy() * yr_max)
            log_var = reg[:, 1].numpy()
            if logvar_clip is not None:
                log_var = np.clip(log_var, logvar_clip[0], logvar_clip[1])
            var_batches.append(np.exp(log_var) * yr_max ** 2)

        cls_samples.append(np.concatenate(cls_batches, axis=0))
        mu_samples.append(np.concatenate(mu_batches, axis=0))
        var_samples.append(np.concatenate(var_batches, axis=0))

    cls_mean = np.mean(cls_samples, axis=0)
    mu_samples = np.vstack(mu_samples)
    var_samples = np.vstack(var_samples)

    pred_cls = cls_mean.argmax(axis=1)
    pred_day = mu_samples.mean(axis=0)
    epistemic_std = mu_samples.std(axis=0, ddof=1)
    aleatoric_std = np.sqrt(var_samples.mean(axis=0))
    total_std = np.sqrt(epistemic_std ** 2 + aleatoric_std ** 2)
    return pred_cls, pred_day, aleatoric_std, epistemic_std, total_std


def particle_aggregate(pid, y_cls, p_cls, y_day, p_day, ale, epi, total, std_mode="mean"):
    pixel_df = pd.DataFrame({
        "particle_id": pid,
        "true_cls": y_cls,
        "pred_cls": p_cls,
        "true_day": y_day,
        "pred_day": p_day,
        "aleatoric_std_day": ale,
        "epistemic_std_day": epi,
        "total_std_day": total,
    })

    if std_mode not in {"mean", "rms"}:
        raise ValueError("std_mode must be 'mean' or 'rms'")

    def std_agg(values):
        values = np.asarray(values, dtype=float)
        if std_mode == "mean":
            return float(values.mean())
        return float(np.sqrt(np.mean(values ** 2)))

    rows = []
    for particle_id, group in pixel_df.groupby("particle_id", sort=False):
        rows.append({
            "particle_id": particle_id,
            "true_cls": int(group["true_cls"].iloc[0]),
            "pred_cls": int(np.bincount(group["pred_cls"]).argmax()),
            "true_day": float(group["true_day"].iloc[0]),
            "pred_day": float(group["pred_day"].mean()),
            "aleatoric_std_day": std_agg(group["aleatoric_std_day"]),
            "epistemic_std_day": std_agg(group["epistemic_std_day"]),
            "total_std_day": std_agg(group["total_std_day"]),
            "n_pixels": int(len(group)),
        })
    return pd.DataFrame(rows)


def regression_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae_days": float(mean_absolute_error(y_true, y_pred)),
        "rmse_days": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/uamnet_split_data.npz")
    parser.add_argument("--model", default="weights/UAMNet_best.h5")
    parser.add_argument("--out", default="results/internal_test")
    parser.add_argument("--mc", type=int, default=40)
    parser.add_argument("--batch", type=int, default=0, help="Inference batch size. Use 0 for full-batch inference, matching the notebook.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--std-mode", choices=["mean", "rms"], default="mean")
    parser.add_argument("--clip-logvar", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    data = np.load(args.data, allow_pickle=True)
    model = keras.models.load_model(args.model, custom_objects=CUSTOM_OBJECTS)

    logvar_clip = (-4.0, 2.0) if args.clip_logvar else None
    pred_cls, pred_day, ale, epi, total = mc_predict(
        model, data["X_te"], float(data["yr_max"]), args.mc, args.batch, logvar_clip=logvar_clip
    )

    particles = particle_aggregate(
        data["pid_te"], data["yc_te"], pred_cls, data["yr_te"], pred_day, ale, epi, total,
        std_mode=args.std_mode,
    )
    particles["abs_error_day"] = np.abs(particles["pred_day"] - particles["true_day"])
    particles.to_csv(os.path.join(args.out, "particle_predictions.csv"), index=False)

    y_true = particles["true_day"]
    y_pred = particles["pred_day"]
    overall_reg = regression_metrics(y_true, y_pred)
    metrics = {
        "n_particles": int(len(particles)),
        "mc_samples": int(args.mc),
        "particle_std_aggregation": args.std_mode,
        "logvar_clip": list(logvar_clip) if logvar_clip is not None else None,
        "accuracy": float(accuracy_score(particles["true_cls"], particles["pred_cls"])),
        "macro_f1": float(f1_score(particles["true_cls"], particles["pred_cls"], average="macro")),
        **overall_reg,
        "coverage_95": float(np.mean(np.abs(y_true - y_pred) <= 1.96 * particles["total_std_day"])),
        "mean_aleatoric_std_day": float(particles["aleatoric_std_day"].mean()),
        "mean_epistemic_std_day": float(particles["epistemic_std_day"].mean()),
        "mean_total_std_day": float(particles["total_std_day"].mean()),
    }

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    classes = data["classes"]
    material_rows = []
    for cls_id, group in particles.groupby("true_cls", sort=True):
        material = str(classes[int(cls_id)]) if int(cls_id) < len(classes) else str(cls_id)
        row = {
            "material": material,
            "n_particles": int(len(group)),
            "accuracy": float(accuracy_score(group["true_cls"], group["pred_cls"])),
            **regression_metrics(group["true_day"], group["pred_day"]),
            "mean_aleatoric_std_day": float(group["aleatoric_std_day"].mean()),
            "mean_epistemic_std_day": float(group["epistemic_std_day"].mean()),
            "mean_total_std_day": float(group["total_std_day"].mean()),
        }
        material_rows.append(row)
    pd.DataFrame(material_rows).to_csv(os.path.join(args.out, "material_metrics.csv"), index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
