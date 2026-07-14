import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Convert the saved particle-level split pickle to the npz file used by training/testing scripts."
    )
    parser.add_argument("--params-pkl", required=True, help="Path to all_data_and_params.pkl.")
    parser.add_argument("--out", default="data/uamnet_split_data.npz")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.params_pkl, "rb") as f:
        data = pickle.load(f)

    pids = np.asarray(data["pids"])
    train_pids = np.asarray(data["train_pids"])
    val_pids = np.asarray(data["val_pids"])
    test_pids = np.asarray(data["test_pids"])

    pid_tr = pids[np.isin(pids, train_pids)]
    pid_va = pids[np.isin(pids, val_pids)]
    pid_te = pids[np.isin(pids, test_pids)]

    np.savez_compressed(
        out,
        X_tr=data["X_tr"],
        X_va=data["X_va"],
        X_te=data["X_te"],
        yc_tr=data["yc_tr"],
        yc_va=data["yc_va"],
        yc_te=data["yc_te"],
        yr_tr=data["yr_tr"],
        yr_va=data["yr_va"],
        yr_te=data["yr_te"],
        yr_tr_n=data["yr_tr_n"],
        yr_va_n=data["yr_va_n"],
        yr_te_n=data["yr_te_n"],
        pid_tr=pid_tr,
        pid_va=pid_va,
        pid_te=pid_te,
        train_pids=train_pids,
        val_pids=val_pids,
        test_pids=test_pids,
        classes=np.asarray(data["classes"]),
        yr_max=float(data["yr_max"]),
        input_dim=int(data["IN_DIM"]),
        n_classes=int(data["N_CLS"]),
    )

    meta = {
        "yr_max": float(data["yr_max"]),
        "input_dim": int(data["IN_DIM"]),
        "n_classes": int(data["N_CLS"]),
        "classes": [str(x) for x in data["classes"]],
        "n_train_pixels": int(len(data["X_tr"])),
        "n_val_pixels": int(len(data["X_va"])),
        "n_test_pixels": int(len(data["X_te"])),
        "n_train_particles": int(len(train_pids)),
        "n_val_particles": int(len(val_pids)),
        "n_test_particles": int(len(test_pids)),
    }
    with open(out.with_name(out.stem + "_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved split data: {out}")


if __name__ == "__main__":
    main()
