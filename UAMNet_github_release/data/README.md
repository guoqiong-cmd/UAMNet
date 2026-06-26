# Data

The raw and processed hyperspectral spectral datasets are not included in this
repository. They are available from the corresponding author upon reasonable
request.

The training and testing scripts expect an already split processed file:

```text
data/uamnet_split_data.npz
```

This file contains the processed spectra and labels used for model development,
so it is intentionally excluded from the public repository.

If the saved split pickle is available locally, generate the processed file with:

```bash
python data/prepare_split_data.py \
  --params-pkl /path/to/all_data_and_params.pkl \
  --out data/uamnet_split_data.npz
```

This file should contain:

```text
X_tr, X_va, X_te
yc_tr, yc_va, yc_te
yr_tr, yr_va, yr_te
yr_tr_n, yr_va_n, yr_te_n
pid_tr, pid_va, pid_te
train_pids, val_pids, test_pids
classes
yr_max
input_dim
n_classes
```

`particle_split_info.csv` records the particle-level 7:1:2 split used in the
study.
