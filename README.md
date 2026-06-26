# UAMNet

UAMNet is an uncertainty-aware multi-task 1D-CNN for polymer identification and
aging-duration regression from hyperspectral spectra.

This repository provides the main network architecture, training code, testing
code, pretrained model weights, and summary test-set results used in the paper.
Raw and processed hyperspectral spectral datasets are not included and are
available from the corresponding author upon reasonable request.

## Repository Structure

```text
UAMNet_github_release/
  data/
    README.md
    prepare_split_data.py         # optional local conversion script
    particle_split_info.csv
    uamnet_split_data.npz        # processed split file, not included

  models/
    uamnet.py                    # network, MC dropout, NLL loss

    train_uamnet.py              # model training

    test_internal.py             # internal test-set evaluation

  weights/
    UAMNet_best.h5               # pretrained model
    training_history.json
    config.json

  results/
    internal_test/
      metrics.json
      material_metrics.csv

  requirements.txt
```

## Data

The model-development dataset was split at the particle level into training,
validation, and internal test sets at a 7:1:2 ratio. The split information is
provided in:

```text
data/particle_split_info.csv
```

The training and testing scripts expect the processed split file:

```text
data/uamnet_split_data.npz
```

This file is not included in the public repository because it contains the
processed spectra used for model development.

If the saved split pickle is available locally, the processed file can be
generated with:

```bash
python data/prepare_split_data.py \
  --params-pkl /path/to/all_data_and_params.pkl \
  --out data/uamnet_split_data.npz
```

## Training

```bash
python train/train_uamnet.py \
  --data data/uamnet_split_data.npz \
  --out weights/retrained \
  --seed 42
```

The training script performs MSE warm-up followed by NLL fine-tuning.

## Internal Test Evaluation

```bash
python test/test_internal.py \
  --data data/uamnet_split_data.npz \
  --model weights/UAMNet_best.h5 \
  --out results/internal_test \
  --mc 40 \
  --std-mode mean \
  --batch 0 \
  --seed 42
```

Pixel-level predictions are aggregated by particle ID before calculating
classification and aging-regression metrics. Overall metrics are saved to
`metrics.json`, and polymer-specific metrics are saved to
`material_metrics.csv`. The default full-batch inference matches the evaluation
notebook used to generate the reported test-set results.

## Environment

The original analysis used Python 3.8.15, TensorFlow 2.13.1, and Keras 2.13.
See `requirements.txt`.
