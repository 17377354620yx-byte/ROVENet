# ROVENet P2P Liver experiment

`config.py` defines the unchanged P2P Liver data, backbone, matching, registration, and loss settings for the fixed ROVENet architecture. `dataset.py` preserves the original preprocessing and test protocol. `trainval.py` saves the best training-side validation checkpoint; `test.py` strictly loads it and exports the original per-case and aggregate metrics.

RTOR and A3 are always present. The `cooperative`, `legacy`, and `soft_overlap` interaction profiles describe coarse proposal training and overlap selection behavior; they do not remove either module. For a trained checkpoint, choose the profile recorded in its metadata.

Run commands from the repository root. See [README.md](../../README.md) and [TRAIN_TEST.md](../../TRAIN_TEST.md).

Historical module ablation material is retained under `archive/` and is not an entry point for the fixed ROVENet model.
