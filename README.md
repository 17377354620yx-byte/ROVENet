# ROVENet

ROVENet is a rigid complete-to-partial liver point cloud registration network. It combines the reusable GeoTransformer backbone and registration components with two fixed model components: **RTOR** (topology and overlap refinement) and **A3** (geometry-aware fine patch refinement). Both components are always constructed and executed. The network predicts one rigid transform `(R, t)`.

The Python package `geotransformer` retains upstream common components and the compiled point cloud operators. ROVENet-specific code lives in `rovenet/`. Existing checkpoint parameter names remain unchanged.

## Repository layout

| Path | Purpose |
| --- | --- |
| `geotransformer/` | KPConv-FPN, Geometric Transformer, matching, Sinkhorn, registration, utilities, and C++ operators |
| `rovenet/models/registration.py` | Fixed ROVENet network assembly and forward pass |
| `rovenet/modules/` | RTOR, A3, overlap region selection, and coarse proposal mixing |
| `experiments/geotransformer.p2p_liver/` | Original P2P Liver data adapter, configuration, training, test, and evaluation metrics |
| `experiments/geotransformer.ircadb_v3/` | Additional 3Dircadb experiment |
| `archive/` | Historical ablation and unrelated experimental code, retained for provenance |
| `docs/` | Architecture and refactor reports |

## Installation

The verified local environment is Python 3.10, PyTorch 2.9.1 with CUDA, Open3D 0.16.0, and a CUDA toolkit. Install a PyTorch build appropriate for your GPU first, then from this directory run:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
python setup.py build_ext --inplace
```

The last command builds `geotransformer.ext`. The checked local Conda environment is `geo_v2`; for that environment use `conda run --no-capture-output -n geo_v2` before Python commands. Dataset files and model checkpoints are not included in Git.

## P2P Liver data

The default training and test root is `/mnt/data3/yangx/P2P/Dataset`. Override it with `P2P_DATA_ROOT`, and optionally set `P2P_TEST_DATA_ROOT` for a separate test root. The in-vitro root defaults to `/mnt/data3/yangx/P2P/in_vitro` and can be changed with `P2P_IN_VITRO_ROOT`. The input format and original split, marker-based RMS-TRE in millimetres, RRE, RTE, and success-rate protocol are retained.

## Train

From the repository root, using the verified `geo_v2` environment:

```bash
P2P_RUN_NAME=rovenet_cooperative_v1 bash scripts/train_cooperative.sh
```

The command trains the fixed RTOR+A3 network with the existing cooperative coarse proposal schedule. It writes `output/geotransformer.p2p_liver.rovenet_cooperative_v1/snapshots/best.pth.tar`. Set `P2P_RUN_NAME` and pass `--resume` to continue a run. Detailed direct commands are in [TRAIN_TEST.md](TRAIN_TEST.md).

## Test and evaluate

```bash
P2P_RUN_NAME=rovenet_cooperative_v1 bash scripts/test_cooperative.sh
```

This runs the original in-silico noise settings (`none`, `2`, `4`) and in-vitro evaluation, writing JSON summaries, per-case CSV files, and RMS-TRE arrays under the run directory. An existing complete RTOR+A3 checkpoint can be supplied without copying it:

```bash
P2P_RUN_NAME=rovenet_existing_eval \
P2P_SNAPSHOT=/absolute/path/to/best.pth.tar \
bash scripts/test_cooperative.sh
```

The example checkpoint must have the `cooperative` interaction profile. `test.py` loads model weights with `strict=True` and rejects checkpoints whose metadata specifies an incomplete architecture. The retired `--architecture` option errors explicitly. `--interaction_profile` remains available because it controls the historical proposal schedule, not whether RTOR or A3 exists.

## Documentation and attribution

- [Model components](docs/MODEL_COMPONENTS.md)
- [Refactor and checkpoint verification](docs/REFACTOR_REPORT.md)
- [Historical P2I protocol audit](P2I_LREG_PROTOCOL_AUDIT.md)

ROVENet derives its common code from [GeoTransformer](https://github.com/qinzheng93/GeoTransformer), released under the MIT License. The upstream copyright and license are preserved in [LICENSE](LICENSE). The original RTOR+A3 design and experimental lineage are documented in the reports above.
