# Fin Whale Detection in DAS

Reproduction of the pipeline from article “Automated Detection of Fin Whales with Distributed Acoustic Sensing in the Arctic and Mediterranean.”

Authors: Khanh Truong, Jo Eidsvik, Robin Andre Rørstadbotnen, Jan Petter Morten, Laurine Andres, Anthony Sladen.

## Installation
Create an isolated environment (Python 3.10) and install the package:

Conda:
```bash
conda create --name dasly-env python=3.10
conda activate dasly-env
pip install -e .
```

venv (no conda):
```bash
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

Install directly from Git:
```bash
pip install "git+https://github.com/ktruongme/fin-whale-detection-das.git"
```

Update an existing installation from Git:
```bash
pip install --force-reinstall --no-deps \
  "git+https://github.com/ktruongme/fin-whale-detection-das.git"
```

## CLI
Example command with all required arguments:

```bash
dasly whales \
  --exp-path /path/to/experiment \
  --chunk-size 6 \
  --chunk-stride 5 \
  --db-table events_v1 \
  --connection-string "postgresql+psycopg2://user:pass@host:5432/db"
```

Optional channel bounds:
```bash
dasly whales \
  --exp-path /path/to/experiment \
  --chunk-size 6 \
  --chunk-stride 5 \
  --db-table events_v1 \
  --connection-string "postgresql+psycopg2://user:pass@host:5432/db" \
  --n-start 5000 \
  --n-end 115000
```

## Notebooks
Example workflows are in `notebooks/` (e.g., `dbscan.ipynb`, `ht.ipynb`, `tm.ipynb`, `yolo.ipynb`). Run after installing the package so imports resolve.

## Data
- Included: a 1-minute segment of the Svalbard DAS recording (UTC 2022-08-22 12:25:09), high-pass filtered at 1 Hz, located in `data/`. This is sufficient to run the notebooks and demonstrate the pipeline.
- Not included: the full Svalbard dataset and the Italy-Monaco dataset used in the paper (access-restricted).
- Alternative public data: a related Svalbard DAS dataset (Rørstadbotnen et al., 2023) is available at https://doi.org/10.18710/Q8OSON. It is in MATLAB (.mat) format rather than HDF5, so you would need to adapt the loading code to use it.

## Models
- A pretrained fin-whale detector is bundled with the package and used by the
  CLI by default.
- You can override it with `--model-path` if you want to use different
  weights.

## License
This project is released under the MIT License. See `LICENSE` for details.

## Contact
For inquiries, please contact **khanh.p.d.truong@ntnu.no** or **ktruong.me@gmail.com**.
