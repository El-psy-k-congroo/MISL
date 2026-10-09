# MISL: Mechanism-Inspired Slot Learning for Drug Synergy Prediction

## Requirements

- Python 3.10.16
- PyTorch 2.5.1
- CUDA runtime 11.8
- RDKit 2024.9.6

The dependency versions are pinned to the working Windows environment, tested
with an NVIDIA GeForce RTX 3090 Ti. An NVIDIA GPU and a driver compatible with
CUDA 11.8 are required for the GPU environment below.

From the repository root, create and activate the environment:

```bash
conda env create -f environment.yml
conda activate misl
```

Alternatively, in an existing Python 3.10.16 environment, install the CUDA 11.8
build of PyTorch first, followed by the pinned requirements:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

`requirements.txt` includes the project's Python dependencies and their
transitive dependencies without optional extras; it is not a full export of
unrelated packages from the local environment. `environment.yml` also specifies
Python and the PyTorch CUDA runtime. Neither file pins the operating system,
GPU driver, or every native-library build.

## Dataset Setup

Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1tNi_Y2DhgMu0kg3H9OKXrH1mvf-DBLTm?usp=sharing) and place them under the `data/` directory:

```text
MISL/
└── data/
    ├── DrugCombDB/
    │   ├── cell_protein.csv
    │   ├── drug_combinations.csv
    │   ├── drug_protein.csv
    │   ├── drug_smiles.csv
    │   ├── entity_id.json
    │   ├── Expression.csv
    │   ├── kg.csv
    │   ├── protein-protein_network.xlsx
    │   └── smile2graph.json
    └── OncologyScreen/
        ├── cell_protein.csv
        ├── drug_combinations.csv
        ├── drug_protein.csv
        ├── drug_smiles.csv
        ├── entity_id.json
        ├── Expression.csv
        ├── kg.csv
        ├── protein-protein_network.xlsx
        └── smile2graph.json
```

## Running MISL

Warm-start evaluation:

```bash
python main.py --dataset OncologyScreen --split_strategy random
python main.py --dataset DrugCombDB --split_strategy random
```

Cold-start evaluation:

```bash
# Transductive KG mode (default): held-out entities have no synergy labels, but
# their non-label KG nodes and relations remain available during training.
python main.py --dataset OncologyScreen --split_strategy cold_drug
python main.py --dataset DrugCombDB --split_strategy cold_drug

python main.py --dataset OncologyScreen --split_strategy cold_cell
python main.py --dataset DrugCombDB --split_strategy cold_cell

python main.py --dataset OncologyScreen --split_strategy cold_comb
python main.py --dataset DrugCombDB --split_strategy cold_comb
```

Strict inductive KG evaluation is enabled with `--kg_mode inductive`:

```bash
# Held-out drug nodes and their incident KG edges are masked during training
# and validation, then restored for frozen-model test inference.
python main.py --dataset OncologyScreen --split_strategy cold_drug --kg_mode inductive
python main.py --dataset DrugCombDB --split_strategy cold_drug --kg_mode inductive

# The same protocol is applied to held-out cell-line nodes.
python main.py --dataset OncologyScreen --split_strategy cold_cell --kg_mode inductive
python main.py --dataset DrugCombDB --split_strategy cold_cell --kg_mode inductive
```

For `random` and `cold_comb`, `--kg_mode inductive` leaves the KG unchanged
because these protocols do not hold out individual drug or cell-line nodes.

## Hyperparameter Settings

| Hyperparameter | Value |
|---|---:|
| Random seed | 2025 |
| Cross-validation folds | 5 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 768 |
| Maximum epochs | 600 |
| Early-stopping patience | 70 |
| Slot interaction depth | 2 |
| Number of mechanism slots | 4 |
| Attention heads | 4 |
| RGCN input dimension | 300 |
| RGCN hidden dimension | 600 |
| RGCN output dimension | 300 |
| Projection dimension | 256 |
| Dropout | 0.5 |
| Auxiliary prediction-loss weight | 1 |
| Consistency-loss weight | 5 |
| Optimizer | Adam |
| Learning-rate scheduler | OneCycleLR |
| Model-selection metric | Validation accuracy |

## Outputs

Logs are written to `logs/`. Fold-level checkpoints, metrics, and run configurations are saved under:

```text
experiment/<split_strategy>/<dataset>/<timestamp>/
```
