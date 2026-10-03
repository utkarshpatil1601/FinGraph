# FinGraph Phase 1 — Foundation

This phase implements:
1. PaySim raw-data inspection
2. Data cleaning
3. Transaction feature engineering
4. Synthetic Device/IP/Merchant/Session relationship generation
5. Merge into a FinGraph transaction dataset
6. Random Forest baseline
7. Baseline evaluation
8. Sampled heterogeneous graph construction

## Expected raw file

Put the original PaySim CSV at:

data/raw/PS_20174392719_1491204439457_log.csv

The code also accepts any single CSV inside data/raw/ when run with --input.

## Important methodology note

The synthetic Device/IP/Merchant/Session layer is project-generated because standard PaySim does not contain all of these entities.

For a leakage-safe ML experiment, the baseline model in this phase uses transaction features only and does NOT use synthetic relationship fields.

The optional demo-aligned relationship mode can create stronger shared relationships around known fraud-linked accounts, but that mode must NOT be used as evidence for model performance because it uses the fraud label to create graph structure.

## Run order

From the fingraph_phase1 directory:

1. Create environment
2. Install requirements
3. Put PaySim CSV in data/raw/
4. Run:
   python scripts/01_eda.py
5. Run:
   python scripts/02_preprocess.py
6. Run:
   python scripts/03_generate_relationships.py
7. Run:
   python scripts/04_merge_fingraph_dataset.py
8. Run:
   python scripts/05_train_baseline.py
9. Run:
   python scripts/06_evaluate_baseline.py
10. Install graph dependencies and build a graph smoke-test artifact:
   pip install torch torch-geometric
   python graph/build_graph.py --sample-size 100000

Outputs are written to data/processed, data/synthetic, models, and reports.
