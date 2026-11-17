ese 4020 final project

group: David, Saranya, Peter

run:
  pip install numpy pandas scikit-learn matplotlib pyarrow
  python baseline.py --data data/ --out results/

data is Open Source Asset Pricing (Chen & Zimmermann 2022), release 2025.10, long-short returns:
https://huggingface.co/datasets/OpenSourceAP/CrossSection
data/ only has 5 of the signals for now. to run on everything: python baseline.py --data PredictorLSretWide.parquet --out results_full/

2010-2024 is the test period, the code never uses it.

## draft checkpoint (16 nov)

new: experiments.py (ridge, lasso, elastic net, point-in-time filter, sharpe + turnover costs) and figures.py.
same split, folds, 12-month gap, bootstrap and seed as baseline.py; ridge on the 5 signals reproduces the baseline numbers exactly.
experiments.py drops every month from 2010 on when it loads the data.

full data (not committed, download from branch 2025.10 of the link above):
  Portfolios/Full Sets OP/PredictorLSretWide.parquet
  SignalDoc.csv   (publication years, for --pit)

  python experiments.py --data data/ --out results_5/                                    
  python experiments.py --data PredictorLSretWide.parquet --out results_all/             
  python experiments.py --data PredictorLSretWide.parquet --doc SignalDoc.csv --pit --models hm,ridge,lasso --out results_pit/   
  python figures.py results_5 results_all results_pit --out paper/figs/

long runs save one checkpoint per year in <out>/checkpoints/ and resume if restarted (--budget N stops after N seconds).

paper/: draft.tex (NeurIPS 2025 style, preprint option), draft.pdf, figs/.
