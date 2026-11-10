ese 4020 final project - baseline checkpoint

run:
  pip install numpy pandas scikit-learn matplotlib pyarrow
  python baseline.py --data data/ --out results/

data is Open Source Asset Pricing (Chen & Zimmermann 2022), release 2025.10, long-short returns:
https://huggingface.co/datasets/OpenSourceAP/CrossSection
data/ only has 5 of the signals for now. to run on everything: python baseline.py --data PredictorLSretWide.parquet --out results_full/

2010-2024 is the test period, the code never uses it.
