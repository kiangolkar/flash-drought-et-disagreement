# flash-drought-et-disagreement

Code for the paper *Satellite Evapotranspiration Model Disagreement as an Early Indicator of Flash Drought: a CONUS-scale Deep Learning Analysis*.

## Files

- `train_daily2.py` — training, model definitions, ablation variants, evaluation
- `exp3_sm_retrain.py` — retraining on independent SMAP soil moisture labels
- `config.pkl` — feature dimensions, column names, year splits
- `scalers.pkl` — scalers fitted on 2016–2021
- `sm_onset_labels.csv` — derived SMAP onset labels

Preprocessing, Earth Engine extraction, the monthly disagreement analysis, and figure scripts will be added before publication. Available from the corresponding author in the meantime.

## Running it

```bash
pip install -r requirements.txt

python train_daily2.py --model full \
    --data_dir /path/to/preprocessed_daily_v3 \
    --output_dir results/full --epochs 50 --batch_size 256
```

For the soil moisture experiment:

```bash
python exp3_sm_retrain.py --data_dir /path/to/preprocessed_daily_v3 \
    --sm_labels sm_onset_labels.csv --output_dir results/sm --which both
```

Each run writes `test_metrics.json`, `history.csv`, and `test_predictions.npz`.

## Data

Preprocessed sequence files are not included; the processed dataset goes to Zenodo on acceptance.

Raw inputs are all public: OpenET, GridMET, MODIS MOD13A1, Sentinel-1, SRTM, POLARIS, SMAP L4, and the U.S. Drought Monitor. Only derived layers are shared here.

Note `scalers.pkl` was written with scikit-learn 1.5.1 — newer versions load it but complain.

## License

MIT for the code. The satellite datasets keep their own licenses.
