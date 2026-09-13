import os
from pathlib import Path

ROOT = Path(os.environ.get('DROUGHT_ROOT', 'C:/drought'))

DATA_DIR = ROOT / 'preprocessed_daily_v3'
RESULTS_DIR = ROOT / 'results'
RESULTS_SM_DIR = ROOT / 'results_sm'
SM_LABELS = ROOT / 'sm_onset_labels.csv'
