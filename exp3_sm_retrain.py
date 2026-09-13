import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA_DIR, RESULTS_SM_DIR, SM_LABELS
from train_daily2 import DISAGREE_IDX, build_hicagat, compute_loss

SEQ_KEYS = ['type_f_seq', 'type_m_seq', 'static', 'et_target', 'et_ref', 'fd_target',
            'patch_id', 'year', 'month']
FLOAT_KEYS = {'type_f_seq', 'type_m_seq', 'static', 'et_target', 'et_ref', 'fd_target'}


class SeqSubset(Dataset):
    def __init__(self, tensors, idx):
        self.T = tensors
        self.idx = idx
        seq_len = tensors['type_f_seq'].shape[1]
        self._f = torch.ones(seq_len, tensors['type_f_seq'].shape[2])
        self._m = torch.ones(seq_len, tensors['type_m_seq'].shape[2])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return {'type_f_seq': self.T['type_f_seq'][j], 'type_m_seq': self.T['type_m_seq'][j],
                'type_f_mask': self._f, 'type_m_mask': self._m, 'static': self.T['static'][j],
                'et_target': self.T['et_target'][j], 'et_ref': self.T['et_ref'][j],
                'fd_target': self.T['fd_target'][j], 'patch_id': self.T['patch_id'][j],
                'year': self.T['year'][j], 'month': self.T['month'][j]}


def load_pooled(data_dir):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob('train_part*.pt')) + [data_dir / 'val_data.pt', data_dir / 'test_data.pt']
    acc = {k: [] for k in SEQ_KEYS}

    for f in files:
        if not f.exists():
            continue
        print(f"  loading {f.name}")
        d = torch.load(f, weights_only=False)
        for k in SEQ_KEYS:
            t = torch.as_tensor(d[k])
            acc[k].append(t.float() if k in FLOAT_KEYS else t.long())
        del d

    T = {k: torch.cat(v, 0) for k, v in acc.items()}
    print(f"  pooled: {len(T['fd_target']):,}")
    return T


def swap_to_sm_label(T, sm_csv):
    sm = pd.read_csv(sm_csv)
    key = lambda p, y, m: (int(p) << 20) | (int(y) << 4) | int(m)
    sm_map = {key(r.patch_id, r.year, r.month): int(r.sm_onset) for r in sm.itertuples()}

    pid = T['patch_id'].numpy()
    yr = T['year'].numpy()
    mo = T['month'].numpy()
    new = np.zeros(len(pid), dtype=np.float32)
    for i in range(len(pid)):
        new[i] = sm_map.get(key(pid[i], yr[i], mo[i]), 0)

    T['fd_target'] = torch.from_numpy(new)
    print(f"  SM-label positive rate: {new.mean():.3%}  ({int(new.sum()):,} positives)")
    return T


def train_eval(T, tr_idx, te_idx, cfg, args, dev, zero_cv=False):
    if zero_cv:
        T = dict(T)
        T['type_f_seq'] = T['type_f_seq'].clone()
        T['type_f_seq'][:, :, DISAGREE_IDX] = 0

    model = build_hicagat(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)
    tr = DataLoader(SeqSubset(T, tr_idx), batch_size=args.batch_size, shuffle=True,
                    num_workers=0, pin_memory=True, drop_last=True)
    te = DataLoader(SeqSubset(T, te_idx), batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=True)

    use_amp = dev.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    has_notears = cfg['causal_mode'] == 'notears'

    for ep in range(1, args.epochs + 1):
        model.train()
        model.current_epoch = ep
        t0 = time.time()
        for batch in tr:
            bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(bd)
                L = compute_loss(model, out, bd, args, has_notears)
            scaler.scale(L['total']).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        if ep % 5 == 0 or ep == args.epochs:
            print(f"    epoch {ep}/{args.epochs} ({time.time() - t0:.0f}s)")

    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for batch in te:
            bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(bd)
            P.extend(out['fd_prob'].float().cpu().numpy())
            Y.extend(bd['fd_target'].cpu().numpy())

    P, Y = np.array(P), np.array(Y)
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        f = f1_score(Y, (P > t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    preds_05 = (P > 0.5).astype(int)
    return {'f1_t05': float(f1_score(Y, preds_05, zero_division=0)),
            'f1_opt': float(best_f1), 'opt_thresh': float(best_t),
            'pr_auc': float(average_precision_score(Y, P)),
            'roc_auc': float(roc_auc_score(Y, P)),
            'prec_t05': float(precision_score(Y, preds_05, zero_division=0)),
            'rec_t05': float(recall_score(Y, preds_05, zero_division=0)),
            'pos_rate': float(Y.mean())}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(DATA_DIR))
    ap.add_argument('--sm_labels', default=str(SM_LABELS))
    ap.add_argument('--output_dir', default=str(RESULTS_SM_DIR))
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_notears', type=float, default=5e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lambda_et', type=float, default=0.5)
    ap.add_argument('--lambda_fd', type=float, default=1.0)
    ap.add_argument('--lambda_physics', type=float, default=0.2)
    ap.add_argument('--lambda_recon', type=float, default=0.05)
    ap.add_argument('--lambda_dag', type=float, default=0.01)
    ap.add_argument('--lambda_sparse', type=float, default=0.001)
    ap.add_argument('--mask_lambda_entropy', type=float, default=0.005)
    ap.add_argument('--which', default='both', choices=['both', 'with_cv', 'without_cv'])
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(Path(args.data_dir) / 'config.pkl', 'rb') as f:
        c = pickle.load(f)
    cfg = {'n_type_f': c['n_type_f'], 'n_type_m': c['n_type_m'], 'n_static': c['n_static'],
           'd_model': 128, 'n_layers': 3, 'n_heads': 4, 'dropout': 0.1, 'n_lags': 7,
           'causal_mode': 'notears', 'notears_subsample': 1, 'notears_noise': 0.0,
           'mask_lambda_l1': 0.01, 'mask_temperature': 1.0}

    print("Pooling...")
    T = load_pooled(args.data_dir)
    print("Swapping to SM label...")
    T = swap_to_sm_label(T, args.sm_labels)

    yr = T['year'].numpy()
    tr_idx = np.where(yr <= 2021)[0]
    te_idx = np.where(yr >= 2023)[0]
    print(f"Train {len(tr_idx):,} | Test {len(te_idx):,}")

    res_path = out / 'sm_retrain_results.json'
    results = json.load(open(res_path)) if res_path.exists() else {}

    if args.which in ('both', 'with_cv'):
        print("\n=== FULL (with CV) on SM label ===")
        results['with_cv'] = train_eval(T, tr_idx, te_idx, cfg, args, dev, zero_cv=False)
        r = results['with_cv']
        print(f"  F1={r['f1_t05']:.4f}  PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}")
        json.dump(results, open(res_path, 'w'), indent=2)

    if args.which in ('both', 'without_cv'):
        print("\n=== WITHOUT CV (disagreement zeroed) on SM label ===")
        results['without_cv'] = train_eval(T, tr_idx, te_idx, cfg, args, dev, zero_cv=True)
        r = results['without_cv']
        print(f"  F1={r['f1_t05']:.4f}  PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}")
        json.dump(results, open(res_path, 'w'), indent=2)

    if 'with_cv' in results and 'without_cv' in results:
        results['cv_increment_f1'] = results['with_cv']['f1_t05'] - results['without_cv']['f1_t05']
        results['cv_increment_prauc'] = results['with_cv']['pr_auc'] - results['without_cv']['pr_auc']
    json.dump(results, open(res_path, 'w'), indent=2)

    print("\n" + "=" * 60)
    print("EXP 3 SUMMARY - independent SM-label prediction")
    print("=" * 60)
    if 'with_cv' in results:
        r = results['with_cv']
        print(f"  With CV:    F1={r['f1_t05']:.4f}  PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}")
    if 'without_cv' in results:
        r = results['without_cv']
        print(f"  Without CV: F1={r['f1_t05']:.4f}  PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}")
    if 'cv_increment_f1' in results:
        print(f"  CV increment: F1 {results['cv_increment_f1']:+.4f}  "
              f"PR-AUC {results['cv_increment_prauc']:+.4f}")


if __name__ == '__main__':
    main()
