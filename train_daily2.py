import argparse
import json
import math
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from paths import DATA_DIR, RESULTS_DIR

DISAGREE_IDX = [7, 8, 9, 14, 15]
OPENET_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15]
ESI_IDX = [10, 15]
ESI_ENSEMBLE_IDX = [6, 10, 15]
LABEL_PROXIMAL_IDX = [6, 7, 8, 9, 10, 14, 15]
ET_REF_IDX_M = [1, 2]


class HiCaGATDataset(Dataset):
    def __init__(self, data_dict, jitter=0):
        self.type_f_seq = torch.FloatTensor(data_dict['type_f_seq'])
        self.type_m_seq = torch.FloatTensor(data_dict['type_m_seq'])
        self.static = torch.FloatTensor(data_dict['static'])
        self.et_target = torch.FloatTensor(data_dict['et_target'])
        self.et_ref = torch.FloatTensor(data_dict['et_ref'])
        self.fd_target = torch.FloatTensor(data_dict['fd_target'])
        self.patch_ids = torch.LongTensor(_as_int64(data_dict['patch_id']))
        self.years = torch.LongTensor(_as_int64(data_dict['year']))
        self.months = torch.LongTensor(_as_int64(data_dict['month']))
        self.jitter = jitter
        self.seq_len = self.type_f_seq.shape[1]
        self.n_type_f = self.type_f_seq.shape[2]
        self.n_type_m = self.type_m_seq.shape[2]
        self.training_mode = False
        self._f_mask = torch.ones(self.seq_len, self.n_type_f)
        self._m_mask = torch.ones(self.seq_len, self.n_type_m)

    def __len__(self):
        return len(self.et_target)

    def __getitem__(self, idx):
        f_seq = self.type_f_seq[idx]
        m_seq = self.type_m_seq[idx]

        if self.jitter > 0 and self.training_mode:
            shift = torch.randint(-self.jitter, self.jitter + 1, (1,)).item()
            if 0 < shift < self.seq_len:
                f_seq = torch.cat([f_seq[shift:], f_seq[-1:].expand(shift, -1)], 0)
                m_seq = torch.cat([m_seq[shift:], m_seq[-1:].expand(shift, -1)], 0)
            elif shift < 0 and -shift < self.seq_len:
                f_seq = torch.cat([f_seq[0:1].expand(-shift, -1), f_seq[:self.seq_len + shift]], 0)
                m_seq = torch.cat([m_seq[0:1].expand(-shift, -1), m_seq[:self.seq_len + shift]], 0)

        return {
            'type_f_seq': f_seq,
            'type_m_seq': m_seq,
            'type_f_mask': self._f_mask,
            'type_m_mask': self._m_mask,
            'static': self.static[idx],
            'et_target': self.et_target[idx],
            'et_ref': self.et_ref[idx],
            'fd_target': self.fd_target[idx],
            'patch_id': self.patch_ids[idx],
            'year': self.years[idx],
            'month': self.months[idx],
        }

    def set_training(self, mode=True):
        self.training_mode = mode


def _as_int64(x):
    return x.astype(np.int64) if isinstance(x, np.ndarray) else x


def load_train_dataset(data_dir, jitter=3):
    data_dir = Path(data_dir)
    parts = sorted(data_dir.glob('train_part*.pt'))

    if not parts:
        d = torch.load(data_dir / 'train_data.pt', weights_only=False)
        ds = HiCaGATDataset(d, jitter=jitter)
        ds.set_training(True)
        print(f"  Train: {len(ds):,}")
        del d
        return ds

    print(f"Loading {len(parts)} train parts (jitter=+-{jitter} days)...")
    datasets = []
    total = 0
    for p in parts:
        print(f"  Loading {p.name}...")
        d = torch.load(p, weights_only=False)
        ds = HiCaGATDataset(d, jitter=jitter)
        ds.set_training(True)
        datasets.append(ds)
        total += len(ds)
        del d
    print(f"  Total train: {total:,}")
    return ConcatDataset(datasets)


def zero_features(train_ds, raw_dicts, idx, key='type_f_seq'):
    if isinstance(train_ds, ConcatDataset):
        for ds in train_ds.datasets:
            getattr(ds, key)[:, :, idx] = 0
    elif hasattr(train_ds, key):
        getattr(train_ds, key)[:, :, idx] = 0

    for d in raw_dicts:
        for i in idx:
            if i < d[key].shape[-1]:
                d[key][:, :, i] = 0


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiSourceEncoder(nn.Module):
    def __init__(self, n_f, n_m, n_s, d):
        super().__init__()
        self.f_proj = nn.Sequential(nn.Linear(n_f, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.1))
        self.m_proj = nn.Sequential(nn.Linear(n_m, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.1))
        self.s_proj = nn.Sequential(nn.Linear(n_s, d), nn.LayerNorm(d), nn.GELU())
        self.pe = TemporalPositionalEncoding(d)

    def forward(self, f_seq, m_seq, f_mask, m_mask, static):
        hf = self.pe(self.f_proj(f_seq * f_mask))
        hm = self.pe(self.m_proj(m_seq * m_mask))
        hs = self.s_proj(static)
        return hf + hs.unsqueeze(1), hm + hs.unsqueeze(1), hs


class NOTEARSModule(nn.Module):
    def __init__(self, n_var, n_lags=3, temporal_subsample=1, recon_noise=0.0):
        super().__init__()
        self.n_var = n_var
        self.n_lags = n_lags
        self.temporal_subsample = temporal_subsample
        self.recon_noise = recon_noise
        self.W = nn.Parameter(torch.randn(n_var, n_var) * 0.1)
        self.W_lag = nn.ParameterList([
            nn.Parameter(torch.randn(n_var, n_var) * 0.05) for _ in range(n_lags)])

    def get_causal_matrix(self):
        return self.W * (1 - torch.eye(self.n_var, device=self.W.device))

    def get_causal_weights(self):
        W = self.get_causal_matrix()
        return torch.sigmoid(W[-1, :-1].abs() * 3.0)

    def reconstruction_loss(self, x):
        if self.temporal_subsample > 1:
            x = x[:, ::self.temporal_subsample, :]
        if self.recon_noise > 0 and self.training:
            x = x + torch.randn_like(x) * self.recon_noise

        W = self.get_causal_matrix()
        xh = x @ W.T
        for i, Wl in enumerate(self.W_lag):
            lag = i + 1
            if lag < x.size(1):
                xh = xh + F.pad(x[:, :-lag, :], (0, 0, lag, 0)) @ Wl.T
        return F.mse_loss(xh, x)

    def dag_loss(self):
        W = self.get_causal_matrix()
        Wsq = W * W
        d = self.n_var
        M = torch.eye(d, device=W.device)
        P = torch.eye(d, device=W.device)
        for k in range(1, 11):
            P = P @ Wsq / k
            M = M + P
        return torch.trace(M) - d

    def sparsity_loss(self):
        l1 = self.get_causal_matrix().abs().sum()
        for Wl in self.W_lag:
            l1 = l1 + Wl.abs().sum()
        return l1


class LearnedSparseMask(nn.Module):
    def __init__(self, n_met, lambda_l1=0.01, temperature=1.0):
        super().__init__()
        self.n_met = n_met
        self.lambda_l1 = lambda_l1
        self.temperature = temperature
        self.raw_weights = nn.Parameter(torch.zeros(n_met))

    def get_weights(self):
        return torch.sigmoid(self.raw_weights * self.temperature)

    def get_causal_weights(self):
        return self.get_weights()

    def sparsity_loss(self):
        return self.lambda_l1 * self.get_weights().sum()

    def entropy_loss(self):
        w = self.get_weights().clamp(1e-6, 1 - 1e-6)
        return -(w * w.log() + (1 - w) * (1 - w).log()).mean()

    def get_importance_dict(self, var_names):
        w = self.get_weights().detach().cpu().numpy()
        return {name: float(wi) for name, wi in zip(var_names, w)}


class TemporalSelfAttention(nn.Module):
    def __init__(self, d, nh=4, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nh, dropout=drop, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(d * 4, d), nn.Dropout(drop))

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.n1(x + a)
        return self.n2(x + self.ff(x))


class HierarchicalAttention(nn.Module):
    def __init__(self, d, n_m, nl=3, nh=4, drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TemporalSelfAttention(d, nh, drop) for _ in range(nl)])
        self.fusion = nn.Sequential(nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU())
        self.gate_proj = nn.Linear(n_m, d)

    def forward(self, hf, hm, cw=None, cs=1.0):
        if cw is not None:
            g = torch.sigmoid(self.gate_proj(cw))
            hm = hm + cs * (hm * g.unsqueeze(0).unsqueeze(0))
        h = self.fusion(torch.cat([hf, hm], -1))
        for layer in self.layers:
            h = layer(h)
        return h


class HiCaGAT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['d_model']
        self.causal_mode = cfg.get('causal_mode', 'notears')

        self.encoder = MultiSourceEncoder(cfg['n_type_f'], cfg['n_type_m'], cfg['n_static'], d)

        self.notears = None
        self.sparse_mask = None
        if self.causal_mode == 'notears':
            self.notears = NOTEARSModule(
                cfg['n_type_m'] + 1,
                cfg.get('n_lags', 3),
                temporal_subsample=cfg.get('notears_subsample', 1),
                recon_noise=cfg.get('notears_noise', 0.0))
        elif self.causal_mode == 'sparse_mask':
            self.sparse_mask = LearnedSparseMask(
                cfg['n_type_m'],
                lambda_l1=cfg.get('mask_lambda_l1', 0.01),
                temperature=cfg.get('mask_temperature', 1.0))

        self.attn = HierarchicalAttention(d, cfg['n_type_m'], cfg.get('n_layers', 3),
                                          cfg.get('n_heads', 4), cfg.get('dropout', 0.1))

        self.et_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d // 2, 1))
        self.fd_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d // 2, 1))

        self.register_buffer('dag_alpha', torch.tensor(0.0))
        self.register_buffer('dag_rho', torch.tensor(1.0))
        self.current_epoch = 0

    def forward(self, batch):
        type_m_seq = batch['type_m_seq']
        if self.sparse_mask is not None:
            w = self.sparse_mask.get_weights()
            type_m_seq = type_m_seq * w.unsqueeze(0).unsqueeze(0)

        hf, hm, hs = self.encoder(
            batch['type_f_seq'], type_m_seq,
            batch['type_f_mask'], batch['type_m_mask'], batch['static'])

        cw = None
        met_et = None
        if self.notears is not None:
            et_seq = batch['type_f_seq'][:, :, 6:7]
            met_et = torch.cat([batch['type_m_seq'], et_seq], -1)
            cw = self.notears.get_causal_weights()
            cs = min(1.0, self.current_epoch / 15.0)
            h = self.attn(hf, hm, cw=cw, cs=cs)
        else:
            if self.sparse_mask is not None:
                cw = self.sparse_mask.get_causal_weights()
            h = self.attn(hf, hm, cw=None, cs=0.0)

        h_last = h[:, -1, :]
        et_pred = self.et_head(h_last).squeeze(-1)
        fd_logits = self.fd_head(h_last).squeeze(-1)

        return {
            'et_pred': et_pred,
            'fd_logits': fd_logits,
            'fd_prob': torch.sigmoid(fd_logits),
            'causal_weights': cw,
            'met_et_seq': met_et,
        }

    def update_dag(self, h):
        with torch.no_grad():
            self.dag_alpha += self.dag_rho * h
            if h > 0.25:
                self.dag_rho *= 2.0
            self.dag_rho = torch.clamp(self.dag_rho, max=1e6)


def build_hicagat(cfg):
    m = HiCaGAT(cfg)
    n = sum(p.numel() for p in m.parameters())
    print(f"HiCaGAT: {n:,} params (causal_mode={cfg.get('causal_mode', 'notears')})")
    return m


class BaselineLSTM(nn.Module):
    def __init__(self, n_in, d, n_s, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(n_in, d)
        self.lstm = nn.LSTM(d, d, 2, batch_first=True, dropout=drop)
        self.sp = nn.Linear(n_s, d)
        self.et_head = nn.Sequential(nn.Linear(d * 2, d // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d // 2, 1))
        self.fd_head = nn.Sequential(nn.Linear(d * 2, d // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d // 2, 1))

    def forward(self, batch):
        x = torch.cat([batch['type_f_seq'], batch['type_m_seq']], -1)
        h, _ = self.lstm(self.proj(x))
        h = torch.cat([h[:, -1, :], self.sp(batch['static'])], -1)
        fd_logits = self.fd_head(h).squeeze(-1)
        return {'et_pred': self.et_head(h).squeeze(-1), 'fd_logits': fd_logits,
                'fd_prob': torch.sigmoid(fd_logits),
                'causal_weights': None, 'met_et_seq': None}


class BaselineTransformer(nn.Module):
    def __init__(self, n_in, d, n_s, nl=3, nh=4, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(n_in, d)
        self.pe = TemporalPositionalEncoding(d)
        enc = nn.TransformerEncoderLayer(d, nh, d * 4, drop, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, nl)
        self.sp = nn.Linear(n_s, d)
        self.et_head = nn.Sequential(nn.Linear(d * 2, d // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d // 2, 1))
        self.fd_head = nn.Sequential(nn.Linear(d * 2, d // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d // 2, 1))

    def forward(self, batch):
        x = self.pe(self.proj(torch.cat([batch['type_f_seq'], batch['type_m_seq']], -1)))
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        h = self.enc(x, mask=mask)
        h = torch.cat([h[:, -1, :], self.sp(batch['static'])], -1)
        fd_logits = self.fd_head(h).squeeze(-1)
        return {'et_pred': self.et_head(h).squeeze(-1), 'fd_logits': fd_logits,
                'fd_prob': torch.sigmoid(fd_logits),
                'causal_weights': None, 'met_et_seq': None}


def compute_loss(model, out, batch, args, has_notears):
    dev = out['et_pred'].device
    L = {}

    etp, ett = out['et_pred'], batch['et_target']
    v = ~torch.isnan(ett) & (ett != 0)
    L['et'] = F.huber_loss(etp[v], ett[v]) if v.sum() > 0 else torch.tensor(0., device=dev)

    fl = out['fd_logits']
    ft = batch['fd_target']
    if getattr(model, 'fd_loss_fn', None) is not None:
        L['fd'] = model.fd_loss_fn(fl, ft)
    else:
        bce = F.binary_cross_entropy_with_logits(fl, ft, reduction='none')
        p = torch.sigmoid(fl)
        pt = torch.where(ft == 1, p, 1 - p)
        at = torch.where(ft == 1, 0.75, 0.25)
        L['fd'] = (at * (1 - pt) ** 2.0 * bce).mean()

    er = batch['et_ref']
    vr = ~torch.isnan(er)
    if vr.sum() > 0:
        L['physics'] = (F.relu(etp[vr] - er[vr] * 1.2) ** 2).mean()
    else:
        L['physics'] = torch.tensor(0., device=dev)

    total = args.lambda_et * L['et'] + args.lambda_fd * L['fd'] + args.lambda_physics * L['physics']

    if has_notears and getattr(model, 'notears', None) is not None:
        L['recon'] = model.notears.reconstruction_loss(out['met_et_seq'])
        total = total + args.lambda_recon * L['recon']

        h_W = model.notears.dag_loss()
        L['dag'] = model.dag_alpha * h_W + 0.5 * model.dag_rho * h_W ** 2
        total = total + args.lambda_dag * L['dag']

        L['sparse'] = model.notears.sparsity_loss()
        total = total + args.lambda_sparse * L['sparse']

    if getattr(model, 'sparse_mask', None) is not None:
        L['mask_l1'] = model.sparse_mask.sparsity_loss()
        total = total + L['mask_l1']

        L['mask_entropy'] = model.sparse_mask.entropy_loss()
        total = total + args.mask_lambda_entropy * L['mask_entropy']

    L['total'] = total
    return L


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='full',
                   choices=['full', 'no_disagree', 'no_causal', 'no_physics',
                            'daily_native_only', 'baseline_transformer', 'baseline_lstm',
                            'no_esi', 'no_esi_ensemble', 'no_label_proximal',
                            'daily_native_no_etref', 'mrpt'])
    p.add_argument('--data_dir', default=str(DATA_DIR))
    p.add_argument('--output_dir', default=str(RESULTS_DIR))
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--n_lags', type=int, default=7)
    p.add_argument('--causal_mode', default='notears', choices=['notears', 'sparse_mask', 'none'])
    p.add_argument('--notears_subsample', type=int, default=1)
    p.add_argument('--notears_noise', type=float, default=0.0)
    p.add_argument('--mask_lambda_l1', type=float, default=0.01)
    p.add_argument('--mask_temperature', type=float, default=1.0)
    p.add_argument('--mask_lambda_entropy', type=float, default=0.005)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_notears', type=float, default=5e-3)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lambda_et', type=float, default=0.5)
    p.add_argument('--lambda_fd', type=float, default=1.0)
    p.add_argument('--lambda_physics', type=float, default=0.2)
    p.add_argument('--lambda_recon', type=float, default=0.05)
    p.add_argument('--lambda_dag', type=float, default=0.01)
    p.add_argument('--lambda_sparse', type=float, default=0.001)
    p.add_argument('--fd_loss', default='focal', choices=['focal', 'asl', 'cb_focal'])
    return p.parse_args()


def apply_ablation(args, train_ds, raw_dicts):
    if args.model == 'no_disagree':
        print(f"Zeroing disagreement features {DISAGREE_IDX} in all splits")
        zero_features(train_ds, raw_dicts, DISAGREE_IDX)

    elif args.model == 'daily_native_only':
        print(f"Zeroing OpenET features {OPENET_IDX} in all splits")
        zero_features(train_ds, raw_dicts, OPENET_IDX)

    elif args.model == 'no_physics':
        print("Setting lambda_physics=0")
        args.lambda_physics = 0.0

    elif args.model == 'no_esi':
        print(f"Zeroing ESI features {ESI_IDX} in all splits")
        zero_features(train_ds, raw_dicts, ESI_IDX)

    elif args.model == 'no_esi_ensemble':
        print(f"Zeroing {ESI_ENSEMBLE_IDX} in all splits, lambda_physics=0")
        zero_features(train_ds, raw_dicts, ESI_ENSEMBLE_IDX)
        args.lambda_physics = 0.0

    elif args.model == 'no_label_proximal':
        print(f"Zeroing {LABEL_PROXIMAL_IDX} in all splits, lambda_physics=0")
        zero_features(train_ds, raw_dicts, LABEL_PROXIMAL_IDX)
        args.lambda_physics = 0.0

    elif args.model == 'daily_native_no_etref':
        print(f"Zeroing OpenET features {OPENET_IDX} and Type-M {ET_REF_IDX_M}, lambda_physics=0")
        zero_features(train_ds, raw_dicts, OPENET_IDX)
        zero_features(train_ds, raw_dicts, ET_REF_IDX_M, key='type_m_seq')
        args.lambda_physics = 0.0


def build_model(args, nf, nm, ns):
    if args.model == 'baseline_lstm':
        return BaselineLSTM(nf + nm, args.d_model, ns, args.dropout), 'none'

    if args.model == 'baseline_transformer':
        return BaselineTransformer(nf + nm, args.d_model, ns,
                                   args.n_layers, args.n_heads, args.dropout), 'none'

    if args.model == 'mrpt':
        from mrpt_model import AsymmetricLossBinary, ClassBalancedFocal, MultiRatePatchTransformer
        model = MultiRatePatchTransformer(
            n_type_f=nf, n_type_m=nm, n_static=ns,
            d_model=args.d_model, patch_f=15, patch_m=5,
            n_layers=2, n_heads=args.n_heads, dropout=args.dropout,
            causal=False)
        if args.fd_loss == 'asl':
            model.fd_loss_fn = AsymmetricLossBinary(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
        elif args.fd_loss == 'cb_focal':
            model.fd_loss_fn = ClassBalancedFocal(n_pos=119500, n_neg=3643000,
                                                  beta=0.9999, gamma=2.0)
        return model, 'none'

    causal_mode = 'none' if args.model == 'no_causal' else args.causal_mode
    model_cfg = {
        'n_type_f': nf, 'n_type_m': nm, 'n_static': ns,
        'd_model': args.d_model, 'n_layers': args.n_layers,
        'n_heads': args.n_heads, 'dropout': args.dropout,
        'n_lags': args.n_lags,
        'causal_mode': causal_mode,
        'notears_subsample': args.notears_subsample,
        'notears_noise': args.notears_noise,
        'mask_lambda_l1': args.mask_lambda_l1,
        'mask_temperature': args.mask_temperature,
    }
    model = build_hicagat(model_cfg)

    if causal_mode == 'notears':
        print(f"NOTEARS: subsample={args.notears_subsample}, noise={args.notears_noise}")
    elif causal_mode == 'sparse_mask':
        print(f"Sparse mask: lambda_l1={args.mask_lambda_l1}, temperature={args.mask_temperature}, "
              f"lambda_entropy={args.mask_lambda_entropy}")

    return model, causal_mode


def build_optimizer(model, args, causal_mode):
    if causal_mode == 'notears' and getattr(model, 'notears', None) is not None:
        notears_params = list(model.notears.parameters())
        other = [p for n, p in model.named_parameters() if 'notears' not in n]
        return optim.AdamW([{'params': other, 'lr': args.lr, 'weight_decay': 1e-4},
                            {'params': notears_params, 'lr': args.lr_notears, 'weight_decay': 0}])

    if causal_mode == 'sparse_mask' and getattr(model, 'sparse_mask', None) is not None:
        mask_params = list(model.sparse_mask.parameters())
        other = [p for n, p in model.named_parameters() if 'sparse_mask' not in n]
        return optim.AdamW([{'params': other, 'lr': args.lr, 'weight_decay': 1e-4},
                            {'params': mask_params, 'lr': args.lr * 10, 'weight_decay': 0}])

    return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    data_dir = Path(args.data_dir)
    with open(data_dir / 'config.pkl', 'rb') as f:
        cfg = pickle.load(f)

    nf, nm, ns = cfg['n_type_f'], cfg['n_type_m'], cfg['n_static']

    print("Loading data...")
    train_ds = load_train_dataset(data_dir)
    val_d = torch.load(data_dir / 'val_data.pt', weights_only=False)
    test_d = torch.load(data_dir / 'test_data.pt', weights_only=False)

    apply_ablation(args, train_ds, [val_d, test_d])

    val_ds = HiCaGATDataset(val_d)
    test_ds = HiCaGATDataset(test_d)
    del val_d, test_d

    n_workers = 0 if platform.system() == 'Windows' else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=n_workers, pin_memory=True)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    model, causal_mode = build_model(args, nf, nm, ns)
    model = model.to(dev)
    has_notears = causal_mode == 'notears'
    has_sparse_mask = causal_mode == 'sparse_mask'

    print(f"Model: {args.model}, Params: {sum(p.numel() for p in model.parameters()):,}, "
          f"causal_mode={causal_mode}")

    opt = build_optimizer(model, args, causal_mode)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)

    met_names = cfg.get('type_m_cols', [f'met{i}' for i in range(nm)])
    best_f1, best_ep, patience = 0, 0, 0
    history = []

    print(f"\n{'=' * 80}")
    print(f"  TRAINING: {args.model} (daily, 90d lookback, 14d horizon)")
    print(f"  lambda_et={args.lambda_et}, lambda_fd={args.lambda_fd}, lambda_physics={args.lambda_physics}")
    print(f"{'=' * 80}\n")

    for epoch in range(1, args.epochs + 1):
        if hasattr(model, 'current_epoch'):
            model.current_epoch = epoch

        t0 = time.time()
        model.train()
        eloss, nb = 0, 0
        preds_all, tgt_all = [], []

        for batch in train_loader:
            bd = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(bd)
            L = compute_loss(model, out, bd, args, has_notears)
            opt.zero_grad()
            L['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            eloss += L['total'].item()
            nb += 1
            preds_all.extend((out['fd_prob'] > 0.5).cpu().numpy())
            tgt_all.extend(bd['fd_target'].cpu().numpy())

        tr_loss = eloss / max(nb, 1)
        tr_f1 = f1_score(tgt_all, preds_all, zero_division=0)

        model.eval()
        vl, vn = 0, 0
        vp, vt, vep, vet = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                bd = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(bd)
                L = compute_loss(model, out, bd, args, has_notears)
                vl += L['total'].item()
                vn += 1
                vp.extend((out['fd_prob'] > 0.5).cpu().numpy())
                vt.extend(bd['fd_target'].cpu().numpy())
                vep.extend(out['et_pred'].cpu().numpy())
                vet.extend(bd['et_target'].cpu().numpy())

        val_loss = vl / max(vn, 1)
        val_f1 = f1_score(vt, vp, zero_division=0)
        val_prec = precision_score(vt, vp, zero_division=0)
        val_rec = recall_score(vt, vp, zero_division=0)

        etp, ett = np.array(vep), np.array(vet)
        v = ~np.isnan(ett) & (ett != 0)
        val_rmse = np.sqrt(np.mean((etp[v] - ett[v]) ** 2)) if v.sum() > 0 else 0
        val_r2 = 1 - np.sum((etp[v] - ett[v]) ** 2) / np.sum((ett[v] - ett[v].mean()) ** 2) if v.sum() > 0 else 0

        sched.step()
        dag_h = 0.
        if has_notears and getattr(model, 'notears', None) is not None:
            dag_h = model.notears.dag_loss().item()
            model.update_dag(dag_h)

        dt = time.time() - t0
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_f1': tr_f1,
                        'val_loss': val_loss, 'val_f1': val_f1, 'val_prec': val_prec,
                        'val_rec': val_rec, 'val_rmse': val_rmse, 'val_r2': val_r2,
                        'dag_h': dag_h, 'time': dt})

        print(f"E{epoch:3d}/{args.epochs} ({dt:.0f}s) | "
              f"Tr: L={tr_loss:.4f} F1={tr_f1:.3f} | "
              f"Va: L={val_loss:.4f} F1={val_f1:.3f} P={val_prec:.3f} R={val_rec:.3f} "
              f"RMSE={val_rmse:.3f} R2={val_r2:.3f} | DAG={dag_h:.4f}")

        if has_sparse_mask and getattr(model, 'sparse_mask', None) is not None and epoch % 5 == 0:
            w = model.sparse_mask.get_weights().detach().cpu().numpy()
            print("  Mask: " + ' '.join(f'{n}={wi:.3f}' for n, wi in zip(met_names, w)))

        if val_f1 > best_f1:
            best_f1, best_ep, patience = val_f1, epoch, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_f1': val_f1, 'args': vars(args), 'history': history},
                       out_dir / 'best_model.pt')
            print(f"  >> Best F1={best_f1:.4f}")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop E{epoch} (best E{best_ep} F1={best_f1:.4f})")
                break

    print(f"\n{'=' * 60}\n  TEST\n{'=' * 60}")
    ckpt = torch.load(out_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tp, tt, tep, tet, t_pids = [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            bd = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(bd)
            tp.extend(out['fd_prob'].cpu().numpy())
            tt.extend(bd['fd_target'].cpu().numpy())
            tep.extend(out['et_pred'].cpu().numpy())
            tet.extend(bd['et_target'].cpu().numpy())
            t_pids.extend(bd['patch_id'].cpu().numpy())

    tp, tt = np.array(tp), np.array(tt)
    tep, tet = np.array(tep), np.array(tet)
    t_pids = np.array(t_pids)
    v = ~np.isnan(tet) & (tet != 0)

    np.savez(out_dir / 'test_predictions.npz',
             fd_prob=tp, fd_target=tt, et_pred=tep, et_target=tet, patch_id=t_pids)

    best_test_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        f = f1_score(tt, (tp > t).astype(int), zero_division=0)
        if f > best_test_f1:
            best_test_f1, best_thresh = f, t

    preds_05 = (tp > 0.5).astype(int)
    metrics = {
        'f1_t05': float(f1_score(tt, preds_05, zero_division=0)),
        'prec_t05': float(precision_score(tt, preds_05, zero_division=0)),
        'rec_t05': float(recall_score(tt, preds_05, zero_division=0)),
        'f1_optimal': float(best_test_f1),
        'optimal_threshold': float(best_thresh),
        'et_rmse': float(np.sqrt(np.mean((tep[v] - tet[v]) ** 2))) if v.sum() > 0 else 0,
        'et_r2': float(1 - np.sum((tep[v] - tet[v]) ** 2) / np.sum((tet[v] - tet[v].mean()) ** 2)) if v.sum() > 0 else 0,
    }

    print(f"\n  F1 (t=0.5):     {metrics['f1_t05']:.4f}  P={metrics['prec_t05']:.4f}  R={metrics['rec_t05']:.4f}")
    print(f"  F1 (optimal):   {metrics['f1_optimal']:.4f}  (threshold={metrics['optimal_threshold']:.2f})")
    print(f"  ET: RMSE={metrics['et_rmse']:.4f}  R2={metrics['et_r2']:.4f}")

    with open(out_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)

    if has_notears and getattr(model, 'notears', None) is not None:
        W = model.notears.get_causal_matrix().detach().cpu().numpy()
        np.save(out_dir / 'causal_matrix.npy', W)
        cw = model.notears.get_causal_weights().detach().cpu().numpy()
        print("\n  Causal weights:")
        for n, w in zip(met_names, cw):
            print(f"    {n:10s}: {w:.4f} {'#' * int(w * 30)}")

    if has_sparse_mask and getattr(model, 'sparse_mask', None) is not None:
        w = model.sparse_mask.get_weights().detach().cpu().numpy()
        raw = model.sparse_mask.raw_weights.detach().cpu().numpy()
        order = np.argsort(-w)

        print("\n  Learned variable importance:")
        for idx in order:
            print(f"    {met_names[idx]:10s}: {w[idx]:.4f} (raw={raw[idx]:+.3f}) {'#' * int(w[idx] * 30)}")

        print(f"\n  Max weight:  {met_names[order[0]]} = {w[order[0]]:.4f}")
        print(f"  Min weight:  {met_names[order[-1]]} = {w[order[-1]]:.4f}")
        print(f"  Spread:      {w.max() - w.min():.4f}")
        print(f"  Gini coeff:  {np.abs(np.subtract.outer(w, w)).mean() / (2 * w.mean()):.4f}")

        np.save(out_dir / 'sparse_mask_weights.npy', w)
        with open(out_dir / 'sparse_mask_weights.json', 'w') as f:
            json.dump({met_names[i]: float(w[i]) for i in range(len(met_names))}, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  DONE: {args.model} | Best E{best_ep} F1={best_f1:.4f} | Test F1={metrics['f1_t05']:.4f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
