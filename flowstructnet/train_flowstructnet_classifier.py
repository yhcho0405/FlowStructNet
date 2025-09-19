#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math, os, random, copy
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from collections import Counter


# 분산/유틸
def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def setup_distributed(args):
    if not args.distributed:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, dist.get_world_size(), device

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def all_reduce_sum(t: torch.Tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)



# 경로/라벨 유틸
def _path_matches_fold_and_split(fp: Path, split: str, fold_name: Optional[str]) -> bool:
    parts = list(fp.parts)
    if split not in parts:
        return False
    if fold_name is None:
        return True
    for i, p in enumerate(parts):
        if p == fold_name and i+1 < len(parts) and parts[i+1] == split:
            return True
    return False

def build_label_map(data_root: str, splits=("train","val","test"), fold_name: Optional[str]=None) -> Dict[str,int]:
    root = Path(data_root).resolve()
    keys = set()
    for fp in sorted(root.rglob("*.jsonl")):
        for sp in splits:
            if not _path_matches_fold_and_split(fp, sp, fold_name):
                continue
            parts = list(fp.parts)
            try:
                si = parts.index(sp)
            except ValueError:
                continue
            cls_key_parts = parts[si+1:-1]
            if not cls_key_parts: continue
            keys.add("/".join(cls_key_parts))
            break
    if not keys:
        raise RuntimeError(f"No class directories under {data_root} (fold={fold_name}, splits={splits})")
    return {k:i for i,k in enumerate(sorted(keys))}



# DT 미드포인트 로딩
def load_dt_bin_mids(data_root: str) -> List[float]:
    cfg_path = Path(data_root) / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        edges = cfg.get("dt_ms_bins", [1,4,16,64,256,1024])
    else:
        edges = [1,4,16,64,256,1024]  # ms
    mids = []
    if edges:
        mids.append(edges[0]*0.5)
        for i in range(1, len(edges)):
            mids.append(0.5*(edges[i-1]+edges[i]))
        mids.append(edges[-1]*1.5)  # last open bin
    else:
        mids = [1,2,4,8,16,32,64]
    return mids  # len = #bins+1


# 데이터셋 / 콜레이터
class FlowJsonlDataset(Dataset):
    def __init__(self, data_root: str, split: str, label_map: Dict[str, int],
                 max_pos: int, fold_name: Optional[str]=None):
        self.root = Path(data_root).resolve()
        self.split = split
        self.label_map = label_map
        self.max_pos = max_pos
        self.fold_name = fold_name
        self.samples: List[Tuple[List[int], int]] = []
        self.lengths: List[int] = []

        for fp in sorted(self.root.rglob("*.jsonl")):
            if not _path_matches_fold_and_split(fp, split, fold_name):
                continue
            parts = list(fp.parts)
            try:
                si = parts.index(split)
            except ValueError:
                continue
            cls_key_parts = parts[si+1:-1]
            if not cls_key_parts: continue
            cls_key = "/".join(cls_key_parts)
            if cls_key not in self.label_map: continue
            y = self.label_map[cls_key]

            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    rec = json.loads(line)
                    ids = rec.get("input_ids")
                    if not ids: continue
                    # max_pos는 모델에서 다시 처리하므로 여기서는 전체 유지(길이 정보 보존)
                    self.samples.append((ids, y))
                    self.lengths.append(len(ids))

        if not self.samples:
            raise RuntimeError(f"No samples for split={split}, fold={fold_name}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class PacketCollator:
    def __init__(self, pad_id: int, stoi: Dict[str,int], itos: Dict[str,str], dt_mids: List[float]):
        self.pad_id = pad_id
        self.stoi = stoi
        self.itos = itos
        self.dt_mids = dt_mids
        # 키 토큰 id
        self.bos_id = stoi.get("[BOS]", None)
        self.eos_id = stoi.get("[EOS]", None)
        # 카테고리 맵
        self.dir_ids = {stoi.get("C2S", -1), stoi.get("S2C", -1)}
        # len/dt prefix 파싱
        self.len_map: Dict[int,int] = {}
        self.dt_map: Dict[int,int] = {}
        for tid, tok in itos.items():
            if tok.startswith("LEN_"):
                try:
                    i = int(tok.split("_")[1])
                    self.len_map[int(tid)] = i
                except: pass
            if tok.startswith("DT_"):
                try:
                    i = int(tok.split("_")[1])
                    self.dt_map[int(tid)] = i
                except: pass
        # 각 임베딩용 padding 인덱스
        self.dir_pad = 2
        self.len_pad = 10
        self.dt_pad  = 7

    def _parse_to_packets(self, ids: List[int]):
        prefix_end = 0
        if self.bos_id is not None:
            try:
                prefix_end = ids.index(self.bos_id) + 1
            except ValueError:
                prefix_end = 0
        prefix_ids = ids[:prefix_end] if prefix_end > 0 else []

        dirs, lens, dts = [], [], []
        cum_dt_vals = []
        t = 0.0
        i = prefix_end
        state = 0  # 0: expect dir, 1: len, 2: dt
        cur_dir = cur_len = cur_dt = None

        while i < len(ids):
            tid = int(ids[i])
            if self.eos_id is not None and tid == self.eos_id:
                break
            tok = self.itos.get(str(tid), None)

            if state == 0:
                if tid in self.dir_ids:
                    cur_dir = 0 if tid == self.stoi.get("C2S", -9999) else 1
                    state = 1
                # else skip
            elif state == 1:
                if tid in self.len_map:
                    cur_len = self.len_map[tid]  # 0..9
                    state = 2
                else:
                    state = 0  # resync
            else:  # state == 2
                if tid in self.dt_map:
                    cur_dt = self.dt_map[tid]    # 0..6
                    # finalize one packet
                    dirs.append(cur_dir)
                    lens.append(cur_len)
                    dts.append(cur_dt)
                    # cum dt
                    mid = self.dt_mids[cur_dt] if cur_dt < len(self.dt_mids) else float(cur_dt)
                    t += mid
                    cum_dt_vals.append(t)
                # regardless, resync to expect dir
                state = 0
            i += 1

        # 정규화: [0,1]
        if len(cum_dt_vals) > 0:
            last = max(cum_dt_vals[-1], 1e-6)
            cum_dt_vals = [x/last for x in cum_dt_vals]
        return prefix_ids, dirs, lens, dts, cum_dt_vals

    def __call__(self, batch):
        # 1) 시퀀스 파싱
        parsed = []
        max_prefix = 0
        max_packets = 0
        for ids, y in batch:
            ids = ids.tolist()
            pfx, dirs, lens, dts, cum_dt = self._parse_to_packets(ids)
            parsed.append((pfx, dirs, lens, dts, cum_dt, int(y.item())))
            max_prefix = max(max_prefix, len(pfx))
            max_packets = max(max_packets, len(dirs))

        B = len(parsed)
        # 2) 텐서 생성/패딩
        prefix_ids = torch.full((B, max_prefix), self.pad_id, dtype=torch.long)
        prefix_mask = torch.zeros((B, max_prefix), dtype=torch.bool)

        dir_ids = torch.full((B, max_packets), self.dir_pad, dtype=torch.long)
        len_ids = torch.full((B, max_packets), self.len_pad, dtype=torch.long)
        dt_ids  = torch.full((B, max_packets), self.dt_pad,  dtype=torch.long)
        pkt_mask = torch.zeros((B, max_packets), dtype=torch.bool)
        cum_dt = torch.zeros((B, max_packets), dtype=torch.float)

        labels = torch.zeros((B,), dtype=torch.long)

        for i, (pfx, d, l, dt, cd, y) in enumerate(parsed):
            if len(pfx) > 0:
                prefix_ids[i, :len(pfx)] = torch.tensor(pfx, dtype=torch.long)
                prefix_mask[i, :len(pfx)] = True
            if len(d) > 0:
                n = len(d)
                dir_ids[i, :n] = torch.tensor(d, dtype=torch.long)
                len_ids[i, :n] = torch.tensor(l, dtype=torch.long)
                dt_ids[i,  :n] = torch.tensor(dt, dtype=torch.long)
                pkt_mask[i, :n] = True
                cum_dt[i,  :n] = torch.tensor(cd, dtype=torch.float)
            labels[i] = y

        return prefix_ids, prefix_mask, dir_ids, len_ids, dt_ids, pkt_mask, cum_dt, labels


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(self, lengths: List[int], batch_size: int, shuffle=True, bucket_size=None, drop_last=False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = bucket_size or (batch_size*50)
        self.drop_last = drop_last

    def __iter__(self):
        n = len(self.lengths)
        idx = list(range(n))
        if self.shuffle: random.shuffle(idx)
        for i in range(0, n, self.bucket_size):
            bucket = idx[i:i+self.bucket_size]
            bucket.sort(key=lambda j: self.lengths[j], reverse=True)
            for j in range(0, len(bucket), self.batch_size):
                b = bucket[j:j+self.batch_size]
                if len(b) < self.batch_size and self.drop_last: continue
                yield b

    def __len__(self):
        n = len(self.lengths)
        return (n + self.batch_size - 1) // self.batch_size


# 모델: FlowStructNet
class SqueezeExcite(nn.Module):
    def __init__(self, d_model: int, se_ratio: float=0.25):
        super().__init__()
        hidden = max(8, int(d_model * se_ratio))
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
    def forward(self, x, mask):
        denom = mask.sum(1).clamp_min(1).unsqueeze(-1).float()
        mean = (x * mask.unsqueeze(-1).float()).sum(1) / denom
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(mean))))
        return x * s.unsqueeze(1)


class PacketComposer(nn.Module):
    def __init__(self, d_model: int, dir_pad=2, len_pad=10, dt_pad=7, dropout: float=0.0):
        super().__init__()
        self.dir_emb = nn.Embedding(3, d_model, padding_idx=dir_pad)  # 0,1 valid / 2 pad
        self.len_emb = nn.Embedding(11, d_model, padding_idx=len_pad) # 0..9 valid / 10 pad
        self.dt_emb  = nn.Embedding(8, d_model, padding_idx=dt_pad)   # 0..6 valid / 7 pad
        self.mlp = nn.Sequential(
            nn.Linear(3*d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False)
        )
        self.drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.dir_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.len_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.dt_emb.weight,  std=0.02)

    def forward(self, dir_ids, len_ids, dt_ids):
        ed = self.dir_emb(dir_ids)
        el = self.len_emb(len_ids)
        et = self.dt_emb(dt_ids)
        y = torch.cat([ed, el, et], dim=-1)
        y = self.mlp(y) + (ed + el + et)
        return self.drop(y)  # (B,P,D)


class ConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dilation: int,
                 dropout: float, mlp_mult: int=2, use_se: bool=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        pad = (kernel_size-1)//2 * dilation
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=pad, dilation=dilation, groups=d_model, bias=False)
        self.pw = nn.Conv1d(d_model, 2*d_model, kernel_size=1, bias=False)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model, bias=False), nn.Dropout(dropout),
        )
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcite(d_model)

    def forward(self, x, mask):
        y = self.norm1(x)
        y = y.transpose(1,2)
        y = self.dw(y)
        y = self.pw(y)
        a, b = torch.chunk(y, 2, dim=1)
        y = torch.tanh(a) * torch.sigmoid(b)
        y = y.transpose(1,2)
        x = x + self.drop1(y)
        y = self.ffn(self.norm2(x))
        if self.use_se: y = self.se(y, mask)
        x = x + y
        return x


class AttentionPool1D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))
    def forward(self, x, mask):
        score = torch.einsum("btd,d->bt", x, self.q)
        m = mask.bool()
        neg = torch.finfo(score.dtype).min
        score = score.masked_fill(~m, neg)
        w = torch.softmax(score.float(), dim=1).to(x.dtype).unsqueeze(-1)
        return (x * w).sum(1)


class FlowStructNet(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int,
                 d_model: int=384, n_layers: int=8,
                 kernel_size: int=9, dilation_base: int=2,
                 dropout: float=0.15, mlp_mult: int=2, use_se: bool=True,
                 max_pos: int=4096, pad_id: int=0,
                 pooling: str="meanmax_attn"):
        super().__init__()
        self.pad_id = pad_id
        self.max_pos = max_pos
        self.pooling = pooling

        # Prefix 임베딩(+세그먼트)
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.seg_emb = nn.Embedding(2, d_model)  # 0=prefix, 1=packet

        # 패킷 컴포저 + 시간 투영
        self.composer = PacketComposer(d_model, dropout=dropout)
        self.time_proj = nn.Linear(1, d_model, bias=False)

        # 위치 임베딩
        self.pos_emb = nn.Embedding(max_pos, d_model)

        self.drop_in = nn.Dropout(dropout)

        # dilation 스케줄
        dils = []
        d = 1
        for _ in range(n_layers):
            dils.append(d)
            d = min(d * dilation_base, max_pos // 2 if max_pos>0 else 512)
            if d < 1: d = 1

        self.blocks = nn.ModuleList([
            ConvBlock(d_model, kernel_size, dil, dropout, mlp_mult, use_se)
            for dil in dils
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # 풀링
        in_dim = {"mean": d_model, "attn": d_model, "meanmax": d_model*2,
                  "meanmax_attn": d_model*3}.get(pooling, None)
        if in_dim is None:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.attn_pool = AttentionPool1D(d_model) if "attn" in pooling else None

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes, bias=True)
        )

        self._init_params()

    def _init_params(self):
        nn.init.trunc_normal_(self.tok_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.seg_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_emb.weight, std=0.02)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _masked_mean(self, x, mask):
        denom = mask.sum(1).clamp_min(1).unsqueeze(-1).float()
        return (x * mask.unsqueeze(-1).float()).sum(1) / denom

    def _masked_max(self, x, mask):
        neg_inf = torch.finfo(x.dtype).min
        masked = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
        return masked.max(dim=1).values

    def forward(self, prefix_ids, prefix_mask, dir_ids, len_ids, dt_ids, pkt_mask, cum_dt):
        B, Tp = prefix_ids.shape
        _, P = dir_ids.shape

        # prefix 임베딩
        x_prefix = self.tok_emb(prefix_ids) + self.seg_emb(torch.zeros_like(prefix_ids))
        # packet 임베딩 (+ 시간)
        x_pkt = self.composer(dir_ids, len_ids, dt_ids) + self.seg_emb(torch.ones_like(dir_ids))
        if P > 0:
            time_feat = self.time_proj(cum_dt.unsqueeze(-1))  # (B,P,1)->(B,P,D)
            x_pkt = x_pkt + time_feat

        # 결합 및 위치/드롭
        x = torch.cat([x_prefix, x_pkt], dim=1)  # (B, Tp+P, D)
        mask = torch.cat([prefix_mask, pkt_mask], dim=1)     # (B, T)

        # max_pos 초과 시 패킷 부분을 잘라냄(프리픽스는 유지)
        T = x.size(1)
        if T > self.max_pos:
            keep_packets = max(0, self.max_pos - Tp)
            if keep_packets < 0:  # 이례적으로 prefix가 너무 긴 경우
                x = x[:, :self.max_pos]
                mask = mask[:, :self.max_pos]
            else:
                x = torch.cat([x[:, :Tp], x[:, Tp:Tp+keep_packets]], dim=1)
                mask = torch.cat([mask[:, :Tp], mask[:, Tp:Tp+keep_packets]], dim=1)
            T = x.size(1)

        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_emb(pos)
        # 임베딩 직후 PAD 0화
        x = x * mask.unsqueeze(-1).float()
        x = self.drop_in(x)

        # 컨브 블록
        for blk in self.blocks:
            x = blk(x, mask)
            # 블록 후 PAD 0화(누수 차단)
            x = x * mask.unsqueeze(-1).float()

        x = self.norm_out(x)

        # 풀링
        if self.pooling == "mean":
            pooled = self._masked_mean(x, mask)
        elif self.pooling == "attn":
            pooled = self.attn_pool(x, mask)
        elif self.pooling == "meanmax":
            pooled = torch.cat([self._masked_mean(x, mask),
                                self._masked_max(x, mask)], dim=-1)
        else:  # meanmax_attn
            pooled = torch.cat([
                self._masked_mean(x, mask),
                self._masked_max(x, mask),
                self.attn_pool(x, mask)
            ], dim=-1)

        logits = self.head(pooled)
        return logits



# EMA
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for e_p, p in zip(self.ema.parameters(), model.parameters()):
            e_p.copy_(e_p * d + p * (1.0 - d))
    def state_dict(self):
        return self.ema.state_dict()
    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)



# 스케줄러/루프
def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6, base_lr=None):
    if base_lr is None: base_lr = optimizer.param_groups[0]['lr']
    def lr_lambda(step):
        if warmup_steps>0 and step < warmup_steps:
            return max(1e-8, float(step+1)/float(max(1,warmup_steps)))
        if total_steps <= warmup_steps:
            return min_lr/base_lr
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr/base_lr, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler,
                    ce_loss, grad_clip=1.0, grad_accum_steps: int=1,
                    mixup_alpha: float=0.0, ema: 'ModelEMA|None'=None,
                    autocast_dtype="cuda"):
    model.train()
    total_loss=0.0; total_correct=0; total_count=0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        prefix_ids, prefix_mask, dir_ids, len_ids, dt_ids, pkt_mask, cum_dt, labels = batch
        prefix_ids = prefix_ids.to(device)
        prefix_mask = prefix_mask.to(device)
        dir_ids = dir_ids.to(device)
        len_ids = len_ids.to(device)
        dt_ids = dt_ids.to(device)
        pkt_mask = pkt_mask.to(device)
        cum_dt = cum_dt.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=autocast_dtype):
            if mixup_alpha and mixup_alpha > 0.0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(labels.size(0), device=labels.device)
                labels_b = labels[perm]
                logits = model(prefix_ids, prefix_mask, dir_ids, len_ids, dt_ids, pkt_mask, cum_dt)
                loss = (lam * ce_loss(logits, labels) + (1.0 - lam) * ce_loss(logits, labels_b)) / grad_accum_steps
            else:
                logits = model(prefix_ids, prefix_mask, dir_ids, len_ids, dt_ids, pkt_mask, cum_dt)
                loss = ce_loss(logits, labels) / grad_accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step+1) % grad_accum_steps == 0:
            if scaler:
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None: scheduler.step()
            if ema is not None:
                ema.update(model)

        with torch.no_grad():
            total_loss += (loss.item() * grad_accum_steps) * labels.size(0)
            total_correct += (logits.argmax(-1)==labels).sum().item()
            total_count += labels.size(0)

    return total_loss, total_correct, total_count


@torch.no_grad()
def eval_one_epoch(model, loader, device, ce_loss, autocast_dtype="cuda"):
    model.eval()
    total_loss=0.0; total_correct=0; total_count=0
    for batch in loader:
        prefix_ids, prefix_mask, dir_ids, len_ids, dt_ids, pkt_mask, cum_dt, labels = batch
        prefix_ids = prefix_ids.to(device)
        prefix_mask = prefix_mask.to(device)
        dir_ids = dir_ids.to(device)
        len_ids = len_ids.to(device)
        dt_ids = dt_ids.to(device)
        pkt_mask = pkt_mask.to(device)
        cum_dt = cum_dt.to(device)
        labels = labels.to(device)
        with torch.autocast(device_type=autocast_dtype):
            logits = model(prefix_ids, prefix_mask, dir_ids, len_ids, dt_ids, pkt_mask, cum_dt)
            loss = ce_loss(logits, labels)
        total_loss += loss.item()*labels.size(0)
        total_correct += (logits.argmax(-1)==labels).sum().item()
        total_count += labels.size(0)
    return total_loss, total_correct, total_count



# 로더 빌드
def build_loaders(data_root, train_split, val_split, label_map, fold_name,
                  pad_id, stoi, itos, dt_mids,
                  batch_size, num_workers, bucket_by_len, distributed):
    train_ds = FlowJsonlDataset(data_root, train_split, label_map, max_pos=10**9, fold_name=fold_name)
    val_fold = (fold_name if val_split != "test" else None)
    val_ds   = FlowJsonlDataset(data_root, val_split,   label_map, max_pos=10**9, fold_name=val_fold)

    collate = PacketCollator(pad_id, stoi, itos, dt_mids)

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=True, collate_fn=collate, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, sampler=val_sampler,
                                  num_workers=num_workers, pin_memory=True, collate_fn=collate, drop_last=False)
    else:
        if bucket_by_len:
            train_sampler = BucketBatchSampler(train_ds.lengths, batch_size=batch_size, shuffle=True)
            val_sampler   = BucketBatchSampler(val_ds.lengths,   batch_size=batch_size, shuffle=False)
            train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=num_workers,
                                      pin_memory=True, collate_fn=collate)
            val_loader   = DataLoader(val_ds, batch_sampler=val_sampler,   num_workers=num_workers,
                                      pin_memory=True, collate_fn=collate)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers,
                                      pin_memory=True, collate_fn=collate, drop_last=False)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                      pin_memory=True, collate_fn=collate, drop_last=False)
    return train_loader, val_loader



def main():
    ap = argparse.ArgumentParser()
    # 데이터/출력
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    ap.add_argument("--fold_name", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)

    # 모델
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--kernel_size", type=int, default=9)
    ap.add_argument("--dilation_base", type=int, default=2)
    ap.add_argument("--mlp_mult", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--use_se", action="store_true")
    ap.add_argument("--max_pos", type=int, default=4096)
    ap.add_argument("--pooling", type=str, default="meanmax_attn",
                    choices=["mean","attn","meanmax","meanmax_attn"])

    # 학습
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=3000)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_accum_steps", type=int, default=1)

    # 효율
    ap.add_argument("--bucket_by_len", action="store_true")

    # 일반화/재현
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)

    ap.add_argument("--mixup_alpha", type=float, default=0.0)
    ap.add_argument("--resume", type=str, default=None)

    ap.add_argument("--class_weight", type=str, default="none",
                    choices=["none","effective","balanced"])
    ap.add_argument("--class_weight_beta", type=float, default=0.9999)

    # 분산
    ap.add_argument("--distributed", action="store_true")

    args = ap.parse_args()
    local_rank, world_size, device = setup_distributed(args)
    set_seed(args.seed + local_rank)

    if is_main_process():
        out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    # vocab
    vocab_path = Path(args.data_root) / "vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    stoi = vocab_json["stoi"]
    itos = {str(k): v for k, v in vocab_json.get("itos", {}).items()}
    if not itos:  # 안전장치: stoi로부터 재구성 불가 → 필수
        raise RuntimeError("vocab.json 에 'itos' 필요")
    pad_id = stoi.get("[PAD]", 0); vocab_size = len(stoi)

    # DT bin mids
    dt_mids = load_dt_bin_mids(args.data_root)

    # label map
    if is_main_process():
        label_map_path = Path(args.out_dir) / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, "r", encoding="utf-8") as f:
                label_map = json.load(f)
        else:
            lm_splits = (args.train_split, args.val_split)
            label_map = build_label_map(args.data_root, splits=lm_splits, fold_name=args.fold_name)
            with open(label_map_path, "w", encoding="utf-8") as f:
                json.dump(label_map, f, ensure_ascii=False, indent=2)
    barrier()
    if not is_main_process():
        # 비마스터도 label_map 로드
        label_map_path = Path(args.out_dir) / "label_map.json"
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)

    num_classes = len(label_map)
    if is_main_process():
        print(f"[INFO] classes={num_classes}, vocab={vocab_size}, fold={args.fold_name}, world_size={world_size}")

    # loaders
    train_loader, val_loader = build_loaders(
        args.data_root, args.train_split, args.val_split,
        label_map, args.fold_name, pad_id, stoi, itos, dt_mids,
        args.batch_size, args.num_workers, args.bucket_by_len, args.distributed
    )

    # 모델
    torch.backends.cudnn.benchmark = True
    model = FlowStructNet(
        vocab_size=vocab_size, num_classes=num_classes,
        d_model=args.d_model, n_layers=args.n_layers,
        kernel_size=args.kernel_size, dilation_base=args.dilation_base,
        dropout=args.dropout, mlp_mult=args.mlp_mult, use_se=args.use_se,
        max_pos=args.max_pos, pad_id=pad_id, pooling=args.pooling
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    # DDP 래핑
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.num_epochs * steps_per_epoch // max(1, args.grad_accum_steps))
    scheduler = cosine_with_warmup(optimizer, args.warmup_steps, total_steps, min_lr=args.min_lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and torch.cuda.is_available()))
    ce_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    # 클래스 가중치 옵션
    if args.class_weight != "none":
        cnt = Counter()
        # rank 모두 같은 분포라 가볍게 한 번만
        for _, y in train_loader.dataset.samples:
            cnt[int(y)] += 1
        freq = torch.zeros(num_classes, dtype=torch.float)
        for c, n in cnt.items():
            freq[c] = float(n)
        weights = None
        if args.class_weight == "effective":
            beta = float(args.class_weight_beta)
            eff_num = 1.0 - torch.pow(torch.tensor(beta), freq)
            eff_num = eff_num.clamp_min(1e-8)
            weights = (1.0 - beta) / eff_num
            weights = (weights / weights.sum().clamp_min(1e-8)) * num_classes
        elif args.class_weight == "balanced":
            weights = 1.0 / freq.clamp_min(1.0)
            weights = (weights / weights.sum().clamp_min(1e-8)) * num_classes
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device),
                                      label_smoothing=args.label_smoothing).to(device)

    ema = ModelEMA(model_without_ddp, decay=args.ema_decay) if args.ema else None

    # resume
    start_epoch = 1
    if args.resume is not None and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception:
            pass
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass
        if ema is not None and "ema_state" in ckpt and ckpt["ema_state"] is not None:
            try:
                ema.load_state_dict(ckpt["ema_state"])
            except Exception:
                pass
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if is_main_process():
            print(f"[INFO] resumed from {args.resume} (start_epoch={start_epoch})")

    best_val_acc = 0.0
    autocast_dtype = "cuda" if device.type == "cuda" else "cpu"

    for epoch in range(start_epoch, args.num_epochs+1):
        if args.distributed:
            # DistributedSampler 사용 시 매 epoch 셔플 시드 지정
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

        tr_loss_sum, tr_correct_sum, tr_count_sum = train_one_epoch(
            #model_without_ddp if ema is None else model_without_ddp,  # forward는 래핑된 model로 수행되지만 loss 집계용은 무관
            model,
            train_loader, optimizer, scheduler, device, scaler, ce_loss,
            grad_clip=args.grad_clip, grad_accum_steps=args.grad_accum_steps,
            mixup_alpha=args.mixup_alpha, ema=ema, autocast_dtype=autocast_dtype
        )

        # DDP 집계
        tr_loss_sum_t = torch.tensor([tr_loss_sum], device=device)
        tr_correct_sum_t = torch.tensor([tr_correct_sum], device=device)
        tr_count_sum_t = torch.tensor([tr_count_sum], device=device)
        all_reduce_sum(tr_loss_sum_t); all_reduce_sum(tr_correct_sum_t); all_reduce_sum(tr_count_sum_t)
        tr_loss = (tr_loss_sum_t.item() / max(1, tr_count_sum_t.item()))
        tr_acc  = (tr_correct_sum_t.item() / max(1, tr_count_sum_t.item()))

        # 평가: EMA가 있으면 EMA, 없으면 본 모델 파라미터
        eval_model = ema.ema if ema is not None else model_without_ddp
        va_loss_sum, va_correct_sum, va_count_sum = eval_one_epoch(eval_model, val_loader, device, ce_loss, autocast_dtype=autocast_dtype)
        va_loss_sum_t = torch.tensor([va_loss_sum], device=device)
        va_correct_sum_t = torch.tensor([va_correct_sum], device=device)
        va_count_sum_t = torch.tensor([va_count_sum], device=device)
        all_reduce_sum(va_loss_sum_t); all_reduce_sum(va_correct_sum_t); all_reduce_sum(va_count_sum_t)
        va_loss = (va_loss_sum_t.item() / max(1, va_count_sum_t.item()))
        va_acc  = (va_correct_sum_t.item() / max(1, va_count_sum_t.item()))

        if is_main_process():
            print(f"[Epoch {epoch:02d}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")

            # 저장(EMA 기준)
            save_state = {
                "model_state": eval_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "ema_state": (ema.state_dict() if ema is not None else None),
                "args": vars(args),
                "label_map": label_map,
                "vocab_size": vocab_size,
                "pad_id": pad_id,
                "epoch": epoch,
                "val_acc": va_acc
            }
            out_dir = Path(args.out_dir)
            torch.save(save_state, out_dir / f"checkpoint-epoch{epoch:02d}.pt")
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(save_state, out_dir / f"checkpoint-best.pt")
                print(f"  [*] best updated: acc={best_val_acc:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()


'''

CUDA_VISIBLE_DEVICES=1 python train_packetflownet_classifier.py \ --data_root /home/younghyo/hyena/data/processed/flow-level-classification/tls \ --fold_name train_val_split_1 \ --train_split train --val_split val \ --out_dir ./runs/pfn_w512_d6_k11_ctx4096_mm_ema_f1 \ --num_epochs 100 --batch_size 16 \ --d_model 512 --n_layers 6 \ --kernel_size 11 --dilation_base 2 \ --dropout 0.20 --mlp_mult 2 --use_se --pooling meanmax \ --lr 3e-4 --min_lr 5e-7 --warmup_steps 1500 \ --label_smoothing 0.02 --weight_decay 0.05 \ --max_pos 4096 \ --num_workers 10 --bucket_by_len \ --grad_accum_steps 1 --fp16 --ema --ema_decay 0.9995

'''