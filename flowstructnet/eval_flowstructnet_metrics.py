#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from flowstructnet.train_flowstructnet_classifier import (
    FlowStructNet, FlowJsonlDataset, PacketCollator, load_dt_bin_mids
)

# ---------- metrics helpers ----------
def confusion_metrics(cm: np.ndarray):
    """cm: (K,K), rows=true, cols=pred"""
    K = cm.shape[0]
    support = cm.sum(axis=1)               # true count per class
    preds   = cm.sum(axis=0)               # pred count per class
    tp = np.diag(cm)
    fn = support - tp
    fp = preds   - tp
    tn = cm.sum() - (tp + fp + fn)

    # per-class precision/recall/f1
    eps = 1e-12
    prec_c = tp / np.maximum(tp + fp, eps)
    rec_c  = tp / np.maximum(tp + fn, eps)
    f1_c   = 2 * prec_c * rec_c / np.maximum(prec_c + rec_c, eps)

    # macro/micro/weighted
    macro_prec = np.nanmean(prec_c)
    macro_rec  = np.nanmean(rec_c)
    macro_f1   = np.nanmean(f1_c)

    # micro
    TP = tp.sum(); FP = fp.sum(); FN = fn.sum()
    micro_prec = TP / max(TP + FP, eps)
    micro_rec  = TP / max(TP + FN, eps)
    micro_f1   = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, eps)

    # weighted
    total = support.sum()
    weighted_prec = (prec_c * support).sum() / max(total, eps)
    weighted_rec  = (rec_c  * support).sum() / max(total, eps)
    weighted_f1   = (f1_c   * support).sum() / max(total, eps)

    # accuracy / balanced accuracy
    acc = tp.sum() / max(total, eps)
    recalls = np.divide(tp, np.maximum(support, eps), out=np.zeros_like(tp, dtype=float), where=support>0)
    balanced_acc = np.nanmean(recalls)

    # Cohen's kappa
    s = total
    pe = (preds * support).sum() / max(s*s, eps)    # expected agreement
    po = acc
    kappa = (po - pe) / max(1.0 - pe, eps)

    # 
    c = tp.sum()
    sum_pk_tk = (preds * support).sum()
    mcc_num = c * s - sum_pk_tk
    mcc_den = np.sqrt( (s**2 - (preds**2).sum()) * (s**2 - (support**2).sum()) )
    mcc = mcc_num / max(mcc_den, eps)

    return {
        "per_class": {
            "precision": prec_c.tolist(),
            "recall":    rec_c.tolist(),
            "f1":        f1_c.tolist(),
            "support":   support.tolist(),
            "tp":        tp.tolist(),
            "fp":        fp.tolist(),
            "fn":        fn.tolist(),
            "tn":        tn.tolist(),
        },
        "aggregates": {
            "accuracy": acc,
            "top1": acc,  # 동치
            "macro_precision": macro_prec, "macro_recall": macro_rec, "macro_f1": macro_f1,
            "micro_precision": micro_prec, "micro_recall": micro_rec, "micro_f1": micro_f1,
            "weighted_precision": weighted_prec, "weighted_recall": weighted_rec, "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc,
            "cohen_kappa": kappa,
            "mcc": mcc,
        }
    }

def topk_accuracy(logits: np.ndarray, y_true: np.ndarray, k=5):
    topk = np.argpartition(-logits, kth=k-1, axis=1)[:, :k]
    # 정확히 top-k 안에 정답이 있는지
    correct = np.any(topk == y_true[:, None], axis=1).mean()
    return correct

def try_macro_auc_ovr(probs: np.ndarray, y_true: np.ndarray):
    try:
        from sklearn.metrics import roc_auc_score
        K = probs.shape[1]
        y_true_ovr = np.eye(K, dtype=int)[y_true]
        auc_macro = roc_auc_score(y_true_ovr, probs, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(y_true_ovr, probs, average="weighted", multi_class="ovr")
        return auc_macro, auc_weighted
    except Exception as e:
        return None, None

# ---------- main ----------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--fold_name", required=True)          # train_val_split_0
    ap.add_argument("--split", default="test")             # usually "test"
    ap.add_argument("--ckpt", required=True)               # checkpoint-best.pt
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out_dir", type=str, default=None)   # where to save CSV/JSON
    ap.add_argument("--save_logits", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ck_args = ckpt.get("args", {})
    label_map = ckpt["label_map"]
    id2name = [None]*len(label_map)
    for name, idx in label_map.items():
        id2name[idx] = name

    vocab_path = Path(args.data_root) / "vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    stoi = vocab_json["stoi"]
    itos = {str(k): v for k, v in vocab_json.get("itos", {}).items()}
    pad_id = stoi.get("[PAD]", 0)
    vocab_size = len(stoi)

    fold_for_ds = None if args.split.lower() == "test" else args.fold_name
    dt_mids = load_dt_bin_mids(args.data_root)
    dataset = FlowJsonlDataset(args.data_root, args.split, label_map,
                               max_pos=10**9, fold_name=fold_for_ds)
    collate = PacketCollator(pad_id, stoi, itos, dt_mids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    model = FlowStructNet(
        vocab_size=vocab_size,
        num_classes=len(label_map),
        d_model=int(ck_args.get("d_model", 384)),
        n_layers=int(ck_args.get("n_layers", 6)),
        kernel_size=int(ck_args.get("kernel_size", 7)),
        dilation_base=int(ck_args.get("dilation_base", 2)),
        dropout=float(ck_args.get("dropout", 0.15)),
        mlp_mult=int(ck_args.get("mlp_mult", 2)),
        use_se=bool(ck_args.get("use_se", False)),
        max_pos=int(ck_args.get("max_pos", 2048)),
        pad_id=pad_id,
        pooling=str(ck_args.get("pooling", "meanmax_attn")),
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state"], strict=True)

    y_true = []
    y_pred = []
    all_logits = []

    amp_ctx = torch.amp.autocast("cuda") if (args.fp16 and device.type=="cuda") else torch.autocast("cpu", enabled=False)

    with amp_ctx:
        for batch in loader:
            prefix_ids, prefix_mask, dir_ids, len_ids, dt_ids, pkt_mask, cum_dt, labels = batch
            prefix_ids = prefix_ids.to(device, non_blocking=True)
            prefix_mask = prefix_mask.to(device, non_blocking=True)
            dir_ids = dir_ids.to(device, non_blocking=True)
            len_ids = len_ids.to(device, non_blocking=True)
            dt_ids  = dt_ids.to(device, non_blocking=True)
            pkt_mask = pkt_mask.to(device, non_blocking=True)
            cum_dt = cum_dt.to(device, non_blocking=True)

            logits = model(prefix_ids, prefix_mask, dir_ids, len_ids, dt_ids, pkt_mask, cum_dt)
            all_logits.append(logits.float().cpu().numpy())
            y_pred.append(logits.argmax(-1).cpu().numpy())
            y_true.append(labels.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    # confusion matrix & metrics
    K = len(id2name)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    metrics = confusion_metrics(cm)
    metrics["aggregates"]["top5"] = topk_accuracy(logits, y_true, k=5) if K >= 5 else None

    auc_macro, auc_weighted = try_macro_auc_ovr(probs, y_true)
    metrics["aggregates"]["macro_auc_ovr"] = auc_macro
    metrics["aggregates"]["weighted_auc_ovr"] = auc_weighted

    # ---- print summary ----
    agg = metrics["aggregates"]
    print(f"[EVAL] fold={args.fold_name} split={args.split}")
    print(f"  Top-1 Acc:        {agg['top1']:.4f}")
    if agg["top5"] is not None:
        print(f"  Top-5 Acc:        {agg['top5']:.4f}")
    print(f"  Macro  P/R/F1:    {agg['macro_precision']:.4f} / {agg['macro_recall']:.4f} / {agg['macro_f1']:.4f}")
    print(f"  Micro  P/R/F1:    {agg['micro_precision']:.4f} / {agg['micro_recall']:.4f} / {agg['micro_f1']:.4f}")
    print(f"  Weighted P/R/F1:  {agg['weighted_precision']:.4f} / {agg['weighted_recall']:.4f} / {agg['weighted_f1']:.4f}")
    print(f"  Balanced Acc:     {agg['balanced_accuracy']:.4f}")
    print(f"  Cohen's kappa:    {agg['cohen_kappa']:.4f}")
    print(f"  MCC (multiclass): {agg['mcc']:.4f}")
    if auc_macro is not None:
        print(f"  Macro AUC (OvR):  {auc_macro:.4f}")
        print(f"  W-avg AUC (OvR):  {auc_weighted:.4f}")
    else:
        print("  Macro/W-avg AUC (OvR): scikit-learn 미설치 또는 계산 불가 → 건너뜀")

    # ---- save files ----
    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.ckpt).parent / f"eval_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # per-class.csv
    pc = metrics["per_class"]
    with open(out_dir / "per_class.csv", "w", encoding="utf-8") as f:
        f.write("class_id,class_name,support,precision,recall,f1,tp,fp,fn,tn\n")
        for i in range(K):
            cname = id2name[i] if i < len(id2name) else str(i)
            f.write(f"{i},{cname},{pc['support'][i]},{pc['precision'][i]:.6f},{pc['recall'][i]:.6f},{pc['f1'][i]:.6f},{pc['tp'][i]},{pc['fp'][i]},{pc['fn'][i]},{pc['tn'][i]}\n")

    # confusion_matrix.csv
    with open(out_dir / "confusion_matrix.csv", "w", encoding="utf-8") as f:
        f.write(",".join(["true\\pred"] + [id2name[i] for i in range(K)]) + "\n")
        for i in range(K):
            row = [id2name[i]] + [str(int(x)) for x in cm[i]]
            f.write(",".join(row) + "\n")

    # (optional) logits / probs / preds
    if args.save_logits:
        np.save(out_dir / "logits.npy", logits)
        np.save(out_dir / "probs.npy", probs)
        np.save(out_dir / "y_true.npy", y_true)
        np.save(out_dir / "y_pred.npy", y_pred)

    print(f"[EVAL] saved: {out_dir}/metrics.json, per_class.csv, confusion_matrix.csv")
    if args.save_logits:
        print(f"[EVAL] saved arrays: logits.npy, probs.npy, y_true.npy, y_pred.npy")

if __name__ == "__main__":
    main()
