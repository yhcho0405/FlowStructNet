#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from scapy.all import PcapReader, IP, IPv6, TCP, UDP
except Exception:
    print("scapy is required. install: pip install scapy", file=sys.stderr)
    raise

def build_vocab():
    vocab = []
    vocab += ["[PAD]", "[CLS]", "[BOS]", "[EOS]"]
    vocab += ["C2S", "S2C"]
    vocab += [f"LEN_{i}" for i in range(10)]
    vocab += [f"DT_{i}" for i in range(7)]
    stoi = {t:i for i,t in enumerate(vocab)}
    itos = {i:t for t,i in stoi.items()}
    return stoi, itos

STOI, ITOS = build_vocab()
LEN_BINS = [64, 128, 256, 512, 768, 1024, 1280, 1519, 2048]
DT_MS_BINS = [1, 4, 16, 64, 256, 1024]

def bin_index(value: float, edges: List[float]) -> int:
    for i, e in enumerate(edges):
        if value <= e: return i
    return len(edges)

def len_to_token(length_bytes: int) -> str:
    return f"LEN_{bin_index(length_bytes, LEN_BINS)}"

def dt_to_token(dt_ms: float) -> str:
    return f"DT_{bin_index(dt_ms, DT_MS_BINS)}"

@dataclass
class PacketRec:
    ts: float
    length: int
    src: Tuple[str,int]
    dst: Tuple[str,int]
    is_c2s: Optional[bool] = None
    dt_ms: Optional[float] = None
    tcp_flags: Optional[int] = None

@dataclass
class Flow:
    key: Tuple[str,str,int,int,str]
    proto: str
    ipver: str
    first_client: Tuple[str,int]
    pkts: List[PacketRec] = field(default_factory=list)
    c_bytes: int = 0
    s_bytes: int = 0
    c_pkts: int = 0
    s_pkts: int = 0
    def update(self):
        prev_ts = None
        for p in self.pkts:
            p.is_c2s = (p.src == self.first_client)
            if p.is_c2s:
                self.c_bytes += p.length; self.c_pkts += 1
            else:
                self.s_bytes += p.length; self.s_pkts += 1
            p.dt_ms = 0.0 if prev_ts is None else min(max(0.0, p.ts - prev_ts)*1000.0, 1500.0)
            prev_ts = p.ts

def parse_pcaps_to_flows(pcap_path: Path, udp_timeout_s: float = 120.0) -> List[Flow]:
    flows: Dict[Tuple[str,str,int,int,str], Flow] = {}
    last_seen: Dict[Tuple[str,str,int,int,str], float] = {}
    def flush_udp(ts_now: float):
        to_del = []
        for k, tlast in list(last_seen.items()):
            if k[-1] == "UDP" and ts_now - tlast > udp_timeout_s:
                to_del.append(k)
        for k in to_del: last_seen.pop(k, None)
    with PcapReader(str(pcap_path)) as rd:
        for pkt in rd:
            try: ts = float(pkt.time)
            except Exception: continue
            if not pkt.haslayer(IP) and not pkt.haslayer(IPv6): continue
            ipver = "IPV4" if pkt.haslayer(IP) else "IPV6"
            ip = pkt[IP] if ipver == "IPV4" else pkt[IPv6]
            src_ip, dst_ip = ip.src, ip.dst
            proto = None; sport = dport = None; tcp_flags = None
            if pkt.haslayer(TCP):
                t = pkt[TCP]; proto = "TCP"; sport, dport = int(t.sport), int(t.dport); tcp_flags = int(t.flags)
            elif pkt.haslayer(UDP):
                u = pkt[UDP]; proto = "UDP"; sport, dport = int(u.sport), int(u.dport)
            else:
                continue
            length = None
            if hasattr(pkt, 'wirelen') and isinstance(pkt.wirelen, int): length = pkt.wirelen
            elif hasattr(pkt, 'original') and isinstance(pkt.original, (bytes, bytearray)): length = len(pkt.original)
            else:
                try: length = len(bytes(pkt))
                except Exception: length = int(getattr(ip, 'len', 0)) or 0
            key  = (src_ip, dst_ip, sport, dport, proto)
            rkey = (dst_ip, src_ip, dport, sport, proto)
            fkey = key if key in flows else (rkey if rkey in flows else key)
            if fkey not in flows:
                flows[fkey] = Flow(key=fkey, proto=proto, ipver=ipver, first_client=(src_ip, sport))
            flow = flows[fkey]
            flow.pkts.append(PacketRec(ts=ts, length=length, src=(src_ip, sport), dst=(dst_ip, dport), tcp_flags=tcp_flags))
            last_seen[fkey] = ts; flush_udp(ts)
    for f in flows.values(): f.update()
    return list(flows.values())

def head_stride_packets(pkts: List[PacketRec], head_n: int, stride_every: int) -> List[PacketRec]:
    if len(pkts) <= head_n: return pkts
    head, tail = pkts[:head_n], pkts[head_n:]
    return head + [p for i, p in enumerate(tail) if (i % stride_every) == 0]

def build_tokens_for_flow(flow: Flow, max_ctx_tokens: int, head_keep_packets: int, stride_tail_every: int, stats_sink: Dict):
    tokens: List[str] = []
    tokens.append("[CLS]")
    tokens.append("[BOS]")
    pkts = flow.pkts
    max_packet_tokens = max(0, max_ctx_tokens - 3 - len(tokens))
    max_packets = max_packet_tokens // 3
    if max_packets < len(pkts):
        pkts = head_stride_packets(pkts, head_keep_packets, stride_tail_every)
    used = 0
    for p in pkts:
        if used >= max_packets: break
        tokens.extend(["C2S" if p.is_c2s else "S2C",
                       len_to_token(p.length),
                       dt_to_token(p.dt_ms if p.dt_ms is not None else 0.0)])
        used += 1
    tokens.append("[EOS]")
    stats_sink.setdefault("flow_count", Counter())["flows"] += 1
    input_ids = [STOI.get(t, 0) for t in tokens]
    return tokens, input_ids, used

def mirror_path(in_root: Path, out_root: Path, file_path: Path, new_ext: str) -> Path:
    rel = file_path.relative_to(in_root).with_suffix(new_ext)
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def discover_pcaps(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".pcap", ".pcapng")])

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def hash_flow(flow: Flow) -> str:
    h = hashlib.sha1()
    k = f"{flow.key[0]}:{flow.key[1]}:{flow.key[2]}:{flow.key[3]}:{flow.key[4]}:{len(flow.pkts)}"
    h.update(k.encode("utf-8"))
    return h.hexdigest()[:16]

def process_single_pcap(pcap_file: Path, in_root: Path, out_root: Path, max_ctx_tokens: int, head_keep_packets: int, stride_tail_every: int):
    out_jsonl = mirror_path(in_root, out_root, pcap_file, ".jsonl")
    out_tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".part")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    try:
        flows = parse_pcaps_to_flows(pcap_file)
    except Exception as e:
        return {"ok": False, "pcap": str(pcap_file), "error": str(e)}
    per_stats: Dict[str, Counter] = defaultdict(Counter)
    used_flows = 0
    with open(out_tmp, "w", encoding="utf-8") as f:
        for flow in flows:
            sink = defaultdict(Counter)
            _, ids, used = build_tokens_for_flow(flow, max_ctx_tokens, head_keep_packets, stride_tail_every, sink)
            for k, c in sink.items():
                per_stats[k].update(c)
            rec = {"input_ids": ids, "seq_len": len(ids),
                   "num_packets_used": used,
                   "source": {"pcap_relpath": str(pcap_file.relative_to(in_root)).replace(os.sep, "/"),
                              "flow_hash": hash_flow(flow)}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
            used_flows += 1
    os.replace(out_tmp, out_jsonl)
    rel_dir = str(pcap_file.relative_to(in_root).parent).replace(os.sep, "/")
    return {"ok": True, "pcap": str(pcap_file), "class_key": rel_dir, "stats": per_stats, "flows": used_flows}

def run_convert_parallel(in_root: Path, out_root: Path, workers: int, resume: bool,
                         max_ctx_tokens: int, head_keep_packets: int, stride_tail_every: int):
    pcaps = discover_pcaps(in_root)
    if not pcaps:
        print("no pcap/pcapng found under input", file=sys.stderr)
        return
    if resume:
        kept = []
        for p in pcaps:
            out_jsonl = mirror_path(in_root, out_root, p, ".jsonl")
            out_part  = out_jsonl.with_suffix(out_jsonl.suffix + ".part")
            if out_part.exists():
                try: out_part.unlink()
                except Exception: pass
            if out_jsonl.exists() and out_jsonl.stat().st_size > 0:
                continue
            kept.append(p)
        pcaps = kept
        print(f"[RESUME] remaining pcaps: {len(pcaps)}", flush=True)

    save_json(out_root / "vocab.json", {"stoi": STOI, "itos": ITOS})
    save_json(out_root / "config.json", {"len_bins": LEN_BINS, "dt_ms_bins": DT_MS_BINS,
                                         "max_ctx_tokens": max_ctx_tokens,
                                         "head_keep_packets": head_keep_packets,
                                         "stride_tail_every": stride_tail_every})

    overall: Dict[str, Counter] = defaultdict(Counter)
    class_stats: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_single_pcap, p, in_root, out_root,
                          max_ctx_tokens, head_keep_packets, stride_tail_every) for p in pcaps]
        for fu in as_completed(futs):
            r = fu.result()
            if not r.get("ok", False):
                print(f"[ERR] {r.get('pcap')}: {r.get('error')}", file=sys.stderr)
                continue
            ck = r["class_key"]; stats = r["stats"]
            for k, c in stats.items():
                overall[k].update(c)
                class_stats[ck][k].update(c)

    for cls_key, stats in class_stats.items():
        out_dir = out_root / cls_key
        save_json(out_dir / "_stats.json", {k: dict(v) for k, v in stats.items()})
    save_json(out_root / "overall_stats.json", {k: dict(v) for k, v in overall.items()})

def parse_args():
    ap = argparse.ArgumentParser(description="PCAP → token sequence converter (parallel)")
    ap.add_argument("--input", "-i", type=str, required=True)
    ap.add_argument("--output", "-o", type=str, required=True)
    ap.add_argument("--max-ctx-tokens", type=int, default=4096)
    ap.add_argument("--head-keep-packets", type=int, default=256)
    ap.add_argument("--stride-tail-every", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    run_convert_parallel(in_root, out_root, workers=args.workers, resume=args.resume,
                         max_ctx_tokens=args.max_ctx_tokens,
                         head_keep_packets=args.head_keep_packets,
                         stride_tail_every=args.stride_tail_every)

if __name__ == "__main__":
    main()
