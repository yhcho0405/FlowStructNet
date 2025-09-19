#!/usr/bin/env bash
set -euo pipefail

IN="/path/to/pcap_root"
OUT="/path/to/out_dir"
MAX=4096
W=8

flowstructnet-convert   --input "$IN"   --output "$OUT"   --max-ctx-tokens $MAX   --workers $W   --resume
