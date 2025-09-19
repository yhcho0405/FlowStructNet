# FlowStructNet

## Install
```bash
pip install -e .
```

## Dataset

Public dataset folder (Google Drive):
- https://drive.google.com/drive/folders/1BUo5TMRuXNvTqNYy0RLeHk4l4Q3BuzSk

## Dataset Split(Required)

```python
# Splitter: https://github.com/SmartData-Polito/Debunk_Traffic_Representation/blob/master/process_finetune_data/Split/per-flow-split/flow_classification/split_based_flow.ipynb

dataset = 'tls'
dataset_path = './data/raw/TLS/sessions'
output_path  = './data/flows/TLS/flow' 
k = 3          # K-fold
threshold = 5
```
> paper: https://dl.acm.org/doi/10.1145/3718958.3750498

## Convert PCAP → JSONL
```bash
flowstructnet-convert   --input /path/to/pcap_root   --output /path/to/out_dir   --max-ctx-tokens 4096   --workers 8 --resume
```

## Train
```bash
flowstructnet-train   --data_root /path/to/out_dir   --fold_name train_val_split_0   --train_split train --val_split val   --out_dir ./runs/exp1
```

## Eval
```bash
flowstructnet-eval   --data_root /path/to/out_dir   --fold_name train_val_split_0   --split test   --ckpt ./runs/exp1/checkpoint-best.pt
```

License: Apache-2.0


## Environment
- Python: 3.10.18
- Torch: 2.5.1+cu121

Conda (recommended):
```bash
conda env create -f environment.yml
conda activate flowstructnet
pip install -e .
```

Pip (alternative):
```bash
python -m pip install -r requirements.lock.txt
python -m pip install -e .
```