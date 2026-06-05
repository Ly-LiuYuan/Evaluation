# Evaluation
This module provides an official evaluation pipeline for Joint Audio-Video Generation (JAVG) models, based on the JavisBench benchmark from *JavisDiT (ICLR 2026)*. It computes 14 objective metrics across four dimensions: generation quality, semantic consistency, audio-visual semantic consistency, and temporal synchrony.
## 1. Introduction

This evaluation module is designed to standardize the assessment of JAVG models. It relies on pretrained feature extractors and a reference set of 10,140 real videos from JavisBench. The pipeline is fully automated and requires only a set of generated videos and their corresponding text prompts.

## 2. Installation

```bash
# 1. Clone this repository
git clone https://github.com/Ly-LiuYuan/Evaluation
cd Evaluation

# 2. Create conda environment
conda create -n javis_eval python=3.10 -y
conda activate javis_eval

# 3. Install Python dependencies
pip install -r requirements_eval.txt

# 4. Install system FFmpeg
apt-get install -y ffmpeg

# 5. Download JavisBench evaluation source code
git clone https://github.com/JavisVerse/JavisDiT.git temp_javis
cp -r temp_javis/eval/javisbench ./javisbench
rm -rf temp_javis
```
The javisbench/ directory is not included in this repository; it must be fetched from the official JavisDiT repository.


## 3. Pretrained Weights

Download the following files into `checkpoints/`.  
(You can create this directory with `mkdir -p checkpoints`.)

| File | Approx. Size | Purpose | Official Download Link |
|------|-------------|---------|------------------------|
| `i3d_pretrained_400.pt` | 49 MB | FVD/KVD | [Official JavisBench URL](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.0.0/resolve/810fa8c4bdb3a4c8eec9bd57375c29bde6fb46de/opensora/eval/fvd/videogpt/i3d_pretrained_400.pt) |
| `AudioCLIP-Full-Training.pt` | ~350 MB | FAD | [Official AudioCLIP Release](https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt) |
| `cavp_epoch66.ckpt` | ~1.3 GB | CAVP Score | [Official Diff‑Foley Repository](https://huggingface.co/SimianLuo/Diff-Foley/resolve/main/diff_foley_ckpt/cavp_epoch66.ckpt) |
| `synchformer_state_dict.pth` | ~5.0 GB | DeSync | [Official Synchformer Mirror (MMAudio)](https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth) |
| `imagebind_huge.pth` | 4.5 GB | ImageBind series | [Official Meta ImageBind](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) |

> All links are the exact URLs used by the JavisBench source code or original model repositories.  
> `imagebind_huge.pth` is automatically downloaded on first evaluation run; the link above is provided for manual download if needed.


## 4. Evaluated Metrics

| Dimension | Metric | Direction | Description |
|-----------|--------|-----------|-------------|
| **Generation Quality** | FVD | ↓ | Fréchet Video Distance |
| | KVD | ↓ | Kernel Video Distance |
| | FAD | ↓ | Fréchet Audio Distance |
| | Audio Quality | ↑ | Audio aesthetic score (0‑10) |
| **Semantic Consistency** | CLIPScore | ↑ | Text-to-video alignment |
| | CLAPScore | ↑ | Text-to-audio alignment |
| | IB‑TV | ↑ | ImageBind text‑video similarity |
| | IB‑TA | ↑ | ImageBind text‑audio similarity |
| **AV Semantic Consistency** | CAVP Score | ↑ | Cross-modal audio‑video alignment |
| | AVHScore | ↑ | Frame‑level audio‑visual harmony |
| | IB‑AV | ↑ | ImageBind audio‑video similarity |
| **Temporal Synchrony** | JavisScore | ↑ | Sliding‑window synchrony |
| | AV‑Align | ↑ | Peak‑based temporal alignment |
| | DeSync | ↓ | Predicted audio‑video offset (seconds) |

## 5. Data Preparation

### 5.1 Single Evaluation (one set of videos → average scores)

Place your generated videos inside `data/videos/{model_name}/`.
Audio will be extracted automatically.

**Required structure:**
```
data/videos/my_model/
├── sample_0000.mp4
├── sample_0001.mp4
├── ...
├── prompts_shard_00.txt
├── prompts_shard_01.txt
└── ...
```

**Generate CSV & audio:**
```bash
python scripts/prepare_data.py --video_dir data/videos/my_model
```

### 5.2 Batch Evaluation (multiple seeds per prompt → mean ± std)

Organize your data by prompt and seed:

```
data/videos/my_model/
├── prompts.json
├── prompt_001/
│   ├── seed_00.mp4
│   ├── seed_01.mp4
│   └── seed_02.mp4
├── prompt_002/
│   └── ...
```

**`prompts.json` format:**
```json
[
    {"id": "prompt_001", "text": "A cat playing the piano."},
    {"id": "prompt_002", "text": "Waves crashing on a rocky shore."}
]
```

## 6. Running Evaluation

### 6.1 Single Evaluation (14 metrics, average over all videos)

```bash
python scripts/run_single_eval.py --model_dir data/videos/my_model --model_name my_model
```

- Results saved to `results/single_results/my_model_metrics.json`.
- The script automatically handles assertion patches and restores original files after completion.

### 6.2 Batch Evaluation (variance analysis)

```bash
python scripts/run_batch_eval.py --model my_model --n_seeds 3
```

- Results saved to `results/batch_results/my_model_summary.json`.
- Each per‑prompt entry contains `mean`, `std` and `n` for every metric.

## 7. Output Example

```json
{
    "fvd": 2447.71,
    "kvd": -2.08,
    "fad": 15.87,
    "audio_quality": 5.25,
    "clip_score": 0.2192,
    "clap_score": 0.4505,
    "cavp_score": 0.7063,
    "avh_score": 0.3422,
    "javis_score": 0.3245,
    "ib_tv": 0.2179,
    "ib_ta": 0.1543,
    "ib_av": 0.2041,
    "av_align": 0.1618,
    "desync": 0.4000
}
```

## 8. Directory Structure

```
Evaluation/
├── scripts/               # Main entry points
│   ├── prepare_data.py
│   ├── patch_metrics.py
│   ├── run_single_eval.py
│   ├── run_batch_eval.py
│   ├── collect_results.py
│   └── generate_table.py
├── javisbench/            # Official JavisBench source (from JavisDiT)
├── checkpoints/           # Pretrained model weights
├── data/videos/           # Place your generated videos here
├── results/               # All output scores
└── requirements_eval.txt
```

## 9. Known Issues & Troubleshooting

- **Network / HuggingFace unreachable**
  `export HF_ENDPOINT=https://hf-mirror.com` before running any script.

- **`torio` FFmpeg error**
  Install system FFmpeg: `apt-get install -y ffmpeg`
  Set `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`.

- **`mmcv` import errors**
  Make sure you installed `mmcv==1.7.1` (listed in `requirements_eval.txt`).

- **FVD / KVD unstable with small samples**
  KVD may become negative when fewer than ~50 videos are used.
  For reliable distribution distances, evaluate at least 100 videos per model.

- **CUDA Out of Memory (DeSync)**
  The patch script reduces DeSync batch size to 1. If you still see OOM, restart the GPU instance or free memory before running.
