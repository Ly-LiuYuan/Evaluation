

# Troubleshooting

This document lists common issues encountered during evaluation and their solutions.

## 1. Network / HuggingFace unreachable

**Symptom**: `HTTPError 503` or `Max retries exceeded` when downloading model weights.

**Solution**:
- Set the HuggingFace mirror before running any script:
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- Alternatively, download all required weights manually and place them in `checkpoints/`. See the main README for download links.

## 2. `torio` FFmpeg error

**Symptom**: `ImportError: Failed to initialize FFmpeg extension` or `StreamingMediaDecoder` errors.

**Solution**:
- Install system FFmpeg:
  ```bash
  apt-get install -y ffmpeg
  ```
- Set the library path to use system FFmpeg:
  ```bash
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  ```
- If the error persists, temporarily rename conda FFmpeg libraries:
  ```bash
  mv $CONDA_PREFIX/lib/libavcodec.so.62 $CONDA_PREFIX/lib/libavcodec.so.62.bak 2>/dev/null
  mv $CONDA_PREFIX/lib/libavfilter.so.9 $CONDA_PREFIX/lib/libavfilter.so.9.bak 2>/dev/null
  mv $CONDA_PREFIX/lib/libavformat.so.60 $CONDA_PREFIX/lib/libavformat.so.60.bak 2>/dev/null
  ```
  Remember to restore them after evaluation:
  ```bash
  mv $CONDA_PREFIX/lib/libavcodec.so.62.bak $CONDA_PREFIX/lib/libavcodec.so.62 2>/dev/null
  mv $CONDA_PREFIX/lib/libavfilter.so.9.bak $CONDA_PREFIX/lib/libavfilter.so.9 2>/dev/null
  mv $CONDA_PREFIX/lib/libavformat.so.60.bak $CONDA_PREFIX/lib/libavformat.so.60 2>/dev/null
  ```

## 3. `mmcv` import errors

**Symptom**: `ImportError: cannot import name 'kaiming_init' from 'mmcv.cnn'` or `RuntimeError: dictionary changed size during iteration`.

**Solution**:
- Ensure `mmcv==1.7.1` is installed (listed in `requirements_eval.txt`).
- If you accidentally installed a newer version, downgrade:
  ```bash
  pip install mmcv==1.7.1
  ```

## 4. FVD / KVD unstable with small samples

**Symptom**: KVD is negative, or FVD values fluctuate wildly.

**Reason**: Both FVD and KVD require a reasonable number of samples to estimate feature distributions reliably.

**Solution**:
- Use at least 100 generated videos per model for stable FVD/KVD values.
- For small sample sizes (e.g., 10 videos), treat these metrics as rough relative comparisons only; do not compare them directly with paper results.

## 5. CUDA Out of Memory (DeSync)

**Symptom**: `torch.OutOfMemoryError: CUDA out of memory` during DeSync evaluation.

**Solution**:
- The evaluation scripts automatically reduce DeSync batch size to 1. If you still see OOM, restart the GPU instance to clear memory.
- You can also set the following environment variable before running:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```
- If the problem persists, run DeSync separately for each video rather than all at once.

## 6. AssertionError during evaluation

**Symptom**: `AssertionError` in `calc_clip_score`, `calc_clap_score`, `calc_cavp_score`, or `calc_av_score`.

**Reason**: The official code requires batch size = 1, but DataLoader may return a different batch size in certain configurations.

**Solution**:
- The `patch_metrics.py` script handles this automatically when you use `run_single_eval.py` or `run_batch_eval.py`.
- If you are running the evaluation manually, apply the patch before running:
  ```bash
  python scripts/patch_metrics.py --apply
  ```
  After evaluation, restore the original code:
  ```bash
  python scripts/patch_metrics.py --restore
  ```

## 7. `EncodedVideo.from_path()` unexpected keyword argument `sample_rate`

**Symptom**: `TypeError: EncodedVideo.from_path() got an unexpected keyword argument 'sample_rate'`.

**Reason**: The bundled ImageBind code passes an obsolete `sample_rate` argument to a newer torchvision API.

**Solution**:
- Edit `javisbench/src/ImageBind/imagebind/data.py`, locate the line containing `**{"sample_rate": sample_rate}` (around line 321), and delete that argument.
- After modification, the line should look like:
  ```python
  video = EncodedVideo.from_path(
      video_path,
      decoder="decord",
      decode_audio=False,
  )
  ```

## 8. `eval_metadata.csv` not found

**Symptom**: Script reports `eval_metadata.csv not found`.

**Solution**:
- Run `python scripts/prepare_data.py --video_dir <your_video_dir>` to generate the CSV and extract audio.
- Make sure the directory contains both video files and `prompts_shard_*.txt` files.

## 9. No module named `javisbench`

**Symptom**: `ModuleNotFoundError: No module named 'javisbench'`.

**Solution**:
- Ensure you have copied the `javisbench/` directory from the official JavisDiT repository into your `Evaluation/` folder.
- If you cloned this repository, follow the instructions in README Section 2 (Installation) to download the evaluation source code.

## 10. Pre-trained weights not found

**Symptom**: `FileNotFoundError: ./checkpoints/xxx.pt`.

**Solution**:
- Download the required weights and place them in the `checkpoints/` directory. See the main README for official download links.
- If the evaluation script tries to download `imagebind_huge.pth` automatically but fails, set a HuggingFace mirror:
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- Alternatively, download `imagebind_huge.pth` manually from the provided link and put it in `checkpoints/`.
