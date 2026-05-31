# Metrics Description

This document provides a detailed explanation of the 14 metrics computed by JavisBench, organized by evaluation dimension.

---

## 1. Generation Quality

### FVD (Fréchet Video Distance)

![FVD](https://latex.codecogs.com/svg.latex?\text{FVD}%20=%20\|\mu_r%20-%20\mu_g\|^2%20+%20\text{Tr}\left(\Sigma_r%20+%20\Sigma_g%20-%202(\Sigma_r\Sigma_g)^{1/2}\right))

**Description**:  
Measures the distribution distance between generated and real videos in the I3D feature space. Lower values indicate that the generated videos are closer to real videos in terms of visual quality and motion patterns.

**Direction**: ↓ Lower is better

---

### KVD (Kernel Video Distance)

![KVD](https://latex.codecogs.com/svg.latex?\text{KVD}%20=%20\frac{1}{n(n-1)}\sum_{i\neq%20j}k(x_i,x_j)%20+%20\frac{1}{m(m-1)}\sum_{i\neq%20j}k(y_i,y_j)%20-%20\frac{2}{nm}\sum_{i,j}k(x_i,y_j))

where `k(x,y) = (x^T y + 1)^3` is the polynomial kernel.

**Description**:  
Similar to FVD, but uses a polynomial kernel Maximum Mean Discrepancy (MMD) instead of assuming Gaussian distributions. Can be more robust for non-Gaussian feature distributions, but may become unstable with very few samples (e.g., < 50 videos).

**Direction**: ↓ Lower is better

> **Note**: KVD may become negative when the number of generated videos is very small (e.g., 10). This is a known issue with unbiased MMD estimators and indicates insufficient sample size for reliable distribution estimation.

---

### FAD (Fréchet Audio Distance)

![FAD](https://latex.codecogs.com/svg.latex?\text{FAD}%20=%20\|\mu_r%20-%20\mu_g\|^2%20+%20\text{Tr}\left(\Sigma_r%20+%20\Sigma_g%20-%202(\Sigma_r\Sigma_g)^{1/2}\right))

**Description**:  
Same formula as FVD, but applied to audio features extracted by AudioCLIP. Measures the distribution distance between generated and real audio.

**Direction**: ↓ Lower is better

---

### Audio Quality

![Audio Quality](https://latex.codecogs.com/svg.latex?\text{AQ}%20=%20\frac{1}{4}\sum_{d%20\in%20\{\text{CE},\text{CU},\text{PC},\text{PQ}\}}%20f_d(a))

**Description**:  
Aesthetic score computed by Meta's AudioBox Aesthetics model. Evaluates audio across four dimensions:  
- **CE** (Content Enjoyment)  
- **CU** (Content Usefulness)  
- **PC** (Production Complexity)  
- **PQ** (Production Quality)  

Score ranges from 0 to 10.

**Direction**: ↑ Higher is better

---

## 2. Semantic Consistency (Text Alignment)

### CLIPScore

![CLIPScore](https://latex.codecogs.com/svg.latex?\text{CLIPScore}%20=%20\frac{1}{N}\sum_{i=1}^{N}\frac{E_v(v_i^{\text{mid}})%20\cdot%20E_t(t_i)}{\|E_v(v_i^{\text{mid}})\|%20\cdot%20\|E_t(t_i)\|})

**Description**:  
Measures the cosine similarity between the middle frame of each generated video and its corresponding text prompt, using OpenAI's CLIP ViT-B/32 model.

**Direction**: ↑ Higher is better

---

### CLAPScore

![CLAPScore](https://latex.codecogs.com/svg.latex?\text{CLAPScore}%20=%20\frac{1}{N}\sum_{i=1}^{N}\frac{E_a(a_i)%20\cdot%20E_t(t_i)}{\|E_a(a_i)\|%20\cdot%20\|E_t(t_i)\|})

**Description**:  
Measures the cosine similarity between generated audio and text prompt, using LAION's CLAP model.

**Direction**: ↑ Higher is better

---

### IB‑TV (ImageBind Text‑Video)

![IB-TV](https://latex.codecogs.com/svg.latex?\text{IB-TV}%20=%20\frac{1}{N}\sum_{i=1}^{N}\frac{f_t(t_i)%20\cdot%20f_v(v_i)}{\|f_t(t_i)\|%20\cdot%20\|f_v(v_i)\|})

**Description**:  
Cosine similarity between text and video embeddings in ImageBind's shared multimodal space.

**Direction**: ↑ Higher is better

---

### IB‑TA (ImageBind Text‑Audio)

![IB-TA](https://latex.codecogs.com/svg.latex?\text{IB-TA}%20=%20\frac{1}{N}\sum_{i=1}^{N}\frac{f_t(t_i)%20\cdot%20f_a(a_i)}{\|f_t(t_i)\|%20\cdot%20\|f_a(a_i)\|})

**Description**:  
Cosine similarity between text and audio embeddings in ImageBind space.

**Direction**: ↑ Higher is better

---

## 3. Audio‑Visual Semantic Consistency

### CAVP Score

![CAVP Score](https://latex.codecogs.com/svg.latex?\text{CAVP}%20=%20\frac{1}{N}\sum_{i=1}^{N}\frac{f_{\text{cavp}}(v_i)%20\cdot%20g_{\text{cavp}}(a_i)}{\|f_{\text{cavp}}(v_i)\|%20\cdot%20\|g_{\text{cavp}}(a_i)\|})

**Description**:  
Cross‑modal similarity using Diff‑Foley's pretrained audio‑visual model. Measures semantic alignment between video and audio content.

**Direction**: ↑ Higher is better

---

### AVHScore (Audio‑Visual Harmony Score)

![AVHScore](https://latex.codecogs.com/svg.latex?\text{AVHScore}%20=%20\frac{1}{N}\sum_{i=1}^{N}\frac{1}{T_i}\sum_{t=1}^{T_i}\frac{f_{\text{frame}}(v_{i,t})%20\cdot%20f_{\text{audio}}(a_i)}{\|f_{\text{frame}}(v_{i,t})\|%20\cdot%20\|f_{\text{audio}}(a_i)\|})

**Description**:  
Average frame‑level cosine similarity between each video frame and the entire audio clip. Higher values indicate that the audio "fits" all frames consistently.

**Direction**: ↑ Higher is better

---

### IB‑AV (ImageBind Audio‑Video)

![IB-AV](https://latex.codecogs.com/svg.latex?\text{IB-AV}%20=%20\frac{1}{N}\sum_{i=1}^{N}\frac{f_a(a_i)%20\cdot%20f_v(v_i)}{\|f_a(a_i)\|%20\cdot%20\|f_v(v_i)\|})

**Description**:  
Cosine similarity between audio and video embeddings in ImageBind space.

**Direction**: ↑ Higher is better

---

## 4. Temporal Synchrony

### JavisScore

![JavisScore](https://latex.codecogs.com/svg.latex?\text{JavisScore}%20=%20\frac{1}{W}\sum_{i=1}^{W}\left(\frac{1}{k_i}\sum_{j=1}^{k_i}%20\min\_k\left\{%20\frac{E_v(v_{i,j})%20\cdot%20E_a(a_i)}{\|E_v(v_{i,j})\|%20\cdot%20\|E_a(a_i)\|}%20\right\}\right))

**Description**:  
JavisBench's own synchrony metric. Slides a window across the video‑audio pair, and for each window, averages the lowest similarity frames. This penalizes short‑term asynchrony. Higher values indicate better temporal alignment.

**Direction**: ↑ Higher is better

---

### AV‑Align

![AV-Align](https://latex.codecogs.com/svg.latex?\text{AV-Align}%20=%20\frac{1}{|\mathcal{P}_a|%20+%20|\mathcal{P}_v|}\left(%20\sum_{t_a%20\in%20\mathcal{P}_a}%20\mathbb{1}_{\{\exists%20t_v%20\in%20\mathcal{P}_v%20:%20|t_a%20-%20t_v|%20\leq%20\tau\}}%20+%20\sum_{t_v%20\in%20\mathcal{P}_v}%20\mathbb{1}_{\{\exists%20t_a%20\in%20\mathcal{P}_a%20:%20|t_a%20-%20t_v|%20\leq%20\tau\}}%20\right))

**Description**:  
Measures the bidirectional matching rate between audio energy peaks (P_a) and video motion peaks (P_v) within a tolerance window τ (3 frames). Ranges from 0 to 1.

**Direction**: ↑ Higher is better

---

### DeSync

![DeSync](https://latex.codecogs.com/svg.latex?\text{DeSync}%20=%20\frac{1}{N}\sum_{i=1}^{N}|\Delta_i|)

**Description**:  
Predicted absolute audio‑video offset (in seconds) by the Synchformer model. Lower values indicate better synchrony; 0 means perfect alignment.

**Direction**: ↓ Lower is better

---

## Reference

All metrics are implemented according to the JavisBench paper:  
*Liu, K., et al. "JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization." ICLR 2026.*
