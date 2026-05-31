# Evaluation
This module provides an official evaluation pipeline for Joint Audio-Video Generation (JAVG) models, based on the JavisBench benchmark from *JavisDiT (ICLR 2026)*. It computes 14 objective metrics across four dimensions: generation quality, semantic consistency, audio-visual semantic consistency, and temporal synchrony.
## 1. Introduction

This evaluation module is designed to standardize the assessment of JAVG models. It relies on pretrained feature extractors and a reference set of 10,140 real videos from JavisBench. The pipeline is fully automated and requires only a set of generated videos and their corresponding text prompts.

## 2. Installation

```bash
conda create -n javis_eval python=3.10 -y
conda activate javis_eval
pip install -r requirements_eval.txt
apt-get install -y ffmpeg
