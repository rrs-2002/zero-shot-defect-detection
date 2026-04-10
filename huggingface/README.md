---
title: WinCLIP Zero-Shot Defect Detection
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: true
license: mit
short_description: Zero-shot industrial anomaly detection utilizing WinCLIP.
---

# 🔍 WinCLIP — Zero-Shot Industrial Anomaly Detection

Upload a product image, select its category, and detect defects **without any training data**.

## How it works
- Uses OpenAI CLIP (ViT-B/32) with sliding window analysis
- 136 compositional text prompts (normal + defect states)
- Generates defect heatmap overlay with anomaly score
- Supports all 15 MVTec AD categories

## Author
Ritu Raj Singh | KIIT University | M.Tech CSE
