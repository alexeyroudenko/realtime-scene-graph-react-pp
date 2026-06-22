# Prompt for Cursor: Pure OpenCV Real-time Scene Graph

You are an expert Python developer in real-time computer vision.

Create a complete, clean, high-performance Python project called `realtime-scene-graph-opencv`.

## Goal

Build a fast, lightweight desktop application that runs **real-time Scene Graph Generation** from a webcam using **pure OpenCV** (no NiceGUI, no PyQt, no web frameworks). The focus is on maximum performance and simplicity.

## Tech Stack (strict)

- Python 3.11+
- `opencv-python`
- `onnxruntime-gpu` (with automatic fallback to CPU)
- `huggingface_hub` (for automatic model download)
- `numpy`
- `networkx` (for graph representation)
- Type hints + clean modular code

## Project Structure

See repository layout in README.md.

## Core Requirements

1. **Automatic model download** — REACT++ ONNX from Hugging Face (`maelic/REACTPlusPlus_PSG`)
2. **Inference** — Based on SGG-Benchmark; live `box_conf` / `rel_conf` adjustment
3. **Pure OpenCV Interface** — Webcam, boxes, relations, trackbars, keyboard controls
4. **Scene Graph Handling** — `networkx.DiGraph` + JSON export
