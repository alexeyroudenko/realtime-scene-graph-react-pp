# Real-time Scene Graph — Pure OpenCV

Lightweight desktop app for **real-time Scene Graph Generation** from a webcam using **pure OpenCV** and the REACT++ ONNX model. No GUI frameworks, no web stack — just fast inference and live visualization.

## Features

- Automatic download of **REACT++ YOLO12m** ONNX weights from [maelic/REACTPlusPlus_PSG](https://huggingface.co/maelic/REACTPlusPlus_PSG)
- GPU inference via `onnxruntime-gpu` with automatic CPU fallback
- Live bounding boxes, relation arrows, and predicate labels
- OpenCV trackbars for **Box Confidence**, **Relation Confidence**, and **Max Relations**
- Secondary info panel with object/relation lists
- `networkx` directed graph internally; export current frame as JSON

## Requirements

- Python 3.11+
- Webcam
- Optional: NVIDIA GPU + CUDA for best performance

## Install

```bash
cd realtime-scene-graph-opencv
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Optional arguments:

```bash
python main.py --camera 0
python main.py --model models/react_pp_yolo12m.onnx
python main.py --no-info-panel
```

On first launch, the REACT++ model (~50 MB) is downloaded into `models/`.

## Controls

| Input | Action |
|-------|--------|
| **Box Confidence** trackbar | Filter low-confidence detections |
| **Relation Confidence** trackbar | Filter weak relation triplets |
| **Max Relations** trackbar | Limit displayed relations (top by score) |
| `q` / `ESC` | Quit |
| `s` | Save current scene graph to `output/scene_graph_*.json` |
| `p` | Pause / resume |
| `r` | Reset trackbars to defaults |
| `c` | Switch camera (when multiple devices exist) |

## JSON Export Format

```json
{
  "timestamp": "2026-06-19T12:00:00+00:00",
  "objects": [
    {
      "index": 0,
      "class_id": 1,
      "class_name": "person",
      "confidence": 0.92,
      "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200}
    }
  ],
  "relations": [
    {
      "subject_index": 0,
      "object_index": 1,
      "predicate_id": 3,
      "predicate": "on",
      "triplet_score": 0.15,
      "relation_score": 0.22,
      "triplet": "person - on - chair"
    }
  ]
}
```

## Project Layout

```
realtime-scene-graph-opencv/
├── src/
│   ├── config.py
│   ├── inference/
│   │   ├── model_loader.py    # HF download + ONNX session
│   │   └── scene_graph.py     # Inference + networkx graph
│   ├── visualization/
│   │   └── drawer.py          # OpenCV rendering
│   └── utils/
│       └── helpers.py
├── models/                    # Cached ONNX weights
├── output/                    # Saved JSON exports
├── main.py
├── requirements.txt
└── pyproject.toml
```

## Inference Notes

Inference logic is adapted from the [SGG-Benchmark](https://github.com/Maelic/SGG-Benchmark) standalone ONNX demo:

- Letterbox preprocessing at 640×640
- ONNX outputs: boxes `(N, 6)` and relations `(M, 5)`
- Relation filtering uses triplet score; boxes involved in kept relations are always shown

Default thresholds: `box_conf=0.40`, `rel_conf=0.05`.

## License

MIT. REACT++ model weights are subject to the Hugging Face model card terms.
