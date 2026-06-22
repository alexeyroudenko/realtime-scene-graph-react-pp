# Algorithm Overview

This document describes how **Scene Graph Generation (SGG)** works in this application: from a camera frame to bounding boxes, relation triplets, and a directed `networkx` graph.

> **Russian version:** [DESCRIPTION.ru.md](DESCRIPTION.ru.md)

## Key idea

This project does **not** use a classic two-stage pipeline (detect objects with one model, then run a separate relation model on every pair). The **REACT++** model ([maelic/REACTPlusPlus_PSG](https://huggingface.co/maelic/REACTPlusPlus_PSG)) performs **one end-to-end forward pass** and returns both object detections and their relations at once.

The order “filter relations first, then boxes” in the post-processing code is a **display/filtering policy**, not the order in which the neural network works.

## Pipeline

```
Camera frame (BGR)
    → letterbox preprocess (640×640)
    → REACT++ ONNX inference (single pass)
    → boxes (N×6) + relations (M×5)
    → thresholding & index remapping
    → SceneGraphResult (objects, relations, networkx DiGraph)
    → OpenCV visualization
```

Implementation: `src/inference/scene_graph.py`, `src/inference/model_loader.py`.

## 1. Preprocessing

Each frame is resized with **letterboxing** to `640×640` (`INPUT_SIZE` in `src/config.py`):

- Aspect ratio is preserved; gray padding `(114, 114, 114)` fills the rest.
- BGR → RGB, layout `CHW`, values normalized to `[0, 1]`.
- Output tensor shape: `(1, 3, 640, 640)`.

After inference, box coordinates are mapped back to the original frame: padding is removed and coordinates are divided by the resize `ratio`, then clipped to image bounds.

See `preprocess_image()` in `src/inference/model_loader.py`.

## 2. Single forward pass — two outputs

`onnxruntime` runs the REACT++ ONNX model once per frame. The session returns two arrays:

| Output | Shape | Columns |
|--------|-------|---------|
| Boxes | `(N, 6)` | `x1, y1, x2, y2, class_id, confidence` |
| Relations | `(M, 5)` | `subject_idx, object_idx, predicate_id, triplet_score, relation_score` |

- **Boxes** — detected objects with PSG dataset class labels (e.g. person, chair, cup).
- **Relations** — **triplets** *(subject, predicate, object)* where `subject_idx` and `object_idx` refer to **row indices** in the box array from the same forward pass.

Object and predicate names are loaded from ONNX metadata (`obj_classes`, `rel_classes`) via `load_class_names()`.

At the architecture level, REACT++ (YOLO12m backbone + scene-graph head) is trained **jointly**: detection and relation prediction are not separate Python stages. The model directly predicts which detection pairs are linked and with which predicate (`on`, `wearing`, `holding`, etc.).

## 3. Post-processing

Post-processing lives in `SceneGraphEngine.predict()`. Threshold defaults are in `src/config.py`:

- `box_conf` = 0.40
- `rel_conf` = 0.05
- `max_relations` = 20

### Step A — filter relations

1. Keep relations where `triplet_score >= rel_conf`.
2. Sort by `triplet_score` descending.
3. Keep at most `max_relations` triplets.

`triplet_score` is the confidence for the full triplet (subject + predicate + object).  
`relation_score` is stored in the result and JSON export but **is not used** for filtering.

### Step B — filter boxes

1. Collect box indices that appear in any kept relation (subject or object).
2. Keep boxes with `confidence >= box_conf`.
3. **Exception:** a box is **always kept** if it participates in a kept relation, even when its confidence is below `box_conf`. This avoids showing a relation arrow pointing to a hidden object.

### Step C — remap indices

After boxes are filtered, relation indices are remapped (`old_to_new`) so `subject_index` / `object_index` still point to valid objects in the final list.

## 4. Scene graph construction

`_build_result()` assembles the output structures:

- **Nodes** — one per kept detection (`DetectedObject` + node in `networkx.DiGraph`).
- **Edges** — directed relations from subject to object, labeled with the predicate.

Example: `person - on - chair` → node `person` → edge `on` → node `chair`.

The graph is a **directed** graph (`DiGraph`): `person on chair` is not the same as `chair on person`.

Relations and objects can be exported as JSON (`SceneGraphResult.to_json_dict()` / `save_json()`).

## 5. Worked example

Suppose the model returns 8 boxes and 50 relation candidates:

```
Boxes:  [0: person 0.9] [1: chair 0.7] [2: cup 0.35] [3: table 0.2] ...
Relations:
  (0, 1, "on",      triplet_score=0.18)  — person on chair
  (0, 2, "holding", triplet_score=0.12)  — person holding cup
  (2, 3, "on",      triplet_score=0.08)  — cup on table
```

With default thresholds:

1. All three relations pass `rel_conf` (scores ≥ 0.05).
2. Boxes kept: person and chair by confidence; cup (0.35) and table (0.20) are below `box_conf` but **forced in** because they appear in kept relations.
3. Other boxes with low confidence and no relations are dropped.

The UI draws bounding boxes, arrows between box centers, predicate labels, an info panel, and a mini graph window.

## 6. Comparison with classic two-stage SGG

| | Classic approach | REACT++ (this project) |
|--|------------------|------------------------|
| Detection | Separate detector | Built into the same model |
| Relations | Often O(N²) pairs + relation network | Predicted with box indices in one pass |
| Inference passes | 1 + many per pair | **1** ONNX run per frame |
| Real-time webcam | Harder with many objects | Suited for live use |

## 7. User-adjustable parameters

| Control | Effect |
|---------|--------|
| **Box Confidence** | Fewer spurious objects without relations |
| **Relation Confidence** | Fewer weak or false triplets |
| **Max Relations** | Cap on the number of top-scoring relations per frame |

Trackbars update thresholds live via `SceneGraphEngine.set_thresholds()`.

## 8. What this project does not do

- **No segmentation** — only axis-aligned bounding boxes (no SAM/SAM2 masks).
- **No per-pair relation model in Python** — relations come from the single REACT++ export.
- **No training** — inference only, using the pre-exported ONNX weights.

For inference details aligned with upstream export, see [SGG-Benchmark](https://github.com/Maelic/SGG-Benchmark) and the model card on Hugging Face.
