"""Scene graph inference and graph representation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from src.config import DEFAULT_BOX_CONF, DEFAULT_MAX_RELATIONS, DEFAULT_REL_CONF, INPUT_SIZE
from src.inference.model_loader import create_session, ensure_model, load_class_names, preprocess_image

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    index: int
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float


@dataclass
class DetectedRelation:
    subject_index: int
    object_index: int
    predicate_id: int
    predicate: str
    triplet_score: float
    relation_score: float


@dataclass
class SceneGraphResult:
    objects: list[DetectedObject] = field(default_factory=list)
    relations: list[DetectedRelation] = field(default_factory=list)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    inference_ms: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "objects": [
                {
                    "index": obj.index,
                    "class_id": obj.class_id,
                    "class_name": obj.class_name,
                    "confidence": round(obj.confidence, 4),
                    "bbox": {
                        "x1": round(obj.bbox[0], 2),
                        "y1": round(obj.bbox[1], 2),
                        "x2": round(obj.bbox[2], 2),
                        "y2": round(obj.bbox[3], 2),
                    },
                }
                for obj in self.objects
            ],
            "relations": [
                {
                    "subject_index": rel.subject_index,
                    "object_index": rel.object_index,
                    "predicate_id": rel.predicate_id,
                    "predicate": rel.predicate,
                    "triplet_score": round(rel.triplet_score, 4),
                    "relation_score": round(rel.relation_score, 4),
                    "triplet": (
                        f"{self.objects[rel.subject_index].class_name}"
                        f" - {rel.predicate} - "
                        f"{self.objects[rel.object_index].class_name}"
                    ),
                }
                for rel in self.relations
                if rel.subject_index < len(self.objects) and rel.object_index < len(self.objects)
            ],
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")


class SceneGraphEngine:
    """REACT++ ONNX inference with networkx scene graph construction."""

    def __init__(
        self,
        model_path: Path | None = None,
        box_conf: float = DEFAULT_BOX_CONF,
        rel_conf: float = DEFAULT_REL_CONF,
        max_relations: int = DEFAULT_MAX_RELATIONS,
    ) -> None:
        self.box_conf = box_conf
        self.rel_conf = rel_conf
        self.max_relations = max_relations

        resolved = model_path or ensure_model()
        self.session = create_session(resolved)
        self.input_name = self.session.get_inputs()[0].name
        self.obj_classes, self.rel_classes = load_class_names(self.session)

    def set_thresholds(
        self,
        box_conf: float | None = None,
        rel_conf: float | None = None,
        max_relations: int | None = None,
    ) -> None:
        if box_conf is not None:
            self.box_conf = float(np.clip(box_conf, 0.0, 1.0))
        if rel_conf is not None:
            self.rel_conf = float(np.clip(rel_conf, 0.0, 1.0))
        if max_relations is not None:
            self.max_relations = max(1, int(max_relations))

    def _run_inference(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Run ONNX inference; retry with tiny noise on CPU broadcast edge cases."""
        try:
            return self.session.run(None, {self.input_name: tensor})
        except Exception as first_error:
            noise = np.random.default_rng().normal(0.0, 1e-3, tensor.shape).astype(np.float32)
            perturbed = np.clip(tensor + noise, 0.0, 1.0)
            try:
                return self.session.run(None, {self.input_name: perturbed})
            except Exception:
                logger.warning("Inference failed: %s", first_error)
                raise first_error

    def predict(self, frame: np.ndarray) -> SceneGraphResult:
        import time

        start = time.perf_counter()
        tensor, ratio, (pad_x, pad_y) = preprocess_image(frame, INPUT_SIZE)
        try:
            outputs = self._run_inference(tensor)
        except Exception:
            return SceneGraphResult(inference_ms=(time.perf_counter() - start) * 1000.0)
        inference_ms = (time.perf_counter() - start) * 1000.0

        boxes_raw = outputs[0].copy()
        rels_raw = outputs[1].copy() if len(outputs) > 1 else np.empty((0, 5))

        boxes_raw[:, [0, 2]] = (boxes_raw[:, [0, 2]] - pad_x) / ratio
        boxes_raw[:, [1, 3]] = (boxes_raw[:, [1, 3]] - pad_y) / ratio

        height, width = frame.shape[:2]
        boxes_raw[:, [0, 2]] = np.clip(boxes_raw[:, [0, 2]], 0, width)
        boxes_raw[:, [1, 3]] = np.clip(boxes_raw[:, [1, 3]], 0, height)

        keep_rels = rels_raw[rels_raw[:, 3] >= self.rel_conf] if len(rels_raw) else rels_raw
        if len(keep_rels):
            order = np.argsort(-keep_rels[:, 3])
            keep_rels = keep_rels[order][: self.max_relations]

        rel_box_indices: set[int] = set()
        if len(keep_rels):
            rel_box_indices = set(keep_rels[:, 0].astype(int)) | set(keep_rels[:, 1].astype(int))

        keep_mask = boxes_raw[:, 5] >= self.box_conf
        for idx in rel_box_indices:
            if 0 <= idx < len(keep_mask):
                keep_mask[idx] = True

        keep_indices = np.where(keep_mask)[0]
        final_boxes = boxes_raw[keep_indices]

        old_to_new = {int(old): new for new, old in enumerate(keep_indices)}
        final_rels: list[np.ndarray] = []
        for rel in keep_rels:
            subj, obj = int(rel[0]), int(rel[1])
            if subj in old_to_new and obj in old_to_new:
                remapped = rel.copy()
                remapped[0] = old_to_new[subj]
                remapped[1] = old_to_new[obj]
                final_rels.append(remapped)

        return self._build_result(final_boxes, np.array(final_rels) if final_rels else np.empty((0, 5)), inference_ms)

    def _build_result(
        self,
        boxes: np.ndarray,
        rels: np.ndarray,
        inference_ms: float,
    ) -> SceneGraphResult:
        objects: list[DetectedObject] = []
        graph = nx.DiGraph()

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2, class_id, score = box
            class_id_int = int(class_id)
            class_name = self.obj_classes.get(class_id_int, f"class_{class_id_int}")
            detected = DetectedObject(
                index=idx,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                class_id=class_id_int,
                class_name=class_name,
                confidence=float(score),
            )
            objects.append(detected)
            graph.add_node(
                idx,
                class_id=class_id_int,
                class_name=class_name,
                confidence=float(score),
                bbox=detected.bbox,
            )

        relations: list[DetectedRelation] = []
        if rels.ndim == 2:
            for rel in rels:
                subj_idx, obj_idx, pred_id = int(rel[0]), int(rel[1]), int(rel[2])
                if subj_idx >= len(objects) or obj_idx >= len(objects):
                    continue
                predicate = self.rel_classes.get(pred_id, f"rel_{pred_id}")
                relation = DetectedRelation(
                    subject_index=subj_idx,
                    object_index=obj_idx,
                    predicate_id=pred_id,
                    predicate=predicate,
                    triplet_score=float(rel[3]),
                    relation_score=float(rel[4]) if rel.shape[0] > 4 else float(rel[3]),
                )
                relations.append(relation)
                graph.add_edge(
                    subj_idx,
                    obj_idx,
                    predicate=predicate,
                    predicate_id=pred_id,
                    triplet_score=relation.triplet_score,
                    relation_score=relation.relation_score,
                )

        return SceneGraphResult(
            objects=objects,
            relations=relations,
            graph=graph,
            inference_ms=inference_ms,
        )
