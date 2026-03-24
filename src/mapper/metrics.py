"""
Evaluation metrics for MDS extraction pipeline.

Compares predicted :class:`~mds_schema.MDSAssessment` objects against ground-truth
labels (from ``scripts/generate_labeled_samples.py``) and computes per-item,
per-section, and overall precision / recall / F1 scores.

Focuses on Section I (boolean diagnosis items), Section N (medication class items),
and Section O (treatment/procedure items) — the three sections targeted by the
project.

Usage
-----
    from src.evaluation.metrics import MDSEvaluator

    evaluator = MDSEvaluator(section_ids=["I", "N", "O"])

    # ground_truth: dict mapping hadm_id → {item_id: bool}
    # predictions:  dict mapping hadm_id → MDSAssessment

    report = evaluator.evaluate(ground_truth, predictions)
    print(report.summary())
    report.to_csv("results/eval_report.csv")
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.mds_schema import MDSAssessment, MDSItemType, MDSSchema

logger = logging.getLogger(__name__)


@dataclass
class ItemMetrics:
    """Precision / recall / F1 for a single MDS item."""
    item_id: str
    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def support(self) -> int:
        """Number of ground-truth positive cases."""
        return self.tp + self.fn


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across all items."""
    item_metrics: Dict[str, ItemMetrics] = field(default_factory=dict)
    num_samples: int = 0

    def section_metrics(self, section_id: str) -> Dict[str, float]:
        """Compute micro-averaged metrics for a section."""
        items = [m for m in self.item_metrics.values()
                 if m.item_id.startswith(section_id)]
        tp = sum(m.tp for m in items)
        fp = sum(m.fp for m in items)
        fn = sum(m.fn for m in items)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "section": section_id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "num_items": len(items),
        }

    def overall_metrics(self) -> Dict[str, float]:
        """Compute micro-averaged metrics across all items."""
        tp = sum(m.tp for m in self.item_metrics.values())
        fp = sum(m.fp for m in self.item_metrics.values())
        fn = sum(m.fn for m in self.item_metrics.values())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "num_samples": self.num_samples,
        }

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [f"=== MDS Extraction Evaluation ({self.num_samples} samples) ===\n"]

        overall = self.overall_metrics()
        lines.append(
            f"Overall:  P={overall['precision']:.3f}  "
            f"R={overall['recall']:.3f}  "
            f"F1={overall['f1']:.3f}  "
            f"(TP={overall['tp']} FP={overall['fp']} FN={overall['fn']})\n"
        )

        # Per-section
        sections = sorted({m.item_id[0] for m in self.item_metrics.values()})
        for sec in sections:
            sm = self.section_metrics(sec)
            lines.append(
                f"Section {sec}: P={sm['precision']:.3f}  "
                f"R={sm['recall']:.3f}  "
                f"F1={sm['f1']:.3f}  "
                f"({sm['num_items']} items)"
            )

        # Top items by F1 (those with support > 0)
        active = sorted(
            [m for m in self.item_metrics.values() if m.support > 0],
            key=lambda m: m.f1,
            reverse=True,
        )
        if active:
            lines.append(f"\nTop 15 items by F1 (with support > 0):")
            for m in active[:15]:
                lines.append(
                    f"  {m.item_id:12s} P={m.precision:.2f} R={m.recall:.2f} "
                    f"F1={m.f1:.2f} (support={m.support})"
                )

        # Worst items
        worst = sorted(
            [m for m in self.item_metrics.values() if m.support > 0],
            key=lambda m: m.f1,
        )
        if worst:
            lines.append(f"\nBottom 10 items by F1 (with support > 0):")
            for m in worst[:10]:
                lines.append(
                    f"  {m.item_id:12s} P={m.precision:.2f} R={m.recall:.2f} "
                    f"F1={m.f1:.2f} (support={m.support})"
                )

        return "\n".join(lines)

    def to_csv(self, path: Optional[str] = None) -> str:
        """Export per-item metrics as CSV. Optionally write to file."""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "item_id", "label", "precision", "recall", "f1",
            "tp", "fp", "fn", "tn", "support",
        ])
        writer.writeheader()
        for m in sorted(self.item_metrics.values(), key=lambda x: x.item_id):
            writer.writerow({
                "item_id": m.item_id,
                "label": m.label,
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "f1": round(m.f1, 4),
                "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
                "support": m.support,
            })
        csv_str = output.getvalue()
        if path:
            with open(path, "w") as f:
                f.write(csv_str)
            logger.info("Evaluation report saved to %s", path)
        return csv_str


class MDSEvaluator:
    """
    Evaluates predicted MDS assessments against ground-truth labels.

    Ground truth is a dict mapping ``hadm_id`` → ``{item_id: bool}``,
    as produced by ``scripts/generate_labeled_samples.py``.

    Parameters
    ----------
    section_ids:
        MDS sections to evaluate. Default: ``["I", "N", "O"]``.
    """

    def __init__(
        self,
        section_ids: Optional[List[str]] = None,
    ) -> None:
        self.schema = MDSSchema(section_ids=section_ids or ["I", "N", "O"])
        self._items = self.schema.all_items()
        # Only evaluate boolean items (diagnosis present/absent, drug class taken/not)
        self._eval_items = [
            item for item in self._items
            if item.item_type in (MDSItemType.BOOLEAN, MDSItemType.MULTI)
        ]

    def evaluate(
        self,
        ground_truth: Dict[str, Dict[str, Any]],
        predictions: Dict[str, MDSAssessment],
    ) -> EvaluationReport:
        """
        Compare predictions against ground truth.

        Parameters
        ----------
        ground_truth:
            ``{hadm_id: {item_id: True/False}}`` from labeled samples.
        predictions:
            ``{hadm_id: MDSAssessment}`` from the extraction pipeline.

        Returns
        -------
        EvaluationReport
        """
        report = EvaluationReport()

        # Initialize item metrics
        for item in self._eval_items:
            report.item_metrics[item.item_id] = ItemMetrics(
                item_id=item.item_id,
                label=item.label,
            )

        # Only evaluate admissions present in both
        common_ids = set(ground_truth.keys()) & set(predictions.keys())
        report.num_samples = len(common_ids)

        if not common_ids:
            logger.warning("No overlapping hadm_ids between ground truth and predictions.")
            return report

        for hadm_id in common_ids:
            gt = ground_truth[hadm_id]
            pred = predictions[hadm_id]

            for item in self._eval_items:
                item_id = item.item_id
                metrics = report.item_metrics[item_id]

                gt_positive = self._is_positive(gt.get(item_id))
                pred_positive = self._is_positive(pred.get_field(item_id))

                if gt_positive and pred_positive:
                    metrics.tp += 1
                elif not gt_positive and pred_positive:
                    metrics.fp += 1
                elif gt_positive and not pred_positive:
                    metrics.fn += 1
                else:
                    metrics.tn += 1

        return report

    @staticmethod
    def _is_positive(value: Any) -> bool:
        """Determine if a value represents a positive/present state."""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (list, tuple)):
            # Multi-select: positive if any non-"Z" code present
            return any(str(v).upper() != "Z" for v in value)
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, str):
            lower = value.strip().lower()
            return lower in {"true", "yes", "1", "present", "confirmed"}
        return False
