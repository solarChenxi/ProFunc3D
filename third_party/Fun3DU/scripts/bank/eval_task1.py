#!/usr/bin/env python3
"""SceneFun3D Task 1 evaluation for the class-aware prototype bank.

Task 1 is per-affordance-category functional part segmentation, so a proposal
is only correct when it overlaps a GT part *and* carries the right affordance
label. Existing SceneFun3D work only reports Task 2 (task-driven grounding),
so there is no published number to copy: this computes it from scratch.

Protocol (ScanNet/VOC style, per affordance class):
  * predictions = every proposal whose argmax class is that class, pooled over
    visits, sorted by classifier confidence;
  * greedy matching inside each visit, each GT consumed at most once, a
    proposal matching an already-consumed GT counts as a false positive;
  * AP = all-point interpolated area under the precision/recall curve;
  * mAP = macro mean over classes that have at least one GT part.

Alongside AP it reports three recalls so failures can be attributed:
  * class-agnostic recall  -> proposal-quality ceiling (ignores the label);
  * class top-1 / top-2 recall -> what the classifier keeps of that ceiling;
  * oracle-class AP        -> AP if every matched proposal were labelled by its
    GT, i.e. the ceiling AP given these proposals.

Consumes the `--dump-task1` npz written by `label_proposals_affordance.py`, so
it never has to reload the laser scans.

Usage:
  python \\
    scripts/bank/eval_task1.py \\
      --runs 'exps/prototype_bank/val_task1/*/task1_input.npz' \\
      --out exps/prototype_bank/val_task1/task1_eval.json
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def voc_ap(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP from confidence-sorted TP/FP indicators."""
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp).astype(np.float64)
    cfp = np.cumsum(fp).astype(np.float64)
    rec = ctp / float(n_gt)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    # precision envelope: replace each precision by the max to its right
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class Visit:
    """One scene: proposal posteriors plus the proposal x GT IoU matrix."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.path = path
        self.visit = str(z["visit"])
        self.classes = [str(c) for c in z["classes"]]
        self.proba = z["proba"].astype(np.float64)  # (n_prop, n_cls)
        self.iou = z["iou"].astype(np.float64)  # (n_prop, n_gt)
        self.gt_labels = [str(s) for s in z["gt_labels"]]
        self.gt_annot_ids = [str(s) for s in z["gt_annot_ids"]]
        self.prop_src = [str(s) for s in z["prop_src"]]
        self.gt_parent_cover = (
            z["gt_parent_cover"].astype(np.float64)
            if "gt_parent_cover" in z
            else np.full(len(self.gt_labels), np.nan)
        )
        if self.proba.shape[0] != self.iou.shape[0]:
            raise RuntimeError(f"{path}: proposal count mismatch")
        if self.iou.shape[1] != len(self.gt_labels):
            raise RuntimeError(f"{path}: GT count mismatch")

    @property
    def n_prop(self) -> int:
        return self.iou.shape[0]

    @property
    def n_gt(self) -> int:
        return self.iou.shape[1]

    def argmax_pred(self) -> Tuple[List[str], np.ndarray]:
        j = np.argmax(self.proba, axis=1)
        return [self.classes[k] for k in j], self.proba[np.arange(self.n_prop), j]

    def topk_pred(self, k: int) -> List[List[str]]:
        order = np.argsort(-self.proba, axis=1)[:, :k]
        return [[self.classes[j] for j in row] for row in order]


def gather_detections(
    visits: Sequence[Visit], oracle: bool, iou_thr_oracle: float
) -> Tuple[Dict[str, List[Tuple[float, int, int]]], Dict[str, Dict[str, int]]]:
    """Return per-class detection lists and the per-visit GT index by class.

    Each detection is (score, visit_index, proposal_index). With `oracle=True`
    a proposal that overlaps some GT above `iou_thr_oracle` is re-assigned that
    GT's label, which isolates proposal quality from classification quality.
    """
    dets: Dict[str, List[Tuple[float, int, int]]] = defaultdict(list)
    for vi, v in enumerate(visits):
        preds, scores = v.argmax_pred()
        for pi in range(v.n_prop):
            lab = preds[pi]
            if oracle and v.n_gt:
                j = int(np.argmax(v.iou[pi]))
                if v.iou[pi, j] >= iou_thr_oracle:
                    lab = v.gt_labels[j]
            dets[lab].append((float(scores[pi]), vi, pi))
    return dets, {}


def ap_for_class(
    visits: Sequence[Visit],
    dets: Sequence[Tuple[float, int, int]],
    label: str,
    iou_thr: float,
) -> Tuple[float, int, int]:
    """Greedy per-visit matching -> AP for one affordance class."""
    gt_idx = {
        vi: [j for j, l in enumerate(v.gt_labels) if l == label]
        for vi, v in enumerate(visits)
    }
    n_gt = sum(len(v) for v in gt_idx.values())
    if n_gt == 0:
        return float("nan"), 0, len(dets)
    taken = {vi: np.zeros(len(js), dtype=bool) for vi, js in gt_idx.items()}
    order = sorted(range(len(dets)), key=lambda i: -dets[i][0])
    tp = np.zeros(len(dets), dtype=np.float64)
    fp = np.zeros(len(dets), dtype=np.float64)
    for rank, di in enumerate(order):
        _, vi, pi = dets[di]
        js = gt_idx.get(vi, [])
        if not js:
            fp[rank] = 1.0
            continue
        ious = visits[vi].iou[pi, js]
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr and not taken[vi][best]:
            taken[vi][best] = True
            tp[rank] = 1.0
        else:
            fp[rank] = 1.0
    return voc_ap(tp, fp, n_gt), n_gt, len(dets)


def recalls(visits: Sequence[Visit], iou_thr: float) -> Dict[str, Dict[str, float]]:
    """Per-class class-agnostic / top-1 / top-2 recall (a GT-side view)."""
    acc: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for v in visits:
        preds, _ = v.argmax_pred()
        top2 = v.topk_pred(2)
        for j, lab in enumerate(v.gt_labels):
            col = v.iou[:, j] if v.n_prop else np.zeros(0)
            hit_any = bool(col.size and col.max() >= iou_thr)
            hit_1 = bool(
                any(col[i] >= iou_thr and preds[i] == lab for i in range(v.n_prop))
            )
            hit_2 = bool(
                any(col[i] >= iou_thr and lab in top2[i] for i in range(v.n_prop))
            )
            acc[lab]["agnostic"].append(hit_any)
            acc[lab]["top1"].append(hit_1)
            acc[lab]["top2"].append(hit_2)
    out = {}
    for lab, d in acc.items():
        out[lab] = {"n": len(d["agnostic"])}
        for k, vals in d.items():
            out[lab][k] = float(np.mean(vals))
    return out


def per_visit_recall(visits: Sequence[Visit], iou_thr: float) -> List[Dict]:
    """Same recalls, one row per scene, to expose between-scene variance."""
    rows = []
    for v in visits:
        preds, _ = v.argmax_pred()
        top2 = v.topk_pred(2)
        agn, t1, t2 = [], [], []
        for j in range(v.n_gt):
            col = v.iou[:, j] if v.n_prop else np.zeros(0)
            agn.append(bool(col.size and col.max() >= iou_thr))
            t1.append(any(col[i] >= iou_thr and preds[i] == v.gt_labels[j]
                          for i in range(v.n_prop)))
            t2.append(any(col[i] >= iou_thr and v.gt_labels[j] in top2[i]
                          for i in range(v.n_prop)))
        rows.append({
            "visit": v.visit, "n_gt": v.n_gt, "n_prop": v.n_prop,
            "recall_agnostic": float(np.mean(agn)) if agn else float("nan"),
            "recall_top1": float(np.mean(t1)) if t1 else float("nan"),
            "recall_top2": float(np.mean(t2)) if t2 else float("nan"),
        })
    return rows


def matched_classification(
    visits: Sequence[Visit], iou_thr: float
) -> Dict[str, object]:
    """Classifier accuracy restricted to proposals that do hit a GT part.

    The discriminator was fitted on GT masks but is applied to proposal masks;
    this is the direct measurement of that domain shift.
    """
    y_true: List[str] = []
    y_pred: List[str] = []
    y_top2: List[bool] = []
    for v in visits:
        preds, _ = v.argmax_pred()
        top2 = v.topk_pred(2)
        for pi in range(v.n_prop):
            if not v.n_gt:
                continue
            j = int(np.argmax(v.iou[pi]))
            if v.iou[pi, j] < iou_thr:
                continue
            y_true.append(v.gt_labels[j])
            y_pred.append(preds[pi])
            y_top2.append(v.gt_labels[j] in top2[pi])
    if not y_true:
        return {"n": 0}
    per: Dict[str, Dict[str, float]] = {}
    for lab in sorted(set(y_true)):
        sel = [i for i, t in enumerate(y_true) if t == lab]
        per[lab] = {
            "n": len(sel),
            "top1": float(np.mean([y_pred[i] == lab for i in sel])),
            "top2": float(np.mean([y_top2[i] for i in sel])),
        }
    conf = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    return {
        "n": len(y_true),
        "top1": float(np.mean([p == t for p, t in zip(y_pred, y_true)])),
        "top2": float(np.mean(y_top2)),
        "per_label": per,
        "top_confusions": [
            {"true": t, "pred": p, "n": n} for (t, p), n in conf.most_common(10)
        ],
    }


def class_agnostic_ap(
    visits: Sequence[Visit], iou_thr: float
) -> Tuple[float, int, int]:
    """AP with every proposal pooled into one class, scored by max posterior.

    The discriminator has no background class, so max posterior is a weak
    objectness proxy; read this as a sanity check, not as a headline number.
    """
    dets = []
    for vi, v in enumerate(visits):
        _, scores = v.argmax_pred()
        for pi in range(v.n_prop):
            dets.append((float(scores[pi]), vi, pi))
    gt_idx = {vi: list(range(v.n_gt)) for vi, v in enumerate(visits)}
    n_gt = sum(len(js) for js in gt_idx.values())
    if n_gt == 0:
        return float("nan"), 0, len(dets)
    taken = {vi: np.zeros(len(js), dtype=bool) for vi, js in gt_idx.items()}
    order = sorted(range(len(dets)), key=lambda i: -dets[i][0])
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))
    for rank, di in enumerate(order):
        _, vi, pi = dets[di]
        js = gt_idx[vi]
        if not js:
            fp[rank] = 1.0
            continue
        ious = visits[vi].iou[pi, js]
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr and not taken[vi][best]:
            taken[vi][best] = True
            tp[rank] = 1.0
        else:
            fp[rank] = 1.0
    return voc_ap(tp, fp, n_gt), n_gt, len(dets)


def error_attribution(
    visits: Sequence[Visit], iou_thr: float, cover_thr: float = 0.5
) -> Dict[str, object]:
    """Split every missed GT part into parent / proposal / classification loss.

    The three stages are nested: a part must sit inside some parent ROI before a
    proposer can hit it, and must be hit before the classifier can label it.
    """
    n = 0
    in_parent = 0
    hit = 0
    labelled = 0
    per_label: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "in_parent": 0, "hit": 0, "labelled": 0}
    )
    for v in visits:
        preds, _ = v.argmax_pred()
        for j, lab in enumerate(v.gt_labels):
            n += 1
            d = per_label[lab]
            d["n"] += 1
            cov = v.gt_parent_cover[j]
            ok_parent = bool(np.isnan(cov) or cov >= cover_thr)
            col = v.iou[:, j] if v.n_prop else np.zeros(0)
            ok_hit = bool(col.size and col.max() >= iou_thr)
            ok_lab = bool(
                any(col[i] >= iou_thr and preds[i] == lab for i in range(v.n_prop))
            )
            in_parent += ok_parent
            hit += ok_hit
            labelled += ok_lab
            d["in_parent"] += ok_parent
            d["hit"] += ok_hit
            d["labelled"] += ok_lab
    if n == 0:
        return {}
    return {
        "n_gt": n,
        "cover_thr": cover_thr,
        "frac_in_parent_roi": in_parent / n,
        "frac_hit": hit / n,
        "frac_hit_and_labelled": labelled / n,
        "loss_parent": (n - in_parent) / n,
        "loss_proposal": (in_parent - hit) / n,
        "loss_classification": (hit - labelled) / n,
        "per_label": {k: dict(v) for k, v in per_label.items()},
    }


def fmt(x: Optional[float], width: int = 6) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return " " * (width - 1) + "-"
    return f"{x:{width}.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        default="exps/prototype_bank/val_task1/*/task1_input.npz",
        help="glob of per-visit task1_input.npz dumps",
    )
    ap.add_argument("--iou-thrs", default="0.25,0.5")
    ap.add_argument("--out", default="exps/prototype_bank/val_task1/task1_eval.json")
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.runs))
    if not paths:
        raise SystemExit(f"no runs matched {args.runs}")
    visits = [Visit(p) for p in paths]
    thrs = [float(t) for t in args.iou_thrs.split(",")]

    n_prop = sum(v.n_prop for v in visits)
    n_gt = sum(v.n_gt for v in visits)
    src_mix = Counter(s for v in visits for s in v.prop_src)
    print(
        f"[eval] visits={len(visits)} proposals={n_prop} gt_parts={n_gt} "
        f"src={dict(src_mix)}",
        flush=True,
    )

    labels = sorted({l for v in visits for l in v.gt_labels})
    results: Dict[str, object] = {
        "n_visits": len(visits),
        "visits": [v.visit for v in visits],
        "n_proposals": n_prop,
        "n_gt": n_gt,
        "proposal_src_mix": dict(src_mix),
        "per_visit": [
            {"visit": v.visit, "n_prop": v.n_prop, "n_gt": v.n_gt} for v in visits
        ],
        "by_thr": {},
    }

    for thr in thrs:
        dets_pred, _ = gather_detections(visits, oracle=False, iou_thr_oracle=thr)
        dets_orc, _ = gather_detections(visits, oracle=True, iou_thr_oracle=thr)
        rec = recalls(visits, thr)
        per_class = {}
        for lab in labels:
            ap_pred, ngt, npred = ap_for_class(visits, dets_pred.get(lab, []), lab, thr)
            ap_orc, _, npred_o = ap_for_class(visits, dets_orc.get(lab, []), lab, thr)
            r = rec.get(lab, {})
            per_class[lab] = {
                "n_gt": ngt,
                "n_pred": npred,
                "ap": ap_pred,
                "ap_oracle_class": ap_orc,
                "n_pred_oracle": npred_o,
                "recall_agnostic": r.get("agnostic"),
                "recall_top1": r.get("top1"),
                "recall_top2": r.get("top2"),
            }
        valid = [v["ap"] for v in per_class.values() if not np.isnan(v["ap"])]
        valid_o = [
            v["ap_oracle_class"]
            for v in per_class.values()
            if not np.isnan(v["ap_oracle_class"])
        ]
        agn_ap, _, _ = class_agnostic_ap(visits, thr)
        micro = {
            k: float(
                np.sum([per_class[l]["n_gt"] * per_class[l][k] for l in per_class])
                / max(sum(per_class[l]["n_gt"] for l in per_class), 1)
            )
            for k in ("recall_agnostic", "recall_top1", "recall_top2")
        }
        results["by_thr"][f"{thr}"] = {
            "per_class": per_class,
            "mAP": float(np.mean(valid)) if valid else 0.0,
            "mAP_oracle_class": float(np.mean(valid_o)) if valid_o else 0.0,
            "class_agnostic_ap": agn_ap,
            "recall_micro": micro,
            "matched_classification": matched_classification(visits, thr),
            "per_visit": per_visit_recall(visits, thr),
            "attribution": error_attribution(visits, thr),
        }

        blk = results["by_thr"][f"{thr}"]
        print(f"\n===== IoU >= {thr}")
        print(
            f"{'class':12s} {'n_gt':>5s} {'n_pred':>7s} {'AP':>6s} "
            f"{'AP_orc':>7s} {'R_agn':>6s} {'R_top1':>7s} {'R_top2':>7s}"
        )
        for lab in labels:
            c = per_class[lab]
            print(
                f"{lab:12s} {c['n_gt']:5d} {c['n_pred']:7d} {fmt(c['ap'])} "
                f"{fmt(c['ap_oracle_class'], 7)} {fmt(c['recall_agnostic'])} "
                f"{fmt(c['recall_top1'], 7)} {fmt(c['recall_top2'], 7)}"
            )
        print(
            f"{'mAP':12s} {'':5s} {'':7s} {fmt(blk['mAP'])} "
            f"{fmt(blk['mAP_oracle_class'], 7)} {fmt(micro['recall_agnostic'])} "
            f"{fmt(micro['recall_top1'], 7)} {fmt(micro['recall_top2'], 7)}"
        )
        print(f"  class-agnostic AP (max-posterior score) : {fmt(agn_ap)}")
        mc = blk["matched_classification"]
        if mc.get("n"):
            print(
                f"  classifier on matched proposals: n={mc['n']} "
                f"top1={mc['top1']:.3f} top2={mc['top2']:.3f}"
            )
            print(
                "  top confusions:",
                ", ".join(
                    f"{c['true']}->{c['pred']}({c['n']})" for c in mc["top_confusions"][:5]
                ),
            )
        pv = blk["per_visit"]
        ra = np.array([r["recall_agnostic"] for r in pv if r["n_gt"] > 0])
        r1 = np.array([r["recall_top1"] for r in pv if r["n_gt"] > 0])
        print(
            f"  per-visit recall spread (n={len(ra)} scenes with GT): "
            f"agnostic mean={ra.mean():.3f} sd={ra.std():.3f} "
            f"min={ra.min():.3f} max={ra.max():.3f} | "
            f"top1 mean={r1.mean():.3f} sd={r1.std():.3f}"
        )
        at = blk["attribution"]
        if at:
            print(
                f"  attribution over {at['n_gt']} GT parts: "
                f"in parent ROI {at['frac_in_parent_roi']:.3f} -> "
                f"proposal hit {at['frac_hit']:.3f} -> "
                f"correctly labelled {at['frac_hit_and_labelled']:.3f}"
            )
            print(
                f"    loss: parent {at['loss_parent']:.3f} | "
                f"proposal {at['loss_proposal']:.3f} | "
                f"classification {at['loss_classification']:.3f}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] {out}", flush=True)


if __name__ == "__main__":
    main()
