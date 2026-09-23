"""
Fun3DU LLM stage: K open hypotheses (+ optional visual verification).

Replaces single-shot CoT with:
  1) Generate K diverse functional hypotheses (language)
  2) Optionally score them with scene evidence (OWL mask index / detection)
  3) Emit Fun3DU-compatible CoT: acted_on_object + hierarchy

Key design (spatial landmarks):
  Landmarks like TV / bed / wall are NOT forbidden. They are first-class
  *spatial_anchors* used to locate the controller ("switch next to the TV").
  Fun3DU uses hierarchy[0] as the OWL contextual object for frame retrieval,
  so hierarchy[0] may legitimately be a landmark when that yields better views.
  Visual verification chooses among controller vs anchor retrieval keys.

Backup of the previous multi-mode (skill/gcr/hyp) file:
  run_llm.py.bak_skill_gcr_hyp_20260723
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import ollama
from utils.misc import select_visits, sort_alphanumeric
from utils.sun3d.data_parser import DataParser

OLLAMA_PORT = 11434
OLLAMA_MODEL = "llama3.1"

# ---------------------------------------------------------------------------
# v2 version of the prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You extract structured JSON for a robot that must touch FINE interactive parts "
    "in indoor 3D scans (handles, knobs, buttons, switches, plugs, sockets, dials, latches, levers).\n"
    "\n"
    "HARD RULES for acted_on_object:\n"
    "- Must be the part the hand/tool physically contacts.\n"
    "- For open/close drawer/door/window/fridge/cabinet/closet: ALWAYS use a *handle* "
    "(e.g. drawer handle, door handle, window handle, cabinet door handle). "
    "Never output bare drawer/door/window/cabinet/fridge/closet.\n"
    "- Ceiling / room lights: light switch / toggle button / dimmer switch / dimmer knob "
    "(NOT the bulb or fixture).\n"
    "- Table lamps / floor lamps: for turn on/off or unplug, prefer power plug / lamp switch / "
    "dimmer knob on that lamp — NOT a wall light switch and NOT bed/table as the target.\n"
    "- For plug-in tasks: socket/outlet; for unplug: power plug / plug.\n"
    "\n"
    "HARD RULES for acted_on_object_hierarchy (coarse -> fine):\n"
    "- hierarchy[0] is the PARENT for image retrieval. Prefer the named furniture that owns "
    "the part: cabinet, nightstand, dresser, desk, door, window, closet, wardrobe, fridge, "
    "oven, radiator, shelf, TV stand, dressing table, vanity table, center table, coffee table.\n"
    "- Named '* table' furniture (dressing table, vanity table, center table, coffee table, "
    "bedside table) ARE valid parents. Bare 'table' alone is NOT.\n"
    "- NEVER use as hierarchy[0]: TV, television, bed, chair, sofa, couch, room, bedroom, "
    "kitchen, bathroom, wall, floor, ceiling, living room, shutters, device, person, "
    "or bare 'table'.\n"
    "- For a dimmer/switch on a wall: set hierarchy[0] to 'dimmer switch' or 'light switch' "
    "(single-element hierarchy is OK) — do NOT use 'wall'.\n"
    "- For lamp plug/switch: hierarchy[0] may be the specific lamp name from the task "
    "(e.g. 'red table lamp', 'floor lamp') so retrieval finds that lamp.\n"
    "- For doors: hierarchy[0]='door' (or 'cabinet'/'closet' if it is a cabinet/closet door). "
    "Do NOT use nearby clutter (pet box, chair) as hierarchy[0].\n"
    "- For remote control: hierarchy should be [\"remote control\"] (or remote + button). "
    "Do NOT use the table the remote rests on as hierarchy[0].\n"
    "- hierarchy[-1] must equal acted_on_object.\n"
    "- Typical length 1-3.\n"
    "\n"
    "Actions may include: rotate, key_press, tip_push, hook_pull, pinch_pull, hook_turn, "
    "foot_push, plug_in, unplug."
)

USER_TEMPLATE = """How do I {query}?
Respond with ONLY one JSON object (no markdown fences) using exactly these keys:
{{
  "prompt": "<copy the task string>",
  "task_solving_sequence": ["short substep", "..."],
  "acted_on_object": "<fine part to touch>",
  "acted_on_object_hierarchy": ["<parent>", "...", "<same as acted_on_object>"]
}}

Gold examples (follow this style exactly):
1) Open the fifth drawer of the cabinet located to the left of the TV
   {{"acted_on_object":"drawer handle","acted_on_object_hierarchy":["cabinet","drawer","drawer handle"]}}
2) Open the bottom drawer of the nightstand next to the closet
   {{"acted_on_object":"drawer handle","acted_on_object_hierarchy":["nightstand","drawer handle"]}}
3) Open the middle left drawer of the dressing table
   {{"acted_on_object":"drawer handle","acted_on_object_hierarchy":["dressing table","drawer","drawer handle"]}}
4) Close the bedroom door
   {{"acted_on_object":"door handle","acted_on_object_hierarchy":["door","door handle"]}}
5) Open the left door of the TV stand
   {{"acted_on_object":"door handle","acted_on_object_hierarchy":["TV stand","door","door handle"]}}
6) Open the left window behind the shutters
   {{"acted_on_object":"window handle","acted_on_object_hierarchy":["window","window handle"]}}
7) Turn on the ceiling light
   {{"acted_on_object":"light switch","acted_on_object_hierarchy":["light switch"]}}
8) Control the light intensity using the dimmer on the wall
   {{"acted_on_object":"dimmer knob","acted_on_object_hierarchy":["dimmer switch","dimmer knob"]}}
9) Turn on the red table lamp next to the bed
   {{"acted_on_object":"power plug","acted_on_object_hierarchy":["red table lamp","power plug"]}}
10) Unplug the floor lamp next to the dining table
   {{"acted_on_object":"power plug","acted_on_object_hierarchy":["floor lamp","power plug"]}}
11) Plug the device in the right socket between the radiator and the closet
   {{"acted_on_object":"socket","acted_on_object_hierarchy":["radiator","socket"]}}
12) Unplug the TV from the power supply
   {{"acted_on_object":"power plug","acted_on_object_hierarchy":["TV stand","power plug"]}}
13) Turn on the TV using the remote control on the table
   {{"acted_on_object":"remote control","acted_on_object_hierarchy":["remote control"]}}
"""


# ---------------------------------------------------------------------------
# K open hypotheses
# ---------------------------------------------------------------------------

HYP_SYSTEM = (
    "You generate MULTIPLE open hypotheses for functional 3D interaction grounding.\n"
    "A robot must physically touch a FINE interactive element "
    "(handle, switch, knob, plug, socket, remote, dial, latch, lever, button).\n"
    "Do NOT touch the effect object alone when a controller exists "
    "(e.g. do not point at the bulb/TV screen to 'turn on' — point at switch/remote).\n"
    "\n"
    "IMPORTANT — spatial landmarks ARE allowed and often necessary:\n"
    "  Phrases like 'next to the TV', 'beside the bed', 'on the wall' mean you should "
    "record those objects as spatial_anchors. They help find WHERE the controller is.\n"
    "  Do NOT confuse anchors with the contact part: contact is still the switch/handle/...\n"
    "\n"
    "For EACH hypothesis provide:\n"
    "- control_mode: local_part | on_device | spatial_remote\n"
    "  local_part: part on furniture (drawer handle on cabinet)\n"
    "  on_device: control on the device (lamp switch on table lamp)\n"
    "  spatial_remote: controller separated from effect "
    "(wall switch↔ceiling light; remote↔TV; wall socket)\n"
    "- effect_object: what the task changes (ceiling light, TV, drawer, ...)\n"
    "- spatial_anchors: list of reference objects from the task used for localization "
    "(e.g. [\"TV\"] for 'switch next to the TV'; [\"bed\"] for 'lamp next to the bed'). "
    "Empty list if none.\n"
    "- acted_on_object: FINE contact part. Prefer canonical names when applicable: "
    "drawer handle, door handle, window handle, light switch, dimmer knob, dimmer switch, "
    "power plug, socket, remote control, thermostat knob, radiator dial, lamp switch. "
    "No numbered names like 'drawer handle 2'. Avoid bare 'handle'/'switch' if a specific "
    "name fits.\n"
    "- acted_on_object_hierarchy: coarse→fine; hierarchy[-1]==acted_on_object. "
    "hierarchy[0] is a CANDIDATE for frame retrieval — may be owning furniture, the "
    "controller itself, OR a spatial anchor when that best localizes the target.\n"
    "- controller_query: short open-vocab phrase for detecting/pointing the contact part "
    "(e.g. 'light switch', 'drawer handle', 'remote control').\n"
    "- anchor_query: short phrase for detecting the main spatial anchor, or \"\" if none "
    "(e.g. 'TV', 'bed', 'radiator').\n"
    "- verify_cues: 1-3 visual checks (what should be co-visible if this hyp is correct)\n"
    "- rationale: one short sentence\n"
    "- confidence: float 0-1\n"
    "\n"
    "Diversity:\n"
    "- Cover distinct contact targets and/or retrieval strategies when plausible.\n"
    "- For light/TV/power tasks include at least one spatial_remote hypothesis.\n"
    "- For open/close drawer/door/window include a handle local_part hypothesis.\n"
    "- For 'switch next to the TV', one strong hyp should keep TV as spatial_anchor "
    "and hierarchy[0] may be TV (retrieve frames of the TV, then find the switch).\n"
    "- Do not propose unrelated controllers (light switch for a pure drawer task).\n"
    "\n"
    "Respond with ONLY valid JSON (no markdown):\n"
    '{ \"hypotheses\": [ { ... }, { ... } ] }\n'
)

HYP_USER = """Task: {query}

Generate exactly {k} diverse hypotheses now."""


def _client() -> ollama.Client:
    return ollama.Client(host=f"http://127.0.0.1:{OLLAMA_PORT}", trust_env=False)


def _chat(system: str, user: str) -> str:
    response = _client().chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"]


def parse_json_response(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    def _try_load(s: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    if "```" in text:
        body = text.split("```")[1]
        if body.startswith("json"):
            body = body[4:]
        got = _try_load(body.strip())
        if got is not None:
            return got

    i, j = text.find("{"), text.rfind("}")
    blob = text[i : j + 1] if i >= 0 and j > i else text
    got = _try_load(blob)
    if got is not None:
        return got

    repaired = re.sub(r"\n\s*", " ", blob)
    got = _try_load(repaired)
    if got is not None:
        return got

    return {"_parse_error": True, "_raw": text[:500]}


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _space_name(s: str) -> str:
    s = re.sub(r"[_\-]+", " ", str(s or ""))
    return re.sub(r"\s+", " ", s).strip()


_FINE_RE = re.compile(
    r"handle|knob|button|switch|lever|latch|lock|plug|socket|outlet|key|dial|"
    r"slider|hinge|pull|tab|cord|cable|remote|joystick|thermostat",
    re.I,
)


def _is_fine_contact(name: str) -> bool:
    return bool(_FINE_RE.search(_norm_name(name)))


def _strip_meta_for_save(cot: Dict[str, Any]) -> Dict[str, Any]:
    if not cot:
        return {}
    return {k: v for k, v in cot.items() if not str(k).startswith("_")}


# ---------------------------------------------------------------------------
# Parse / score / convert hypotheses
# ---------------------------------------------------------------------------

def parse_hypotheses_response(text: str, k: int) -> list:
    obj = parse_json_response(text)
    hyps: list = []
    if isinstance(obj, dict) and not obj.get("_parse_error"):
        raw = obj.get("hypotheses")
        if isinstance(raw, list):
            hyps = [h for h in raw if isinstance(h, dict)]
        elif "acted_on_object" in obj:
            hyps = [obj]

    if not hyps:
        text = (text or "").strip()
        i, j = text.find("["), text.rfind("]")
        if i >= 0 and j > i:
            try:
                arr = json.loads(text[i : j + 1])
                if isinstance(arr, list):
                    hyps = [h for h in arr if isinstance(h, dict)]
            except Exception:
                pass

    out = []
    for h in hyps[: max(k, 1)]:
        acted = h.get("acted_on_object") or h.get("contact_part") or ""
        hier = h.get("acted_on_object_hierarchy") or h.get("hierarchy") or []
        if isinstance(hier, str):
            hier = [hier]
        hier = list(hier)
        if acted and (not hier or _norm_name(str(hier[-1])) != _norm_name(str(acted))):
            if hier:
                hier[-1] = acted
            else:
                hier = [acted]
        anchors = h.get("spatial_anchors") or []
        if isinstance(anchors, str):
            anchors = [anchors] if anchors else []
        out.append(
            {
                "control_mode": h.get("control_mode") or "local_part",
                "effect_object": h.get("effect_object") or "",
                "spatial_anchors": list(anchors),
                "acted_on_object": acted,
                "acted_on_object_hierarchy": hier,
                "controller_query": h.get("controller_query") or acted,
                "anchor_query": h.get("anchor_query")
                or (anchors[0] if anchors else ""),
                "verify_cues": h.get("verify_cues") or [],
                "rationale": h.get("rationale") or "",
                "confidence": float(h.get("confidence") or 0.5),
            }
        )
    return out


def score_hypothesis_language(h: Dict[str, Any], query: str = "") -> float:
    """Language-only prior (used when visual verify is off / unavailable)."""
    score = float(h.get("confidence") or 0.5)
    acted = _space_name(h.get("acted_on_object") or "")
    an = _norm_name(acted)
    mode = str(h.get("control_mode") or "")
    q = (query or "").lower()
    anchors = [_norm_name(_space_name(a)) for a in (h.get("spatial_anchors") or [])]

    canonical = {
        "drawer handle",
        "door handle",
        "window handle",
        "light switch",
        "dimmer switch",
        "dimmer knob",
        "power plug",
        "socket",
        "remote control",
        "thermostat knob",
        "radiator dial",
        "lamp switch",
    }
    if an in canonical:
        score += 0.8
    if re.search(r"\d", an) or an in {"handle", "button", "switch", "knob", "latch"}:
        score -= 0.8
    if _is_fine_contact(acted):
        score += 1.5
    else:
        score -= 1.0
    if an in {"tv screen", "screen", "bulb", "ceiling light", "light fixture"}:
        score -= 1.5

    # Reward using anchors when the task mentions them (do NOT ban landmarks)
    for a in anchors:
        if a and a in q:
            score += 0.4
    if "next to the tv" in q or "beside the tv" in q or "near the tv" in q:
        if any(a in {"tv", "television"} for a in anchors):
            score += 0.6
        h0 = _norm_name(_space_name((h.get("acted_on_object_hierarchy") or [""])[0]))
        if h0 in {"tv", "television"}:
            score += 0.5
    # If the task explicitly says switch, prefer switch over remote/plug
    if "switch" in q:
        if "switch" in an:
            score += 0.8
        if "remote" in an:
            score -= 1.0
    if "remote" in q and "switch" not in q and "remote" in an:
        score += 0.5

    if any(k in q for k in ("open", "close")) and any(
        k in q for k in ("drawer", "door", "window")
    ):
        if "handle" in an:
            score += 0.8
        if mode == "spatial_remote":
            score -= 1.0
    if any(k in q for k in ("plug the", "plug in", "socket")) and "unplug" not in q:
        if "socket" in an or "outlet" in an:
            score += 1.0
        if an in {"plug", "power plug"}:
            score -= 0.8
    if "unplug" in q and "plug" in an:
        score += 0.6
    if any(k in q for k in ("light", "lamp", "tv", "remote", "dimmer")):
        if mode in ("spatial_remote", "on_device"):
            score += 0.3
    return score


def hypothesis_to_cot(
    h: Dict[str, Any],
    query: str,
    retrieval_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Map hypothesis → Fun3DU CoT.
    retrieval_key overrides hierarchy[0] when visual verify picks a better OWL query
    (often a spatial anchor like TV, or the controller itself).
    """
    acted = _space_name(h.get("acted_on_object") or h.get("controller_query") or "")
    hier = [_space_name(x) for x in (h.get("acted_on_object_hierarchy") or [])]
    if acted and (not hier or _norm_name(str(hier[-1])) != _norm_name(acted)):
        if hier:
            hier[-1] = acted
        else:
            hier = [acted]

    if retrieval_key:
        key = _space_name(retrieval_key)
        if hier:
            hier[0] = key
        else:
            hier = [key, acted] if acted and _norm_name(key) != _norm_name(acted) else [acted]
    # dedupe consecutive
    out = []
    for x in hier:
        if not out or _norm_name(out[-1]) != _norm_name(x):
            out.append(x)
    hier = out

    cq = _space_name(h.get("controller_query") or acted)
    anchors = h.get("spatial_anchors") or []
    seq = [f"locate {cq}", "make contact"]
    if anchors:
        seq.insert(0, "use spatial anchor: " + ", ".join(_space_name(a) for a in anchors))

    return {
        "prompt": query,
        "task_solving_sequence": seq,
        "acted_on_object": acted,
        "acted_on_object_hierarchy": hier,
    }


# ---------------------------------------------------------------------------
# Visual verification (scene evidence)
# ---------------------------------------------------------------------------

def load_visit_object_index(
    data_root: str, split: str, visit_id: str, mask_type: str
) -> Dict[str, List[str]]:
    """
    Load {label -> [\"video frame\", ...]} from Fun3DU mask index JSON if present.
    Written by run_detection.make_mask_index as {visit}_{mask_type}_masks.json
    """
    path = os.path.join(
        data_root, split, visit_id, f"{visit_id}_{mask_type}_masks.json"
    )
    if not os.path.isfile(path):
        return {}
    try:
        data = json.load(open(path))
    except Exception:
        return {}
    objects = data.get("objects") or {}
    # normalize keys
    return {_norm_name(k): v for k, v in objects.items() if isinstance(v, list)}


def _match_label_frames(
    object_index: Dict[str, List[str]], query: str
) -> Tuple[str, int]:
    """Return (matched_label, n_frames). Prefer exact / strong token overlap."""
    q = _norm_name(query)
    if not q or not object_index:
        return "", 0
    if q in object_index:
        return q, len(object_index[q])

    q_toks = set(q.split())
    # Drop ultra-generic single tokens that cause false hits
    weak = {"the", "a", "an", "on", "in", "to", "of", "and", "for", "with", "use"}
    q_toks = {t for t in q_toks if t not in weak and len(t) > 1}
    if not q_toks:
        return "", 0

    best_label, best_n, best_ov = "", 0, 0.0
    for lab, frames in object_index.items():
        lab_toks = set(lab.split())
        if not lab_toks:
            continue
        inter = q_toks & lab_toks
        if not inter:
            continue
        # Reject weak-only matches when the query has more content
        # (e.g. query "wall switch" must not match label "wall")
        if inter <= {"wall", "floor", "ceiling", "room"} and len(q_toks - inter) > 0:
            continue
        overlap = len(inter) / max(len(q_toks), len(lab_toks))
        # require at least one contentful shared token of length >= 3
        if not any(len(t) >= 3 for t in inter):
            continue
        n = len(frames)
        if overlap > best_ov + 1e-6 or (abs(overlap - best_ov) < 1e-6 and n > best_n):
            best_label, best_n, best_ov = lab, n, overlap
    # Require reasonable overlap (avoid "remote" matching via long garbage queries weakly)
    if best_ov < 0.34:
        return "", 0
    return best_label, best_n


def score_hypothesis_visual(
    h: Dict[str, Any], object_index: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Score one hypothesis using existing OWL detections in the scene.

    Returns dict with:
      visual_score, controller_hits, anchor_hits, retrieval_key, details

    Retrieval policy for Fun3DU hierarchy[0]:
      - If controller is well detected → prefer controller as retrieval key
      - Else if spatial anchor is well detected → use anchor (e.g. TV) so Molmo
        can point the switch in those frames
      - Else fall back to hierarchy[0] / controller_query
    """
    cq = _space_name(h.get("controller_query") or h.get("acted_on_object") or "")
    aq = _space_name(h.get("anchor_query") or "")
    anchors = [_space_name(a) for a in (h.get("spatial_anchors") or [])]
    if not aq and anchors:
        aq = anchors[0]

    c_lab, c_n = _match_label_frames(object_index, cq)
    # also try acted_on_object
    a_lab, a_n = _match_label_frames(
        object_index, _space_name(h.get("acted_on_object") or "")
    )
    if a_n > c_n:
        c_lab, c_n = a_lab, a_n

    anchor_hits = []
    best_anchor_n = 0
    best_anchor_lab = ""
    for cand in [aq] + anchors:
        if not cand:
            continue
        lab, n = _match_label_frames(object_index, cand)
        if n:
            anchor_hits.append({"query": cand, "label": lab, "n_frames": n})
            if n > best_anchor_n:
                best_anchor_n, best_anchor_lab = n, lab or cand

    # Co-visibility bonus: frames where both might appear (string overlap of keys)
    co_vis = 0
    if c_lab and best_anchor_lab and c_lab in object_index and best_anchor_lab in object_index:
        co_vis = len(set(object_index[c_lab]) & set(object_index[best_anchor_lab]))

    visual_score = 0.0
    visual_score += min(c_n, 30) * 0.15  # controller evidence
    visual_score += min(best_anchor_n, 30) * 0.10  # anchor evidence
    visual_score += min(co_vis, 20) * 0.2  # co-visible frames strongly preferred

    # Choose retrieval key for OWL / Fun3DU
    if c_n >= 3:
        retrieval_key = _space_name(h.get("acted_on_object") or cq)
    elif best_anchor_n >= 3:
        retrieval_key = best_anchor_lab or aq
    else:
        hier = h.get("acted_on_object_hierarchy") or []
        retrieval_key = _space_name(hier[0]) if hier else (cq or aq)

    return {
        "visual_score": visual_score,
        "controller_hits": c_n,
        "anchor_hits": best_anchor_n,
        "co_visible_frames": co_vis,
        "retrieval_key": retrieval_key,
        "matched_controller_label": c_lab,
        "matched_anchor_label": best_anchor_lab,
        "anchor_hit_details": anchor_hits,
    }


def select_hypothesis(
    hyps: List[Dict[str, Any]],
    query: str,
    object_index: Optional[Dict[str, List[str]]] = None,
    use_visual: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Rank hypotheses with language prior (+ visual if index available).
    Returns (best_hyp, all_scored, selection_meta).
    """
    scored = []
    for h in hyps:
        lang = score_hypothesis_language(h, query)
        hh = dict(h)
        hh["_lang_score"] = lang
        vis_meta = None
        if use_visual and object_index:
            vis_meta = score_hypothesis_visual(h, object_index)
            hh["_visual"] = vis_meta
            # Combine: visual dominates when any detections exist
            if vis_meta["controller_hits"] or vis_meta["anchor_hits"]:
                hh["_score"] = lang + vis_meta["visual_score"]
            else:
                hh["_score"] = lang
        else:
            hh["_score"] = lang
        scored.append(hh)

    scored.sort(key=lambda x: -float(x.get("_score") or 0))
    best = scored[0]
    retrieval_key = None
    if best.get("_visual"):
        retrieval_key = best["_visual"].get("retrieval_key")

    meta = {
        "k": len(scored),
        "selected_rank": 0,
        "selected_score": best.get("_score"),
        "lang_score": best.get("_lang_score"),
        "control_mode": best.get("control_mode"),
        "controller_query": best.get("controller_query"),
        "spatial_anchors": best.get("spatial_anchors"),
        "retrieval_key": retrieval_key,
        "used_visual": bool(use_visual and object_index),
        "modes": [h.get("control_mode") for h in scored],
    }
    if best.get("_visual"):
        meta["visual"] = {
            k: best["_visual"][k]
            for k in (
                "visual_score",
                "controller_hits",
                "anchor_hits",
                "co_visible_frames",
                "matched_controller_label",
                "matched_anchor_label",
            )
        }
    return best, scored, meta


def get_k_hypotheses(
    query: str,
    k: int = 3,
    object_index: Optional[Dict[str, List[str]]] = None,
    use_visual: bool = True,
) -> Tuple[Dict[str, Any], list]:
    """Generate K hypotheses, select with language (+ optional visual), return CoT."""
    k = max(int(k), 1)
    raw = _chat(HYP_SYSTEM, HYP_USER.format(query=query, k=k))
    hyps = parse_hypotheses_response(raw, k)

    if not hyps:
        draft_raw = _chat(SYSTEM_PROMPT, USER_TEMPLATE.format(query=query))
        draft = parse_json_response(draft_raw)
        hyps = [
            {
                "control_mode": "local_part",
                "effect_object": "",
                "spatial_anchors": [],
                "acted_on_object": draft.get("acted_on_object") or "",
                "acted_on_object_hierarchy": draft.get("acted_on_object_hierarchy")
                or [],
                "controller_query": draft.get("acted_on_object") or "",
                "anchor_query": "",
                "verify_cues": [],
                "rationale": "fallback single-shot",
                "confidence": 0.3,
            }
        ]

    best, scored, meta = select_hypothesis(
        hyps, query, object_index=object_index, use_visual=use_visual
    )
    cot = hypothesis_to_cot(best, query, retrieval_key=meta.get("retrieval_key"))
    cot["_hyp"] = meta
    return cot, scored


def get_LLM_response(statement, query):
    raw = _chat(SYSTEM_PROMPT, statement.format(query=query))
    return raw, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig):
    llm_type = str(args.llm_type)
    llm_mode = str(getattr(args, "llm_mode", "") or "")
    use_hyp = (
        llm_type.endswith("_hyp")
        or llm_type.endswith("_hyps")
        or llm_mode == "hyp"
    )
    hyp_k = int(getattr(args, "hyp_k", 3) or 3)
    use_visual = bool(getattr(args, "hyp_visual", True))
    mask_type = str(getattr(args, "mask_type", "owl2_rsam") or "owl2_rsam")

    parser = DataParser(args.dataset.root, args.dataset.split)
    visits = sort_alphanumeric(list(dict.fromkeys(parser.get_visits())))
    visit_ids = select_visits(visits, args.dataset)

    mode = f"hyp(K={hyp_k}, visual={use_visual})" if use_hyp else "single-shot"
    print(
        f"LLM processing [{mode}] for {len(visit_ids)} visits "
        f"(split {args.dataset.split}, llm_type={args.llm_type}): {visit_ids}"
    )

    for visit_id in tqdm(visit_ids):
        out_dir = f"{args.dataset.root}/{args.dataset.split}/{visit_id}"
        os.makedirs(out_dir, exist_ok=True)
        new_path = f"{out_dir}/{visit_id}_{args.llm_type}_cot.json"
        if (not bool(getattr(args, "overwrite", True))) and os.path.isfile(new_path):
            print(f"skip existing {new_path}")
            continue
        descs = parser.get_descriptions_list(visit_id)
        json_data = []
        hyp_data = []

        object_index = {}
        if use_hyp and use_visual:
            object_index = load_visit_object_index(
                args.dataset.root, args.dataset.split, visit_id, mask_type
            )
            if not object_index:
                print(
                    f"[{visit_id}] no mask index for {mask_type}; "
                    "falling back to language-only ranking"
                )

        for desc_id, query in descs.items():
            if use_hyp:
                cot, hyps = get_k_hypotheses(
                    query,
                    k=hyp_k,
                    object_index=object_index or None,
                    use_visual=use_visual,
                )
                meta = cot.pop("_hyp", {}) if isinstance(cot, dict) else {}
                json_data.append(_strip_meta_for_save(cot))
                hyp_data.append(
                    {
                        "desc_id": desc_id,
                        "query": query,
                        "selected": _strip_meta_for_save(cot),
                        "meta": meta,
                        "hypotheses": hyps,
                    }
                )
            else:
                response = get_LLM_response(USER_TEMPLATE, query)
                try:
                    if "```" in response[0]:
                        json_dict = json.loads(response[0].split("```")[1])
                    else:
                        json_dict = json.loads(response[0])
                    json_data.append(json_dict)
                except Exception as e:
                    print(f"error {e}")
                    json_data.append({})

        out_dir = f"{args.dataset.root}/{args.dataset.split}/{visit_id}"
        os.makedirs(out_dir, exist_ok=True)
        with open(new_path, "w") as out_f:
            json.dump(json_data, out_f, indent=4)

        if use_hyp and hyp_data:
            hyp_path = f"{out_dir}/{visit_id}_{args.llm_type}_hyps.json"
            with open(hyp_path, "w") as hf:
                json.dump(hyp_data, hf, indent=2)

        print(f"saved {new_path}")


if __name__ == "__main__":
    main()
