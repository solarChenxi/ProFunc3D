#!/usr/bin/env python3
"""Build atomic instance/relation contracts from query + existing Llama CoT.

The parser is deliberately deterministic: it adds no VLM/LLM call and never
looks at OWL labels or GT.  OWL aliases are attached only after the semantic
contract has been formed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_s = Path(__file__).resolve().parent
while _s.name != "scripts" and _s.parent != _s:
    _s = _s.parent
if str(_s) not in sys.path:
    sys.path.insert(0, str(_s))
import _pathsetup  # noqa: E402,F401
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]

from io_scene import make_parser  # noqa: E402
from utils import io as fio  # noqa: E402

OUT = ROOT / "exps/task_router/structured_contracts_val445.json"
COLORS = ("blue", "red", "white", "black", "brown", "green", "gray", "grey", "yellow")
MATERIALS = ("wooden", "wood", "glass", "leather", "metal")

ALIASES = {
    "closet": ["closet", "cabinet"],
    "counter": ["counter", "kitchen counter"],
    "kitchen counter": ["kitchen counter", "counter"],
    "tv stand": ["tv stand"],
    "lamp": ["lamp", "table lamp"],
    "sofa": ["sofa", "couch"],
    "couch": ["couch", "sofa"],
    "glass cabinet": ["glass cabinet", "display cabinet", "cabinet"],
    "wall cabinet": ["wall cabinet", "cabinet"],
    "teacups": ["teacups", "teacup", "cups"],
    "bird cage": ["bird cage", "birdcage"],
    "toys": ["toys", "toy"],
    "remote": ["remote control", "remote"],
    "remote control": ["remote control", "remote"],
}


def clean_np(s: str) -> str:
    s = s.lower().strip(" .,;:")
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    # Return an atomic detectable noun phrase.  Scene qualifiers remain useful
    # in the language contract, but must not become a compositional OWL key.
    s = re.split(r"\s+and\s+(?:near|next to)\s+the\s+", s)[0]
    s = re.split(r"\s+located\s+next\s+to\s+the\s+", s)[0]
    s = re.split(r"\s+next\s+to\s+(?:the|he)\s+", s)[0]
    s = re.sub(r"\s+located$", "", s)
    s = re.split(r"\s+(?:mounted\s+)?on\s+the\s+", s)[0]
    s = re.split(r"\s+with\s+the\s+", s)[0]
    s = re.split(r"\s+of\s+the\s+", s)[0]
    s = re.sub(r"\s+(?:decorated|covered)\s+with\b.*$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def retrieval_keys(name: str, attributes: list[str] | None = None) -> list[str]:
    name = clean_np(name)
    keys = list(ALIASES.get(name, [name]))
    for attr in attributes or []:
        if attr in COLORS:
            keys.insert(0, f"{attr} {name}")
    return list(dict.fromkeys(clean_np(x) for x in keys if clean_np(x)))


def parent_from_cot(query: str, acted: str, hierarchy: list[str]) -> str:
    acted = clean_np(acted)
    hs = [clean_np(x) for x in hierarchy if clean_np(x)]
    support = support_host_class(query)
    if support:
        return support
    # Some CoTs encode a landmark as hierarchy[0] for standalone controls,
    # e.g. [door, socket].  Those controls are their own trackable parent.
    if acted in {"socket", "electrical socket", "switch", "button", "thermostat"}:
        return acted
    return hs[0] if hs else acted


def support_host_class(query: str) -> str | None:
    """Entity placed on a support surface, excluding the leading verb 'turn on'."""
    q = query.lower().strip().rstrip(".")
    if "located on the left side of" in q or "located on the right side of" in q:
        return None
    pos = q.rfind(" on the ")
    if pos < 10:
        return None
    prefix = q[:pos]
    # These describe operating within an appliance, not an instance on a support.
    if re.search(r"\b(?:select|control|adjust|set)\b.*\b(?:function|setting|heat|functions)\s*$", prefix):
        return None
    vocab = [
        (r"\bremote controls?\b|\bremotes?\b", "remote control"),
        (r"\bkeyboards?\b", "keyboard"),
        (r"\bdimmers?\b", "dimmer switch"),
        (r"\bjoysticks?\b", "joystick"),
        (r"\btelephones?\b", "telephone"),
        (r"\btable lamps?\b", "table lamp"),
        (r"\blamps?\b", "lamp"),
        (r"\bchests?\b", "chest"),
        (r"\bmirror lights?\b", "mirror light"),
        (r"\bstorage box(?:es)?\b", "storage box"),
        (r"\bstereo systems?\b", "stereo system"),
        (r"\bpower strips?\b", "power strip"),
    ]
    for pat, canonical in vocab:
        if re.search(pat, prefix):
            return canonical
    return None


def parent_attributes(query: str, parent: str) -> list[str]:
    q = query.lower()
    attrs = [c for c in COLORS if re.search(rf"\b{c}\s+{re.escape(parent)}\b", q)]
    return attrs


REL_CUE = re.compile(
    r"\b(?:located\s+)?(?:directly\s+)?(?:to the right of|to the left of|above|below|under|"
    r"next to|near|in front of|behind|between)\b"
)


def relation_host_class(query: str) -> tuple[str, list[str]] | None:
    """Extract the entity constrained by an external relation from its prefix."""
    q = query.lower().strip().rstrip(".")
    m = REL_CUE.search(q)
    if not m:
        return None
    prefix = q[:m.start()].strip()
    # The entity after "of the" is normally the parent of the target part.
    ofs = list(re.finditer(r"\bof the\s+(.+)$", prefix))
    focus = ofs[-1].group(1) if ofs else prefix
    using = re.search(r"\busing (?:the\s+|one of the\s+)?(.+)$", prefix)
    if using:
        focus = using.group(1)
    on_support = re.search(r"\bon the\s+(.+)$", focus)
    if on_support:
        focus = on_support.group(1)

    vocab = [
        (r"\bwall cabinets?\b", "wall cabinet"),
        (r"\bkitchen counters?\b", "kitchen counter"),
        (r"\b(?:white|wooden|blue|glass) cabinets?\b", "cabinet"),
        (r"\b(?:white|wooden|blue) closets?\b", "closet"),
        (r"\bmini music systems?\b", "mini music system"),
        (r"\bspace heaters?\b", "space heater"),
        (r"\bremote controls?\b|\bremotes?\b", "remote control"),
        (r"\blight switch(?:es)?\b|\bswitch(?:es)?\b", "light switch"),
        (r"\bradiator dials?\b|\bradiators?\b", "radiator"),
        (r"\bdials?\b", "dial"),
        (r"\bthermostats?\b", "thermostat"),
        (r"\btelephones?\b", "telephone"),
        (r"\btrash bins?\b", "trash bin"),
        (r"\bjewelry box(?:es)?\b", "jewelry box"),
        (r"\bbread bins?\b", "bread bin"),
        (r"\bside tables?\b", "side table"),
        (r"\bcoffee tables?\b", "coffee table"),
        (r"\bwall cabinets?\b", "wall cabinet"),
        (r"\bnightstands?\b", "nightstand"),
        (r"\bdressing tables?\b", "dressing table"),
        (r"\btv stands?\b", "tv stand"),
        (r"\b(?:kitchen\s+)?counters?\b", "counter"),
        (r"\bcabinets?\b", "cabinet"),
        (r"\bclosets?\b", "closet"),
        (r"\bwindows?\b", "window"),
        (r"\bsockets?\b", "socket"),
        (r"\btable lamps?\b", "table lamp"),
        (r"\blamps?\b", "lamp"),
        (r"\bdoors?\b", "door"),
        (r"\bdrawers?\b", "drawer"),
    ]
    for pat, canonical in vocab:
        if re.search(pat, focus):
            attrs = [x for x in COLORS + MATERIALS if re.search(rf"\b{x}\b", focus)]
            return canonical, attrs
    return None


def make_anchor(name: str, attrs: list[str] | None = None, required: bool = True) -> dict:
    raw = name.lower().strip(" .,;:")
    qualifiers = []
    for pat, tag in [
        (r"\bdecorated\s+with\s+(.+)$", "decorated_with"),
        (r"\bcovered\s+with\s+(.+)$", "covered_with"),
        (r"\bmounted\s+on\s+(?:the\s+)?(.+)$", "mounted_on"),
        (r"\bwith\s+the\s+(.+)$", "with"),
        (r"\band\s+near\s+the\s+(.+)$", "near"),
        (r"\bnext\s+to\s+(?:the|he)\s+(.+)$", "next_to"),
        (r"\blocated\s+next\s+to\s+the\s+(.+)$", "next_to"),
        (r"\bof\s+the\s+(.+)$", "of"),
    ]:
        m = re.search(pat, raw)
        if m:
            qualifiers.append(f"{tag}:{clean_np(m.group(1))}")
    qualifiers = list(dict.fromkeys(qualifiers))
    name = clean_np(raw)
    attrs = list(attrs or [])
    words = name.split()
    if words and words[0] in COLORS and len(words) > 1:
        attrs.insert(0, words[0])
        name = " ".join(words[1:])
    if name.startswith("built-in "):
        attrs.append("built-in")
        name = name[len("built-in "):]
    return {"class": name, "attributes": attrs, "qualifiers": qualifiers, "required": required,
            "retrieval_keys": retrieval_keys(name, attrs)}


def external_relation(query: str, parent: str) -> tuple[list[dict], dict | None]:
    q = query.lower().strip().rstrip(".")
    patterns = [
        (r"\blocated on the right side of the\s+(.+)$", "right_of", "parent"),
        (r"\blocated on the left side of the\s+(.+)$", "left_of", "parent"),
        (r"\b(?:located\s+)?to the right of the\s+(.+)$", "right_of", "parent"),
        (r"\b(?:located\s+)?to the left of the\s+(.+)$", "left_of", "parent"),
        (r"\b(?:located\s+)?above the\s+(.+)$", "above", "parent"),
        (r"\b(?:located\s+)?below the\s+(.+)$", "below", "parent"),
        (r"\b(?:located\s+)?(?:directly\s+)?under the\s+(.+)$", "below", "parent"),
        (r"\bnext to (?:the|he)\s+(.+)$", "next_to", "parent"),
        (r"\bnear the\s+(.+)$", "near", "parent"),
        (r"\bin front of the\s+(.+)$", "in_front_of", "parent"),
        (r"\bbehind the\s+(.+)$", "behind", "parent"),
        (r"\bon top of the\s+(.+)$", "on_top_of", "parent"),
    ]
    candidates = []
    for pat, pred, subject in patterns:
        m = re.search(pat, q)
        if m:
            candidates.append((m.start(), m, pred, subject))
    if candidates:
        _, m, pred, subject = min(candidates, key=lambda x: x[0])
        raw = m.group(1)
        # Keep the atomic landmark, not its trailing descriptive clause.
        anchor = make_anchor(raw)
        rel = {"subject": subject, "predicate": pred,
               "object": "anchor" if subject == "parent" else "parent"}
        if pred == "left_of" and "above and to the left of" in q:
            rel["qualifiers"] = ["above"]
        return [anchor], rel

    m = re.search(r"\bbetween the\s+(.+?)\s+and the\s+(.+)$", q)
    if m:
        anchors = [make_anchor(m.group(1)), make_anchor(m.group(2))]
        return anchors, {"subject": "parent", "predicate": "between", "object": "anchors"}
    m = re.search(r"\bbetween (?:the\s+)?(?:two\s+)?(doors|beds|mirrors|windows)$", q)
    if m:
        singular = {"doors": "door", "beds": "bed", "mirrors": "mirror", "windows": "window"}[m.group(1)]
        anchor = make_anchor(singular)
        anchor["multiplicity"] = 2
        return [anchor], {"subject": "parent", "predicate": "between", "object": "anchor_group"}

    # "parent with anchor(s) on top".  This is identity evidence even when
    # the instruction names a target part between parent and the with-clause.
    m = re.search(r"\bwith the\s+(.+?)\s+on top$", q)
    if m:
        blob = m.group(1)
        names = [clean_np(x) for x in re.split(r"\s+and\s+the\s+|\s+and\s+", blob)]
        anchors = [make_anchor(x) for x in names if x]
        return anchors, {"subject": "anchor", "predicate": "on_top_of", "object": "parent"}
    # Rare attached identity cue, e.g. a closet door with a built-in mirror.
    m = re.search(r"\bwith the\s+(built-in\s+\w+)$", q)
    if m:
        return [make_anchor(m.group(1))], {"subject": "anchor", "predicate": "attached_to", "object": "parent"}
    # Support-surface identity.  Use the final locative "on the ..." so that
    # "Turn on the TV using the remote on the table" does not confuse the verb.
    host = support_host_class(query)
    if host:
        pos = q.rfind(" on the ")
        anchor = make_anchor(q[pos + len(" on the "):])
        pred = "attached_to" if anchor["class"] == "wall" else "on_top_of"
        return [anchor], {"subject": "parent", "predicate": pred, "object": "anchor"}
    return [], None


def intra_parent_relation(query: str, parent: str, acted: str) -> list[dict]:
    q = query.lower()
    out: list[dict] = []
    part_words = r"(?:drawer|door|window|socket|knob|handle|remote|joystick|chest|mirror\s+light|lamp)"
    # Color and parent nouns may sit between the selector and part, e.g.
    # "bottom blue closet drawer" and "left kitchen counter door".
    gap = r"(?:\w+\s+){0,3}"
    m_from = re.search(r"\b(second|third)\s+from the\s+(left|right)\b", q)
    m_row = re.search(r"\b(second|third)-row\b", q)
    if m_from:
        out.append({"axis": "horizontal", "type": "ordinal",
                    "value": {"second": 2, "third": 3}[m_from.group(1)],
                    "order_from": m_from.group(2)})
    elif m_row:
        out.append({"axis": "vertical", "type": "ordinal",
                    "value": {"second": 2, "third": 3}[m_row.group(1)], "order_from": "top"})
    else:
        m_ord = re.search(rf"\b(first|second|third|fourth|fifth)\s+{gap}{part_words}\b", q)
        if m_ord:
            out.append({"axis": "vertical", "type": "ordinal",
                        "value": {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}[m_ord.group(1)],
                        "order_from": "top"})
    if out and out[-1]["type"] == "ordinal":
        pass
    elif re.search(rf"\bbottom\s+{gap}{part_words}\b", q):
        out.append({"axis": "vertical", "type": "extreme", "value": "bottom"})
    elif re.search(rf"\btop\s+{gap}{part_words}\b", q):
        out.append({"axis": "vertical", "type": "extreme", "value": "top"})
    elif re.search(rf"\bmiddle\s+{gap}{part_words}\b", q):
        out.append({"axis": "vertical", "type": "rank", "value": "middle"})

    # Direction closest to the acted part is intra-parent.  Phrases of the
    # form "to the left/right of" were already reserved for external edges.
    for side in ("left", "right"):
        if m_from:
            break
        if re.search(rf"\b{side}most\s+{gap}{part_words}\b", q) or re.search(rf"\b{side}\s+{gap}{part_words}\b", q):
            out.append({"axis": "horizontal", "type": "extreme", "value": side})
            break
    return out


def parse_contract(query: str, acted: str, hierarchy: list[str]) -> dict:
    parent = parent_from_cot(query, acted, hierarchy)
    attrs = parent_attributes(query, parent)
    anchors, edge = external_relation(query, parent)
    host = relation_host_class(query)
    if edge and host:
        parent, attrs = host
    target = clean_np(acted) or "interactive part"
    confidence = 1.0
    reasons = []
    if not parent:
        confidence -= 0.5; reasons.append("missing_parent")
    if edge and not anchors:
        confidence -= 0.5; reasons.append("relation_without_anchor")
    return {
        "parent": {"class": parent, "attributes": attrs,
                   "retrieval_keys": retrieval_keys(parent, attrs)},
        "anchors": anchors,
        "external_relation": edge,
        "target_part": {"class": target},
        "intra_parent_relations": intra_parent_relation(query, parent, target),
        "requires_instance_resolution": bool(edge and anchors),
        "contract_confidence": max(0.0, confidence),
        "abstain_reasons": reasons,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only-hard14", action="store_true")
    ap.add_argument("--data-root", type=Path, default=ROOT.parent / "data/scenefun3d")
    ap.add_argument("--split", default="val")
    ap.add_argument("--llm-type", default="llama_v2")
    args = ap.parse_args()
    parser = make_parser(args.split, str(args.data_root))
    hard_ids = set()
    gold_path = ROOT / "exps/task_router/b1_rel_prompt_contracts.json"
    if args.only_hard14:
        hard_ids = {x["desc_id"] for x in json.loads(gold_path.read_text())["queries"]}
    visits = sorted(fio.get_visit_to_videos(str(args.data_root), args.split))
    rows = []
    for visit in visits:
        descs = parser.get_descriptions(visit)
        llms = parser.get_llm_data(visit, args.llm_type)
        for d, llm in zip(descs, llms):
            if hard_ids and d["desc_id"] not in hard_ids:
                continue
            query = str(d.get("description") or llm.get("prompt") or "").strip()
            acted = str(llm.get("acted_on_object") or "")
            hierarchy = list(llm.get("acted_on_object_hierarchy") or [])
            rows.append({"visit": visit, "desc_id": d["desc_id"], "query": query,
                         "source": "deterministic_query+llama_v2_hierarchy",
                         **parse_contract(query, acted, hierarchy)})
    report = {
        "schema_version": 2,
        "protocol": "Atomic parent/anchor relation contract; deterministic parser, no new model, no OWL/GT input.",
        "n": len(rows),
        "n_requires_instance_resolution": sum(x["requires_instance_resolution"] for x in rows),
        "queries": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {args.out} n={len(rows)} instance={report['n_requires_instance_resolution']}")


if __name__ == "__main__":
    main()
