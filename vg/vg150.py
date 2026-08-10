"""
vg150.py -- VG150 vocabulary and its mapping onto robot actions / AI2-THOR.

This is the single place where "what EGTR can say" is defined.  The decision
tree consumes only this vocabulary, so the mock scene graphs you hand-write are
guaranteed to be expressible by the real model later on.

Three things live here:
  1. The VG150 vocabulary itself (150 object classes, 50 predicates).
  2. PREDICATE_FAMILY: every one of the 50 predicates -> an occlusion family,
     which the decision tree turns into an action.
  3. THOR_TO_VG150 / VG150_TO_THOR: the simulator speaks CamelCase objectTypes,
     EGTR speaks VG150 class names.  The scene-graph layer uses VG150 names and
     the controller translates.

!! IMPORTANT -- validate the vocabulary against your own checkpoint !!
The OBJECT_CLASSES / PREDICATES constants below are a best-effort transcription
and MUST be checked against the dictionary file your EGTR fork trains on
(`VG-SGG-dicts.json`, keys `idx_to_label` and `idx_to_predicate`).  Call
load_vocab() with that path and it will replace the constants and report any
mismatch.  Do not trust the hardcoded lists for anything you publish.

Index convention: in the standard VG-SGG format index 0 is the background /
no-relation class and real classes are 1-indexed.  decode_* helpers below take
`background_offset=1` accordingly -- confirm this matches your fork.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1. Vocabulary
# ---------------------------------------------------------------------------

#: The 50 VG150 predicates, alphabetically ordered as in VG-SGG-dicts.json.
PREDICATES: Tuple[str, ...] = (
    "above", "across", "against", "along", "and", "at", "attached to",
    "behind", "belonging to", "between", "carrying", "covered in", "covering",
    "eating", "flying in", "for", "from", "growing on", "hanging from", "has",
    "holding", "in", "in front of", "laying on", "looking at", "lying on",
    "made of", "mounted on", "near", "of", "on", "on back of", "over",
    "painted on", "parked on", "part of", "playing", "riding", "says",
    "sitting on", "standing on", "to", "under", "using", "walking in",
    "walking on", "watching", "wearing", "wears", "with",
)

#: The 150 VG150 object classes, in label order (1-indexed as in
#: VG-SGG-dicts.json: OBJECT_CLASSES[0] == "airplane" == label index 1).
#:
#: Confirmed against the project's own dictionary.  Note the shape of this
#: vocabulary, because it drives the whole ontology problem below: roughly a
#: third of it is body parts ("arm", "ear", "eye", "face", "finger", "hair",
#: "hand", "head", "leg", "mouth", "neck", "nose", "paw", "tail"), people
#: variants ("boy", "child", "girl", "guy", "kid", "lady", "man", "men",
#: "people", "person", "player", "woman"), outdoor scenery and vehicles.  Very
#: little of it is indoor household objects -- see UNMAPPED_THOR_TYPES.
OBJECT_CLASSES: Tuple[str, ...] = (
    "airplane", "animal", "arm", "bag", "banana", "basket", "beach", "bear",
    "bed", "bench", "bike", "bird", "board", "boat", "book", "boot", "bottle",
    "bowl", "box", "boy", "branch", "building", "bus", "cabinet", "cap", "car",
    "cat", "chair", "child", "clock", "coat", "counter", "cow", "cup",
    "curtain", "desk", "dog", "door", "drawer", "ear", "elephant", "engine",
    "eye", "face", "fence", "finger", "flag", "flower", "food", "fork",
    "fruit", "giraffe", "girl", "glass", "glove", "guy", "hair", "hand",
    "handle", "hat", "head", "helmet", "hill", "horse", "house", "jacket",
    "jean", "kid", "kite", "lady", "lamp", "laptop", "leaf", "leg", "letter",
    "light", "logo", "man", "men", "motorcycle", "mountain", "mouth", "neck",
    "nose", "number", "orange", "pant", "paper", "paw", "people", "person",
    "phone", "pillow", "pizza", "plane", "plant", "plate", "player", "pole",
    "post", "pot", "racket", "railing", "rock", "roof", "room", "screen",
    "seat", "sheep", "shelf", "shirt", "shoe", "short", "sidewalk", "sign",
    "sink", "skateboard", "ski", "skier", "sneaker", "snow", "sock", "stand",
    "street", "surfboard", "table", "tail", "tie", "tile", "tire", "toilet",
    "towel", "tower", "track", "train", "tree", "truck", "trunk", "umbrella",
    "vase", "vegetable", "vehicle", "wave", "wheel", "window", "windshield",
    "wing", "wire", "woman", "zebra",
)


def load_vocab(dicts_path: str, apply_globally: bool = True) -> Dict[str, Any]:
    """
    Load the authoritative vocabulary from VG-SGG-dicts.json.

    Run this once on the server before trusting anything in this module::

        from vg.vg150 import load_vocab
        info = load_vocab("data/visual_genome/VG-SGG-dicts.json")
        print(info["warnings"])

    Returns a dict with the loaded tuples, counts, and any warnings (size
    mismatch, or predicates missing from PREDICATE_FAMILY).
    """
    global OBJECT_CLASSES, PREDICATES

    with open(dicts_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    def ordered(mapping: Dict[str, str]) -> Tuple[str, ...]:
        # idx_to_* keys are stringified ints, 1-indexed (0 == background).
        return tuple(mapping[k] for k in sorted(mapping, key=lambda s: int(s)))

    classes = ordered(data["idx_to_label"])
    predicates = ordered(data["idx_to_predicate"])

    warnings: List[str] = []
    if len(classes) != 150:
        warnings.append(f"expected 150 object classes, file has {len(classes)}")
    if len(predicates) != 50:
        warnings.append(f"expected 50 predicates, file has {len(predicates)}")

    missing = [p for p in predicates if p not in PREDICATE_FAMILY]
    if missing:
        warnings.append(
            f"{len(missing)} predicate(s) have no family mapping: {missing}"
        )
    unknown = [p for p in PREDICATE_FAMILY if p not in predicates]
    if unknown:
        warnings.append(
            f"{len(unknown)} mapped predicate(s) are not in the file: {unknown}"
        )
    dropped = [c for c in OBJECT_CLASSES if c not in classes]
    if dropped:
        warnings.append(f"hardcoded classes not in file: {dropped}")

    if apply_globally:
        OBJECT_CLASSES = classes
        PREDICATES = predicates

    return {
        "object_classes": classes,
        "predicates": predicates,
        "n_classes": len(classes),
        "n_predicates": len(predicates),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 2. Predicate -> occlusion family
# ---------------------------------------------------------------------------


class RelationFamily(str, Enum):
    """
    Why the target is hard to see, which is what determines the manoeuvre.

    Defined here rather than in decision_tree.py because the partition is a
    property of the relation vocabulary, not of the robot.
    """

    HORIZONTAL_OCCLUSION = "HORIZONTAL_OCCLUSION"  # blocked left/right
    VERTICAL_OCCLUSION = "VERTICAL_OCCLUSION"      # blocked up/down
    GAP = "GAP"                                    # visible only through a slot
    ATTACHMENT = "ATTACHMENT"                      # rides on a mobile host
    PART_OF = "PART_OF"                            # a part of a larger object
    CONTAINMENT = "CONTAINMENT"                    # inside something openable
    PROXIMITY = "PROXIMITY"                        # only a location prior
    NON_SPATIAL = "NON_SPATIAL"                    # no geometric information
    UNKNOWN = "UNKNOWN"                            # target not in the graph


_H = RelationFamily.HORIZONTAL_OCCLUSION
_V = RelationFamily.VERTICAL_OCCLUSION
_G = RelationFamily.GAP
_A = RelationFamily.ATTACHMENT
_P = RelationFamily.PART_OF
_C = RelationFamily.CONTAINMENT
_X = RelationFamily.PROXIMITY
_N = RelationFamily.NON_SPATIAL

#: predicate -> (family, pitch_hint).  pitch_hint is only read for _V:
#: "down" = target sits below the host, "up" = target is above the robot.
#:
#: All 50 predicates are covered.  36 carry usable geometry; the 14 mapped to
#: NON_SPATIAL do not constrain a viewpoint at all.
PREDICATE_FAMILY: Dict[str, Tuple[RelationFamily, Optional[str]]] = {
    # --- horizontal occlusion (2) -> orbit -----------------------------------
    "behind": (_H, None),
    "in front of": (_H, None),
    # --- vertical occlusion (13) -> approach + pitch -------------------------
    "under": (_V, "down"),
    "on": (_V, "down"),           # the highest-frequency VG predicate
    "laying on": (_V, "down"),
    "lying on": (_V, "down"),
    "sitting on": (_V, "down"),
    "standing on": (_V, "down"),
    "parked on": (_V, "down"),
    "growing on": (_V, "down"),
    "above": (_V, "up"),
    "over": (_V, "up"),
    "hanging from": (_V, "up"),
    "mounted on": (_V, "up"),
    "attached to": (_V, "up"),
    # --- gap (1) -> align to the gap normal ---------------------------------
    "between": (_G, None),
    # --- attachment (8) -> track the host -----------------------------------
    "holding": (_A, None),
    "carrying": (_A, None),
    "wearing": (_A, None),
    "wears": (_A, None),
    "riding": (_A, None),
    "using": (_A, None),
    "playing": (_A, None),
    "eating": (_A, None),
    # --- part-of (5) -> go to the whole object ------------------------------
    "of": (_P, None),
    "part of": (_P, None),
    "belonging to": (_P, None),
    "has": (_P, None),
    "on back of": (_P, None),
    # --- containment (1) -> open and inspect --------------------------------
    "in": (_C, None),
    # --- proximity (6) -> location prior only -------------------------------
    "near": (_X, None),
    "with": (_X, None),
    "at": (_X, None),
    "against": (_X, None),
    "along": (_X, None),
    "across": (_X, None),
    # --- no geometry (14) -> fall back to exploration -----------------------
    # "covering"/"covered in" DO imply occlusion, but by a draped surface that
    # no viewpoint change can defeat -- it needs manipulation, out of scope.
    "covering": (_N, None),
    "covered in": (_N, None),
    "and": (_N, None),
    "for": (_N, None),
    "from": (_N, None),
    "to": (_N, None),
    "made of": (_N, None),
    "painted on": (_N, None),
    "says": (_N, None),
    "looking at": (_N, None),
    "watching": (_N, None),
    "flying in": (_N, None),
    "walking in": (_N, None),
    "walking on": (_N, None),
}

#: Symmetric predicates: the family is unchanged when subject/object swap.
SYMMETRIC_PREDICATES: Tuple[str, ...] = (
    "near", "with", "at", "against", "along", "across", "and",
)

#: What a relation means when the TARGET is in the object slot.
#: EGTR emits ("person", "holding", "cup"); to search for the cup we must read
#: that edge backwards.  Maps predicate -> (label, family, pitch_hint).
#: The labels are internal descriptions, NOT VG150 predicates -- most VG150
#: predicates have no passive form in the vocabulary.
INVERSE_SPEC: Dict[str, Tuple[str, RelationFamily, Optional[str]]] = {
    "behind": ("in front of", _H, None),
    "in front of": ("behind", _H, None),
    # target is the thing being sat/stood/laid ON -> it is the supporter, which
    # is normally large and easy to see, so only a location prior remains.
    "on": ("supports", _X, None),
    "laying on": ("supports", _X, None),
    "lying on": ("supports", _X, None),
    "sitting on": ("supports", _X, None),
    "standing on": ("supports", _X, None),
    "parked on": ("supports", _X, None),
    "growing on": ("supports", _X, None),
    "under": ("above", _V, "up"),
    "above": ("under", _V, "down"),
    "over": ("under", _V, "down"),
    "attached to": ("has attached", _X, None),
    "hanging from": ("has hanging", _X, None),
    "mounted on": ("has mounted", _X, None),
    "in": ("contains", _X, None),
    "holding": ("held by", _A, None),
    "carrying": ("carried by", _A, None),
    "wearing": ("worn by", _A, None),
    "wears": ("worn by", _A, None),
    "riding": ("ridden by", _A, None),
    "using": ("used by", _A, None),
    "playing": ("played by", _A, None),
    "eating": ("eaten by", _A, None),
    "has": ("part of", _P, None),
    "of": ("has", _X, None),
    "part of": ("has", _X, None),
    "belonging to": ("owns", _X, None),
    "on back of": ("has on back", _X, None),
    # "between" cannot be inverted: knowing X is between the target and Y does
    # not tell us how the target is occluded.
}


def normalise_predicate(predicate: str) -> str:
    """Canonicalise a predicate string to VG150 spelling (spaces, lowercase)."""
    return str(predicate).strip().lower().replace("_", " ").replace("-", " ")


def normalise_class(name: str) -> str:
    """Canonicalise an object class name for comparison."""
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def family_of(predicate: str) -> RelationFamily:
    """Family of a VG150 predicate, or UNKNOWN if outside the vocabulary."""
    spec = PREDICATE_FAMILY.get(normalise_predicate(predicate))
    return spec[0] if spec else RelationFamily.UNKNOWN


def actionable_predicates() -> Tuple[str, ...]:
    """The predicates that constrain a viewpoint (i.e. not NON_SPATIAL)."""
    return tuple(
        p for p, (family, _) in PREDICATE_FAMILY.items() if family is not _N
    )


# ---------------------------------------------------------------------------
# 3. AI2-THOR <-> VG150 object classes
# ---------------------------------------------------------------------------

#: THOR objectType -> VG150 class.  Entries marked LOOSE are semantically
#: approximate; VG150 simply has no better class.  Treat them with suspicion in
#: any quantitative evaluation.
THOR_TO_VG150: Dict[str, str] = {
    # --- exact ------------------------------------------------------------
    "Bed": "bed",
    "Book": "book",
    "Bottle": "bottle",
    "Bowl": "bowl",
    "Box": "box",
    "Cabinet": "cabinet",
    "Chair": "chair",
    "CounterTop": "counter",
    "Cup": "cup",
    "Curtains": "curtain",
    "Desk": "desk",
    "Door": "door",
    "Doorway": "door",
    "Drawer": "drawer",
    "Fork": "fork",
    "HousePlant": "plant",
    "Laptop": "laptop",
    "Pillow": "pillow",
    "Plate": "plate",
    "Pot": "pot",
    "Shelf": "shelf",
    "ShelvingUnit": "shelf",
    "Sink": "sink",
    "SinkBasin": "sink",
    "TennisRacket": "racket",
    "Toilet": "toilet",
    "Towel": "towel",
    "Vase": "vase",
    "Window": "window",
    "Boots": "boot",
    "TVStand": "stand",
    "CellPhone": "phone",
    # --- one-to-many collapses (several THOR types share one VG150 class) ---
    "Mug": "cup",                   # with Cup
    "WineBottle": "bottle",         # with Bottle
    "ArmChair": "chair",            # with Chair
    "DiningTable": "table",         # with CoffeeTable, SideTable
    "CoffeeTable": "table",
    "SideTable": "table",
    "HandTowel": "towel",
    "ShowerCurtain": "curtain",
    "Blinds": "curtain",
    "TissueBox": "box",
    "DeskLamp": "lamp",             # with FloorLamp, metres apart in a scene
    "FloorLamp": "lamp",
    # --- LOOSE: nearest available VG150 class, semantically approximate -----
    #: VG150 has no sofa/couch, and `seat` looked like the nearest word.  It is
    #: not the one used for these: measured over 200 reference views, `seat` was
    #: the model's label for 0 of 29 Sofas and 0 of 12 Stools, while `chair` took
    #: 15 and 5.  The word is not idle either -- it is predicted 234 times, but on
    #: Chairs (29) and ArmChairs (14), i.e. `seat` names a chair's seat, not a
    #: piece of seating furniture.  Under the old mapping GT `seat` scored 0.0%
    #: agreement across all 41 instances.
    "Sofa": "chair",
    "Stool": "chair",
    "Footstool": "chair",
    "Pan": "pot",
    "SoapBottle": "bottle",
    "SprayBottle": "bottle",
    "Newspaper": "paper",
    "PaperTowelRoll": "paper",
    "ToiletPaper": "paper",
    "Television": "screen",         # VG150 has no tv
    "LightSwitch": "light",
    "AlarmClock": "clock",
    "Watch": "clock",
    #: `basket` took 7 of 57 GarbageCans against `box` 16 and `bag` 9.  Changed on
    #: that margin only; `Curtains` and `Lettuce` showed a larger apparent margin
    #: (`window` 12 v `curtain` 4, `fruit` 5 v `vegetable` 0) and were deliberately
    #: NOT changed -- those are the model confusing a curtain with the window
    #: behind it and misnaming a lettuce, not VG using a different word.  Sibling
    #: mappings confirm it: `Blinds`->curtain hits 21 and `ShowerCurtain`->curtain
    #: hits 6.  The mapping table follows VG's vocabulary, not the model's errors.
    "GarbageCan": "box",
    "Lettuce": "vegetable",
    "Tomato": "vegetable",
    "Potato": "vegetable",
    "Bread": "food",
    "Egg": "food",
    "Apple": "fruit",
    "TeddyBear": "animal",
    "ShowerDoor": "door",
}

#: THOR objectTypes with NO defensible VG150 class.
#:
#: This list is the concrete argument for retraining EGTR's object head on a
#: household vocabulary.  VG150 spends roughly a third of its 150 slots on body
#: parts, people variants and outdoor scenery, so an indoor scene is mostly made
#: of things it cannot name: no microwave, no fridge, no toaster, no spoon, no
#: knife, no mirror -- and no key/keychain, which makes the canonical
#: "Find the keys" instruction inexpressible in this vocabulary.
UNMAPPED_THOR_TYPES: Tuple[str, ...] = (
    # kitchen appliances and cutlery -- the bulk of a kitchen
    "Microwave", "Toaster", "Fridge", "CoffeeMachine", "Kettle", "StoveBurner",
    "StoveKnob", "Faucet", "Spoon", "Knife", "ButterKnife", "Spatula", "Ladle",
    "SaltShaker", "PepperShaker", "PepperMill", "DishSponge", "AluminumFoil",
    # the target of the canonical instruction
    "KeyChain",
    # other indoor objects VG150 misses
    "Mirror", "CreditCard", "RemoteControl", "Painting", "Statue",
    "Poster", "Floor", "Bathtub", "BathtubBasin", "Safe", "Candle",
    "SoapBar", "Pen", "Pencil", "Plunger", "ScrubBrush", "GarbageBag",
    "VacuumCleaner", "ShowerHead", "ShowerGlass", "TowelHolder",
    "HandTowelHolder", "ToiletPaperHanger", "Chandelier", "Cloth",
    "BaseballBat", "BasketBall", "Dumbbell", "CD", "Keyboard", "Mouse",
)


def _build_reverse() -> Dict[str, Tuple[str, ...]]:
    reverse: Dict[str, List[str]] = {}
    for thor_type, vg_class in THOR_TO_VG150.items():
        reverse.setdefault(vg_class, []).append(thor_type)
    return {k: tuple(v) for k, v in reverse.items()}


#: VG150 class -> every THOR objectType that maps to it.  One-to-many: VG150
#: "cup" covers both THOR Mug and THOR Cup, so a search for "cup" must consider
#: both instances.
VG150_TO_THOR: Dict[str, Tuple[str, ...]] = _build_reverse()


def thor_to_vg150(object_type: str) -> Optional[str]:
    """VG150 class for a THOR objectType, or None if it has no equivalent."""
    return THOR_TO_VG150.get(object_type)


def vg150_to_thor(vg_class: str) -> Tuple[str, ...]:
    """Every THOR objectType matching a VG150 class (possibly empty)."""
    return VG150_TO_THOR.get(normalise_class(vg_class), ())


def coverage_report(thor_object_types: Iterable[str]) -> Dict[str, Any]:
    """
    How much of a scene EGTR could even name.

    Pass event.metadata["objects"] objectTypes; returns mapped / unmapped lists
    and the coverage fraction -- a cheap sanity metric to log per scene.
    """
    types = sorted(set(thor_object_types))
    mapped = {t: THOR_TO_VG150[t] for t in types if t in THOR_TO_VG150}
    unmapped = [t for t in types if t not in THOR_TO_VG150]
    return {
        "n_types": len(types),
        "mapped": mapped,
        "unmapped": unmapped,
        "coverage": len(mapped) / len(types) if types else 0.0,
    }


# ---------------------------------------------------------------------------
# 4. Validation of hand-written mock graphs
# ---------------------------------------------------------------------------


def validate_scene_graph(
    scene_graph: Sequence[Sequence],
    strict_classes: bool = True,
) -> List[str]:
    """
    Check a mock scene graph is expressible by an EGTR/VG150 model.

    Returns a list of human-readable problems (empty == valid).  Use this on
    every hand-written graph so you never validate the decision layer against
    triplets the real model could not produce.

    Args:
        strict_classes: also require subject/object names to be VG150 classes.
            Set False while prototyping with THOR objectTypes.
    """
    problems: List[str] = []

    for index, triplet in enumerate(scene_graph or ()):
        if not isinstance(triplet, (list, tuple)) or len(triplet) not in (3, 4):
            problems.append(f"[{index}] not a 3- or 4-element triplet: {triplet!r}")
            continue

        subject, predicate, obj = triplet[0], triplet[1], triplet[2]

        predicate_key = normalise_predicate(predicate)
        if predicate_key not in PREDICATE_FAMILY:
            problems.append(
                f"[{index}] '{predicate}' is not a VG150 predicate"
            )
        elif predicate_key not in {normalise_predicate(p) for p in PREDICATES}:
            problems.append(
                f"[{index}] '{predicate}' is mapped but absent from PREDICATES"
            )

        if len(triplet) == 4:
            score = triplet[3]
            if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
                problems.append(f"[{index}] confidence must be in [0, 1]: {score!r}")

        if not strict_classes:
            continue

        names: List[str] = [subject]
        names.extend(obj if isinstance(obj, (list, tuple)) else [obj])
        known = {normalise_class(c) for c in OBJECT_CLASSES}
        for name in names:
            if normalise_class(name) not in known:
                hint = ""
                if str(name) in THOR_TO_VG150:
                    hint = f" (did you mean VG150 '{THOR_TO_VG150[str(name)]}'?)"
                elif str(name) in UNMAPPED_THOR_TYPES:
                    hint = " (THOR type with NO VG150 equivalent)"
                problems.append(
                    f"[{index}] '{name}' is not a VG150 object class{hint}"
                )

    return problems


# ---------------------------------------------------------------------------
# 5. EGTR output -> triplet list
# ---------------------------------------------------------------------------


def triplets_from_dense(
    object_labels: Sequence[int],
    relation_scores: Any,
    threshold: float = 0.3,
    top_k: Optional[int] = 100,
    background_offset: int = 1,
    connectivity: Any = None,
) -> List[Tuple[str, str, str, float]]:
    """
    Convert a dense relation tensor into (subject, predicate, object, score).

    ADAPT THIS TO YOUR FORK -- the assumed shapes are:
        object_labels   : (N,)        class index per object query
        relation_scores : (N, N, 50)  P(predicate | subject, object)
        connectivity    : (N, N)      optional P(any relation), multiplied in

    Thresholding rather than taking a fixed top-K is the point: the decision
    tree can then reason over confidence, and low-confidence edges become
    fallback branches in decide_plan() instead of being silently discarded.
    """
    import numpy as np

    scores = np.asarray(relation_scores, dtype=float)
    if connectivity is not None:
        scores = scores * np.asarray(connectivity, dtype=float)[:, :, None]

    n_objects = len(object_labels)

    def class_name(index: int) -> str:
        real = int(index) - background_offset
        return (
            OBJECT_CLASSES[real]
            if 0 <= real < len(OBJECT_CLASSES)
            else f"<class {index}>"
        )

    triplets: List[Tuple[str, str, str, float]] = []
    for subject in range(n_objects):
        for obj in range(n_objects):
            if subject == obj:
                continue
            predicate_index = int(np.argmax(scores[subject, obj]))
            score = float(scores[subject, obj, predicate_index])
            if score < threshold or predicate_index >= len(PREDICATES):
                continue
            triplets.append(
                (
                    class_name(object_labels[subject]),
                    PREDICATES[predicate_index],
                    class_name(object_labels[obj]),
                    score,
                )
            )

    triplets.sort(key=lambda t: t[3], reverse=True)
    return triplets[:top_k] if top_k else triplets


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"predicates      : {len(PREDICATES)}")
    print(f"object classes  : {len(OBJECT_CLASSES)}")
    print(f"family coverage : {len(PREDICATE_FAMILY)}/{len(PREDICATES)}")

    missing = [p for p in PREDICATES if p not in PREDICATE_FAMILY]
    extra = [p for p in PREDICATE_FAMILY if p not in PREDICATES]
    print(f"unmapped        : {missing or 'none'}")
    print(f"not in VG150    : {extra or 'none'}")

    counts: Dict[str, int] = {}
    for family, _ in PREDICATE_FAMILY.values():
        counts[family.value] = counts.get(family.value, 0) + 1
    print("\npredicates per family:")
    for family, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {family:<22}{count}")
    print(f"\nactionable      : {len(actionable_predicates())}/{len(PREDICATES)}")
    print(f"THOR types mapped: {len(THOR_TO_VG150)}, "
          f"known-unmappable: {len(UNMAPPED_THOR_TYPES)}")

    if os.environ.get("VG_DICTS"):
        info = load_vocab(os.environ["VG_DICTS"])
        print(f"\nloaded from {os.environ['VG_DICTS']}: "
              f"{info['n_classes']} classes, {info['n_predicates']} predicates")
        for warning in info["warnings"]:
            print(f"  WARNING: {warning}")
