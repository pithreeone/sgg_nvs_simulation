"""
vlm.py -- Qwen2.5-VL turned into one bit: step LEFT or step RIGHT.

WHY IT IS NOT ASKED THE QUESTION DIRECTLY.  It was, twice, on the 40 cases of
`cases_slot`: once as "which way should you step", once forced to first name
which edge of the occluder the target peeks past.  Both phrasings answered LEFT
40 times out of 40, which scores 22/40 -- exactly what always saying the
majority scores, and exactly what a constant scores.  It answers LEFT even to
"is the box in the left half or the right half of this image".  Its left/right
JUDGEMENT is a constant; its PERCEPTION is not, since one-sentence descriptions
of those same frames differ from each other and are correct.

So it is asked only for what it can do -- two boxes -- and the side is
arithmetic here: the target peeks out on the side its box centre lies relative
to the occluder's, and stepping that way opens the sightline.  Measured that
way, 38/40.

THAT SPLIT IS THE POINT AND THE LIMITATION.  This arm's direction is not the
model's decision, it is ours over the model's detections, which is a weaker
claim than "a VLM can drive the robot".  The two failures are both cases whose
occluder asset is `bin_3`, a green recycling bin, while the instruction says
"box" -- VG150 files a bin under `box` and the generator screens on EGTR's
class, not on the word.  Asked to outline "the box" in a picture with no box,
the model outlines something else.  35/35 where the noun matches the asset,
3/5 where it does not.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np

#: The commit each model was downloaded at.  Pinned rather than `main`, and
#: keyed by model, because a revision belongs to one repository: carrying 3B's
#: sha to the 7B repo asks the hub for a commit it has never heard of.  Same
#: values as `../ros2/tools/chat_qwen.py`.
REVISIONS = {
    "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
}

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

#: Qwen's own grounding format.  BOTH OBJECTS IN ONE CALL, so the two boxes come
#: back in the same coordinate space -- a left-of test is then scale invariant
#: and the model's resize never has to be undone.
PROMPT = ('Outline the position of the {subject} and of the {object} in this '
          'image. Output the coordinates in JSON format as a list of '
          '{{"bbox_2d": [x1, y1, x2, y2], "label": "..."}} entries, one per '
          'object, and nothing else.')


def _boxes_from(reply: str) -> Dict[str, List[float]]:
    """-> {label: [x1, y1, x2, y2]}.  Tolerates ```json fences and stray prose."""
    body = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.M).strip()
    hit = re.search(r"\[.*\]", body, re.S)
    if not hit:
        return {}
    try:
        items = json.loads(hit.group(0))
    except json.JSONDecodeError:
        return {}
    out: Dict[str, List[float]] = {}
    for item in items if isinstance(items, list) else []:
        box, label = item.get("bbox_2d"), str(item.get("label", "")).lower()
        if isinstance(box, list) and len(box) == 4:
            out.setdefault(label, [float(v) for v in box])
    return out


def _named(found: Dict[str, List[float]], want: str) -> Optional[List[float]]:
    """The box whose label names `want`."""
    for label, box in found.items():
        if want in label or label in want:
            return box
    return None


class Director:
    """The model, loaded once, and one call per decision."""

    def __init__(self, model: str = MODEL, bits: int = 4,
                 max_patches: int = 768, device: str = "cuda") -> None:
        import torch
        from transformers import (AutoProcessor,
                                  Qwen2_5_VLForConditionalGeneration)

        revision = REVISIONS.get(model, "main")
        self.processor = AutoProcessor.from_pretrained(
            model, revision=revision, min_pixels=256 * 28 * 28,
            max_pixels=max_patches * 28 * 28)
        quant = None
        if bits in (4, 8):
            # 4-BIT IS NOT OPTIONAL ON AN 8 GB CARD.  bf16 is 7.5 GB of weights,
            # and THOR and EGTR are already on the card when this loads.  The
            # vision tower is left unquantised: measured on the same photo,
            # quantising it costs visible detail, and detail is all this arm
            # asks the model for.
            from transformers import BitsAndBytesConfig

            skip = ["visual", "lm_head"]
            quant = (BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                        bnb_4bit_use_double_quant=True,
                                        llm_int8_skip_modules=skip)
                     if bits == 4
                     else BitsAndBytesConfig(load_in_8bit=True,
                                             llm_int8_skip_modules=skip))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, revision=revision, device_map=device,
            dtype=torch.bfloat16, quantization_config=quant).eval()
        self.name, self.bits = model, bits
        self.last: Dict[str, Any] = {}

    def direction(self, frame: np.ndarray, subject: str, obj: str
                  ) -> Optional[str]:
        """`"LEFT"`, `"RIGHT"`, or None when either box is missing."""
        import torch
        from PIL import Image

        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text",
             "text": PROMPT.format(subject=subject, object=obj)}]}]
        text = self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image],
                                return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=256,
                                      do_sample=False)
        reply = self.processor.decode(out[0][inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True).strip()

        found = _boxes_from(reply)
        target, occluder = _named(found, subject), _named(found, obj)
        said = None
        if target and occluder:
            # CENTRES, NOT EDGES: an occluder box that swallows the target's
            # would make an edge test read the occluder's own extent.
            said = ("RIGHT" if (target[0] + target[2]) / 2
                    > (occluder[0] + occluder[2]) / 2 else "LEFT")
        self.last = {"said": said, "target_box": target,
                     "occluder_box": occluder, "labels": sorted(found),
                     "reply": reply}
        return said
