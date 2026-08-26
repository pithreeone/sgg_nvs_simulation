"""
vlm.py -- Qwen2.5-VL turned into one bit: step LEFT or step RIGHT.

THE MODEL MAKES THE CALL.  An earlier version asked it only for two boxes and
computed the side here from their centres; that scored well but the direction
was arithmetic, not a decision, so the arm could not be described as a VLM
driving the robot.  It now answers the direction itself.

THE WORDING IS NOT A DETAIL, it is most of the result.  Measured on the 40
frames of `cases_slot_v2`, ground truth 28 LEFT / 12 RIGHT so a constant scores
28/40:

    Your task: Find the {A} behind the {B}. You may take ONE SMALL
    sideways step ... Which way should you step?     RIGHT or LEFT   34/40
                                                     LEFT or RIGHT   28/40
    ... Which way should you step?  (no step described)   RL          33/40
    ... Part of the {A} is already visible past one edge  RL          33/40
    ... decide which way ... so you can SEE IT BETTER     RL          28/40

Four things were measured and each one matters:

  OPTION ORDER.  Every phrasing tried answers LEFT 40 times out of 40 when the
  options are given as "LEFT or RIGHT".  The same phrasing with "RIGHT or LEFT"
  ventures RIGHT eight times and is right seven of them.  So LR measures the
  model's prior and RL measures the picture.  Do not tidy this back.

  NAMING THE GOAL BREAKS IT.  Any wording containing "see it better" or "get a
  better view" returns to the constant, three times out of three.  The bare
  "Which way should you step?" is what makes it look at the image.

  DO NOT SAY WHICH OBJECT OCCLUDES.  Opening with "the {B} is blocking your view
  of the {A}" also returns the constant.  The instruction alone is enough.

  ONE TURN.  Grounding first (boxes in context) drops it to 29/40; asking it
  first which side the target shows past -- which it answers 39/40 -- and then
  for the move drops 7B to 1/40, the exact inverse of its own correct answer,
  and 3B to chance.  Every added step returns the move to the default LEFT.

WHAT THE ARM IS NOT.  It is conservative rather than able: 32 of 40 answers are
LEFT.  What it has is precision on the minority answer, not coverage.  Asking
for a step SIZE as well collapses both -- the size is one constant value and the
direction falls to 12/40 -- so the magnitude stays fixed at `--side-step`.
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

#: The measured wording; see the module docstring for what each clause buys and
#: what removing it costs.  `RIGHT or LEFT` is load-bearing.
PROMPT = ("Your task: Find the {subject} behind the {object}. You may take ONE "
          "SMALL sideways step of about 30 cm, keeping the camera aimed at the "
          "same spot. Which way should you step? Answer with one word, RIGHT "
          "or LEFT.")

#: 7B, not 3B: on the same 40 frames 3B answers LEFT 40 times out of 40 to every
#: phrasing tried.  4-bit NF4 keeps it near 5 GB, which is what lets it sit on
#: the card beside THOR and EGTR.
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


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
        """`"LEFT"`, `"RIGHT"`, or None if the reply contained neither."""
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
            # GREEDY, and short: the answer is one word and sampling would make
            # a rerun of the same case walk somewhere else.
            out = self.model.generate(**inputs, max_new_tokens=8,
                                      do_sample=False)
        reply = self.processor.decode(out[0][inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True).strip()
        hit = re.search(r"\b(LEFT|RIGHT)\b", reply.upper())
        said = hit.group(1) if hit else None
        self.last = {"said": said, "reply": reply}
        return said
