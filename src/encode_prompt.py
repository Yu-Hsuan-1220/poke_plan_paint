"""Cache the fixed UMT5 prompt embeddings used for every clip (Koala captions are not used yet).

    python src/encode_prompt.py   -> data/prompt_embeds.pt {"prompt", "embeds" [1,512,4096], "negative", "neg_embeds"}
"""

import os

import torch

from common import ROOT, encode_prompt

PROMPT = "A realistic video shot with a static camera. The objects in the scene move naturally."
# Wan2.2's default negative prompt
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
            "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
            "三条腿，背景人很多，倒着走")

if __name__ == "__main__":
    os.chdir(ROOT)
    e = encode_prompt([PROMPT, NEGATIVE])
    torch.save({"prompt": PROMPT, "embeds": e[:1].bfloat16(), "negative": NEGATIVE, "neg_embeds": e[1:].bfloat16()},
               "data/prompt_embeds.pt")
    print("saved data/prompt_embeds.pt", e.shape)
