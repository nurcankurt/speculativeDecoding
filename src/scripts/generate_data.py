# scripts/generate_data.py
# Generates real GPT-2 / DistilGPT-2 logits and saves as .npy files
# for the CUDA speculative decoding pipeline.
#
# Usage:
#   pip install transformers torch
#   python scripts/generate_data.py
#   ./speculative_decoding --data-dir data

import os
import numpy as np
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

K      = 5
PROMPT = "The future of artificial intelligence is"
OUTDIR = "data"

print("Loading models (first run downloads ~500MB)...")
tokenizer    = AutoTokenizer.from_pretrained("gpt2")
target_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
draft_model  = GPT2LMHeadModel.from_pretrained("distilgpt2").eval()

inputs    = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"]

# --- Draft model: generate K tokens autoregressively ---
# Each step: get logits, softmax, sample, append to sequence
draft_probs_list = []
draft_token_ids  = []
current_ids = input_ids.clone()

print(f"Generating {K} draft tokens with DistilGPT-2...")
with torch.no_grad():
    for step in range(K):
        out    = draft_model(current_ids)
        logits = out.logits[:, -1, :]          # last position only
        probs  = torch.softmax(logits, dim=-1)
        token  = torch.multinomial(probs, 1)   # sample one token
        draft_probs_list.append(probs.squeeze(0))
        draft_token_ids.append(token.item())
        current_ids = torch.cat([current_ids, token], dim=1)

# --- Target model: evaluate ALL K tokens in ONE forward pass ---
# This is the key insight of speculative decoding:
# instead of K forward passes, we do 1 pass with K+1 tokens.
print("Evaluating all draft tokens with GPT-2 (single forward pass)...")
full_ids = torch.cat([input_ids, torch.tensor([draft_token_ids])], dim=1)
with torch.no_grad():
    target_out = target_model(full_ids)

# Extract target probabilities at each draft position
target_probs_list = []
for i in range(K):
    # Position i in the draft corresponds to index (input_len + i - 1) in output
    pos_idx    = input_ids.shape[1] + i - 1
    pos_logits = target_out.logits[:, pos_idx, :]
    pos_probs  = torch.softmax(pos_logits, dim=-1)
    target_probs_list.append(pos_probs.squeeze(0))

# --- Save as .npy ---
os.makedirs(OUTDIR, exist_ok=True)
np.save(f"{OUTDIR}/p_probs.npy",
        torch.stack(target_probs_list).detach().numpy().astype(np.float32))
np.save(f"{OUTDIR}/q_probs.npy",
        torch.stack(draft_probs_list).detach().numpy().astype(np.float32))
np.save(f"{OUTDIR}/draft_tokens.npy",
        np.array(draft_token_ids, dtype=np.int32))

print(f"\nPrompt       : '{PROMPT}'")
print(f"Draft tokens : {[repr(tokenizer.decode([t])) for t in draft_token_ids]}")
print(f"Saved to     : {OUTDIR}/")
print(f"\nNext step - run CUDA pipeline:")
print(f"  ./speculative_decoding --data-dir {OUTDIR}")