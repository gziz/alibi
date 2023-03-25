import torch
import argparse

from alibi_generator import ALiBiAttentionMaskGenerator


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--loops", type=int, default=1)
args = parser.parse_args()

seq_len = 512
model_size = 100 # arbitrary

alibi_gen = ALiBiAttentionMaskGenerator(num_heads=8)
x = torch.empty((seq_len,model_size))

for _ in range(args.loops):
    alibi_gen = ALiBiAttentionMaskGenerator(num_heads=8)
    alibi_gen(x)