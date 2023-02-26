import torch

from alibi import ALiBiAttentionMaskGenerator


def main():
    alibi_gen = ALiBiAttentionMaskGenerator()
    mask = alibi_gen(torch.empty(6, 8))
    print(mask.shape)
    print(mask[:,:,0])


if __name__ == '__main__':
    main()