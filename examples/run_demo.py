"""Demo mínimo para executar forward do ViT com dados aleatórios."""
import sys
import os
import random

# Ensure src directory is importable when running the demo directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vit.model import init_vit_params, forward_vit


def make_random_image(H, W, C, seed=0):
    random.seed(seed)
    return [[[random.random() for _ in range(C)] for _ in range(W)] for _ in range(H)]


if __name__ == "__main__":
    H = W = 4
    C = 3
    img = make_random_image(H, W, C, seed=1)
    patch_size = 2
    patch_dim = patch_size * patch_size * C
    D = 16
    h = 4
    num_layers = 2
    d_ff = 32
    num_classes = 5
    params = init_vit_params(patch_dim, D, num_layers, h, d_ff, num_classes, seed=1)
    probs = forward_vit(img, patch_size, D, params, num_layers, h)
    print("probs:", probs)
