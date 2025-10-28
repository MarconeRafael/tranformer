"""Testes básicos para o ViT puro em Python."""
import sys
import os
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vit.model import init_vit_params, forward_vit
import random


class TestViTSmall(unittest.TestCase):
    def test_forward_shape_and_probabilities(self):
        random.seed(0)
        H = W = 4
        C = 3
        img = [[[random.random() for _ in range(C)] for _ in range(W)] for _ in range(H)]
        patch_size = 2
        patch_dim = patch_size * patch_size * C
        D = 8
        h = 2
        num_layers = 1
        d_ff = 16
        num_classes = 3
        params = init_vit_params(patch_dim, D, num_layers, h, d_ff, num_classes, seed=0)
        probs = forward_vit(img, patch_size, D, params, num_layers, h)
        self.assertEqual(len(probs), num_classes)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
