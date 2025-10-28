"""Modelo ViT simples (pure Python) e inicialização de parâmetros."""
import random
from typing import List
from .ops import rand_matrix, matmul
from .embedding import split_into_patches, linear_patch_embedding, add_cls_token, positional_encoding, combine_embeddings
from .attention import multi_head_attention
from .encoder import full_encoder_stack
import math


def init_vit_params(patch_dim: int, D: int, num_layers: int, h: int, d_ff: int, num_classes: int, seed: int = 0):
    random.seed(seed)
    params = {}
    params['W_patch'] = rand_matrix(patch_dim, D)
    params['b_patch'] = [0.0] * D
    params['cls'] = [0.0] * D
    layers = []
    for _ in range(num_layers):
        layer = {
            'Wq': rand_matrix(D, D), 'bq': [0.0] * D,
            'Wk': rand_matrix(D, D), 'bk': [0.0] * D,
            'Wv': rand_matrix(D, D), 'bv': [0.0] * D,
            'Wo': rand_matrix(D, D), 'bo': [0.0] * D,
            'ln1_gamma': [1.0] * D, 'ln1_beta': [0.0] * D,
            'ln2_gamma': [1.0] * D, 'ln2_beta': [0.0] * D,
            'W1': rand_matrix(D, d_ff), 'b1': [0.0] * d_ff,
            'W2': rand_matrix(d_ff, D), 'b2': [0.0] * D
        }
        layers.append(layer)
    params['layers'] = layers
    params['W_out'] = rand_matrix(D, num_classes)
    params['b_out'] = [0.0] * num_classes
    return params


def classification_head(encoder_output_cls: List[float], W_out: List[List[float]], b_out: List[float]):
    logits = matmul([encoder_output_cls], W_out)[0]
    logits = [l + bo for l, bo in zip(logits, b_out)]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps) + 1e-12
    return [e / s for e in exps]


def forward_vit(image: List[List[List[float]]],
                patch_size: int,
                D: int,
                params: dict,
                num_layers: int,
                h: int):
    patches = split_into_patches(image, patch_size)
    patch_emb = linear_patch_embedding(patches, params['W_patch'], params['b_patch'])
    seq = add_cls_token(patch_emb, params['cls'])
    pos = positional_encoding(len(seq), D)
    seq = combine_embeddings(seq, pos)
    encoder_out = full_encoder_stack(seq, params['layers'], h, multi_head_attention)
    cls_out = encoder_out[0]
    probs = classification_head(cls_out, params['W_out'], params['b_out'])
    return probs
