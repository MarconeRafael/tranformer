"""Transformações de imagem em embeddings de patch e codificações posicionais."""
from typing import List
from .ops import matmul, add_matrix_bias
import math


def split_into_patches(image: List[List[List[float]]], patch_size: int) -> List[List[float]]:
    H = len(image)
    W = len(image[0])
    C = len(image[0][0])
    assert H % patch_size == 0 and W % patch_size == 0, "Image dims must be divisible by patch_size"
    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = []
            for di in range(patch_size):
                for dj in range(patch_size):
                    pixel = image[i + di][j + dj]
                    for ch in range(C):
                        patch.append(pixel[ch])
            patches.append(patch)
    return patches


def linear_patch_embedding(patches: List[List[float]], W: List[List[float]], b: List[float]) -> List[List[float]]:
    return add_matrix_bias(matmul(patches, W), b)


def add_cls_token(embeddings: List[List[float]], cls_vector: List[float]) -> List[List[float]]:
    return [cls_vector[:]] + embeddings


def positional_encoding(sequence_length: int, D: int) -> List[List[float]]:
    pe = [[0.0] * D for _ in range(sequence_length)]
    for pos in range(sequence_length):
        for i in range(0, D, 2):
            div = 10000 ** (i / D)
            pe[pos][i] = math.sin(pos / div)
            if i + 1 < D:
                pe[pos][i + 1] = math.cos(pos / div)
    return pe


def combine_embeddings(patch_embeddings: List[List[float]], pos_encodings: List[List[float]]) -> List[List[float]]:
    return [[x + p for x, p in zip(row, pos)] for row, pos in zip(patch_embeddings, pos_encodings)]
