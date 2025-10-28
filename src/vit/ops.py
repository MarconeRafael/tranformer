"""Operações matriciais e funções auxiliares (implementação pura em Python).
Boas práticas: funções curtas e tipagem mínima com comentários necessários.
"""
import math
import random
from typing import List


def zeros(shape):
    if len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]


def rand_matrix(rows: int, cols: int, scale: float = 0.02) -> List[List[float]]:
    return [[(random.random() * 2 - 1) * scale for _ in range(cols)] for _ in range(rows)]


def transpose(M: List[List[float]]) -> List[List[float]]:
    if not M:
        return []
    return [[M[r][c] for r in range(len(M))] for c in range(len(M[0]))]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    if not A or not B:
        return []
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    assert n == n2, "Incompatible shapes for matmul"
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        ai = A[i]
        ci = C[i]
        for k in range(n):
            a_ik = ai[k]
            bk = B[k]
            for j in range(p):
                ci[j] += a_ik * bk[j]
    return C


def matvec(M: List[List[float]], v: List[float]) -> List[float]:
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def add_matrix_bias(X: List[List[float]], b: List[float]) -> List[List[float]]:
    return [[x + b[j] for j, x in enumerate(row)] for row in X]


def add_vectors(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def relu_vec(v: List[float]) -> List[float]:
    return [x if x > 0 else 0.0 for x in v]


def softmax(vec: List[float]) -> List[float]:
    m = max(vec) if vec else 0.0
    exps = [math.exp(x - m) for x in vec]
    s = sum(exps) + 1e-12
    return [e / s for e in exps]


def softmax_rows(M: List[List[float]]) -> List[List[float]]:
    return [softmax(row) for row in M]
