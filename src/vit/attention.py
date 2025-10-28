"""Implementação do scaled dot-product attention e multi-head attention."""
from typing import List
from .ops import matmul, transpose, softmax_rows, add_matrix_bias
import math


def linear_projection(input_tensor: List[List[float]], W: List[List[float]], b: List[float]) -> List[List[float]]:
    return add_matrix_bias(matmul(input_tensor, W), b)


def scaled_dot_product_attention(Q: List[List[float]], K: List[List[float]], V: List[List[float]]) -> List[List[float]]:
    K_T = transpose(K)
    scores = matmul(Q, K_T)
    d_k = len(Q[0]) if Q and Q[0] else 1
    scale = math.sqrt(d_k)
    for i in range(len(scores)):
        for j in range(len(scores[0])):
            scores[i][j] /= scale
    attn = softmax_rows(scores)
    out = matmul(attn, V)
    return out


def split_heads(X: List[List[float]], h: int) -> List[List[List[float]]]:
    S = len(X)
    D = len(X[0])
    assert D % h == 0
    dk = D // h
    heads = []
    for head in range(h):
        start = head * dk
        end = start + dk
        head_mat = []
        for i in range(S):
            head_mat.append(X[i][start:end])
        heads.append(head_mat)
    return heads


def concat_heads(heads: List[List[List[float]]]) -> List[List[float]]:
    if not heads:
        return []
    S = len(heads[0])
    out = []
    for i in range(S):
        row = []
        for h_mat in heads:
            row.extend(h_mat[i])
        out.append(row)
    return out


def multi_head_attention(X: List[List[float]],
                         Wq: List[List[float]], bq: List[float],
                         Wk: List[List[float]], bk: List[float],
                         Wv: List[List[float]], bv: List[float],
                         Wo: List[List[float]], bo: List[float],
                         h: int) -> List[List[float]]:
    Q_all = linear_projection(X, Wq, bq)
    K_all = linear_projection(X, Wk, bk)
    V_all = linear_projection(X, Wv, bv)
    Q_heads = split_heads(Q_all, h)
    K_heads = split_heads(K_all, h)
    V_heads = split_heads(V_all, h)
    out_heads = []
    for head in range(h):
        out_h = scaled_dot_product_attention(Q_heads[head], K_heads[head], V_heads[head])
        out_heads.append(out_h)
    concat = concat_heads(out_heads)
    out = linear_projection(concat, Wo, bo)
    return out
