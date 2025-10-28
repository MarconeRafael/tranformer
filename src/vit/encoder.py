"""Bloco encoder Transformer: norm, residual e feedforward."""
from typing import List
from .ops import relu_vec, matmul, add_matrix_bias
import math


def layer_normalization(input_tensor: List[List[float]], gamma: List[float], beta: List[float], epsilon: float = 1e-5) -> List[List[float]]:
    out = []
    D = len(input_tensor[0]) if input_tensor else 0
    for x in input_tensor:
        mean = sum(x) / D
        var = sum((xi - mean) ** 2 for xi in x) / D
        denom = math.sqrt(var + epsilon)
        normalized = [((xi - mean) / denom) * gamma[i] + beta[i] for i, xi in enumerate(x)]
        out.append(normalized)
    return out


def residual_connection(x: List[List[float]], sublayer_out: List[List[float]]) -> List[List[float]]:
    return [[a + b for a, b in zip(x_row, s_row)] for x_row, s_row in zip(x, sublayer_out)]


def positionwise_feedforward(X: List[List[float]], W1: List[List[float]], b1: List[float], W2: List[List[float]], b2: List[float]) -> List[List[float]]:
    hidden = add_matrix_bias(matmul(X, W1), b1)
    hidden = [relu_vec(h) for h in hidden]
    out = add_matrix_bias(matmul(hidden, W2), b2)
    return out


def transformer_encoder_layer(X: List[List[float]],
                              params: dict,
                              h: int,
                              mha_fn) -> List[List[float]]:
    attn_out = mha_fn(X,
                      params['Wq'], params['bq'],
                      params['Wk'], params['bk'],
                      params['Wv'], params['bv'],
                      params['Wo'], params['bo'],
                      h)
    X2 = residual_connection(X, attn_out)
    X2_norm = layer_normalization(X2, params['ln1_gamma'], params['ln1_beta'])
    ff_out = positionwise_feedforward(X2_norm, params['W1'], params['b1'], params['W2'], params['b2'])
    X3 = residual_connection(X2_norm, ff_out)
    X3_norm = layer_normalization(X3, params['ln2_gamma'], params['ln2_beta'])
    return X3_norm


def full_encoder_stack(X: List[List[float]], layers_params: List[dict], h: int, mha_fn) -> List[List[float]]:
    out = X
    for params in layers_params:
        out = transformer_encoder_layer(out, params, h, mha_fn)
    return out
