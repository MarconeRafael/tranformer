# tranformer — Vision Transformer (ViT) em Python puro

Implementação didática de um encoder Vision Transformer (ViT) escrita em Python puro (sem dependências como NumPy/PyTorch). O objetivo é educacional: mostrar cada componente do ViT implementado com listas e loops em vez de bibliotecas de alto nível.

Principais características
- Interface simples para forward de um ViT pequeno (útil para testes e aprendizado).
- Implementação "baixo nível": operações matriciais, atenção, normalização e feed-forward escritas com listas e loops.
- Projeto modular em `src/vit` (ops, embedding, attention, encoder, model).

Aviso de performance
Este código é intencionalmente simples e não é otimizado. Não é recomendado para treino em dados reais ou uso em produção — para isso prefira NumPy/PyTorch/TF.

## Estrutura do repositório

- `src/vit/ops.py` — operações matriciais e utilitários (matmul, transpose, softmax, etc.).
- `src/vit/embedding.py` — funções para dividir em patches, projetar patches em embeddings, token [CLS] e codificação posicional.
- `src/vit/attention.py` — scaled dot-product attention e multi-head attention.
- `src/vit/encoder.py` — layer normalization, residual, feed-forward e camada do encoder.
- `src/vit/model.py` — inicialização de parâmetros, forward pass e cabeçalho de classificação.
- `examples/run_demo.py` — script mínimo que executa um forward com imagem aleatória.
- `tests/test_vit.py` — teste unitário básico.

## Uso rápido

Executar demo:

```bash
cd /home/m/bolsa/tranformer
python3 -m examples.run_demo
```

Executar testes:

```bash
cd /home/m/bolsa/tranformer
python3 -m unittest tests.test_vit
```

Instalar localmente (opcional — facilita imports e desenvolvimento):

```bash
cd /home/m/bolsa/tranformer
pip install -e .
```

Após a instalação editável você pode importar `vit` diretamente de qualquer lugar no projeto sem ajustar `sys.path`.

## Funções e contratos (resumo)

- split_into_patches(image, patch_size) -> List[patch_vectors]
- linear_patch_embedding(patches, W, b) -> List[embeddings]
- add_cls_token(embeddings, cls_vector) -> sequence with CLS at index 0
- positional_encoding(seq_len, D) -> positional encodings
- linear_projection(X, W, b) -> linear projection (matrix multiply + bias)
- scaled_dot_product_attention(Q, K, V) -> attention output
- multi_head_attention(...) -> multi-head attention output
- layer_normalization(X, gamma, beta) -> normalized X
- positionwise_feedforward(X, W1, b1, W2, b2) -> feed-forward output
- transformer_encoder_layer(X, params, h) -> encoder layer output
- full_encoder_stack(X, layers_params, h) -> stacked encoder output
- classification_head(cls_vector, W_out, b_out) -> probability vector

Todos os contratos acima são implementados em `src/vit/*.py`.

## Próximos passos sugeridos

- Adicionar `pyproject.toml` para tornar o pacote instalável localmente.
- Migrar operações pesadas para NumPy/Numba/Cython para ganho de performance.
- Adicionar batching e máscaras de atenção.

## Licença
Veja o arquivo `LICENSE` na raiz do repositório.
*** End Patch