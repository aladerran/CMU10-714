import torch
import torch.nn as nn
import numpy as np

## Attention

# y = (softmax((K @ Q^T) / (d**0/5)) @ V) @ W_out
# where K = X @ W_k, Q = X @ W_q, V = X @ W_v

def softmax (Z):
    Z = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return Z / Z.sum(axis=-1, keepdims=True)

def self_attention(X, mask, W_kqv, W_out):
    K, Q, V = np.split(X @ W_kqv, 3 , axis=-1)
    attn = softmax(K @ Q.swapaxes(-1, -2) / np.sqrt(X.shape[-1]) + mask)
    return attn @ V @ W_out, attn

B, T, d = 50, 100, 64
attn = nn.MultiheadAttention(d, 1, bias=False, batch_first=True)
M = torch.triu(-float("inf")*torch.ones(T,T), 1)
X = torch.randn(B, T, d)
Y_, A_ = attn(X, X, X, attn_mask=M)

# print(attn.in_proj_weight.shape)
# print(attn.out_proj.weight.shape)

Y, A = self_attention(X.numpy(), M.numpy(), 
                      attn.in_proj_weight.detach().numpy().T,
                      attn.out_proj.weight.detach().numpy().T)

print("Y:", Y.shape)
print("Y_:", Y_.shape)

print("Error:", np.linalg.norm(Y - Y_.detach().numpy()))

## Multihead Attention 

# K = [K1, K2, ...] , Q ..., V ...
# y_i = softmax((K_i @ Q_i^T) / ((d/n)**0.5) @ V_i)
# y = [y1, y2, ...] @ W_out

def multihead_attention(X, mask, heads, W_kqv, W_out):
    B, T, d = X.shape
    K, Q, V = np.split(X @ W_kqv, 3, axis=-1)
    # B x T x d -> B x heads x T x d/heads
    K, Q, V = [_.reshape(B, T, heads, d // heads).swapaxes(1,2) for _ in (K, Q, V)]
    attn = softmax(K @ Q.swapaxes(-1,-2) / np.sqrt(d // heads) + mask)
    return (attn @ V).swapaxes(1,2).reshape(B, T, d) @ W_out, attn

heads = 4
attn = nn.MultiheadAttention(d, heads, bias=False, batch_first=True)
Y_, A_ = attn(X, X, X, attn_mask=M)

Y, A = multihead_attention(X.numpy(), M.numpy(), heads,
                      attn.in_proj_weight.detach().numpy().T,
                      attn.out_proj.weight.detach().numpy().T)

print("Y:", Y.shape)
print("Y_:", Y_.shape)

print("Error:", np.linalg.norm(Y - Y_.detach().numpy()))

## Transformer Block

# X -> Q, K, V -> MultiheadAttention -> ADD & LayerNorm -> FF1 -> ReLU -> FF2 -> Add & LayerNorm

def layer_norm(Z, eps):
    return (Z - Z.mean(axis=-1, keepdims=True)) \
        / np.sqrt(Z.var(axis=-1, keepdims=True) + eps)

def relu(Z):
    return np.maximum(Z, 0)

def transformer(X, mask, heads, W_kqv, W_out, W_ff1, W_ff2, eps):
    Z = layer_norm(X + multihead_attention(X, mask, heads, W_kqv, W_out)[0], eps)
    return layer_norm(Z + relu(Z @ W_ff1) @ W_ff2, eps)

trans = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128,
                                    dropout=0.0, batch_first=True)

trans.linear1.bias.data.zero_()
trans.linear2.bias.data.zero_()
Y_ = trans(X, M)

Y = transformer(X.numpy(), M.numpy(), heads,
                      trans.self_attn.in_proj_weight.detach().numpy().T,
                      trans.self_attn.out_proj.weight.detach().numpy().T,
                      trans.linear1.weight.detach().numpy().T,
                      trans.linear2.weight.detach().numpy().T,
                      trans.norm1.eps)

print("Y:", Y.shape)
print("Y_:", Y_.shape)

print("Error:", np.linalg.norm(Y - Y_.detach().numpy()))
