# Trying to do this from the paper 8b and kinda chceking code
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
import fairscale.nn.model_parallel.initialize as fs_init



@dataclass
class ModelArgs:
    dim: int = 4096
    n_head: int = 32
    n_layers: int = 32
    vocab_size: int = 128256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000
    #res is from original config file
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  
    ffn_dim_multiplier: Optional[float] = None
    max_batch_size: int = 32
    max_seq_len: int = 2048
    

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.norm_weights = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x * torch.rsqrt((x.pow(2).mean(dim=-1, keepdim=True)) + self.eps) * self.norm_weights
        return norm

def reshape_for_broadcast(freq_cis, x):
    ndim = x.ndim
    shape = [d if i==1  or i==ndim-1 else 1 for i, d in enumerate(x.shape)]
    return freq_cis.view(*shape)

#not really sure what end is about look it up
def precompute_freqs_cis(dim: int, end:int, theta: float = 1000.0):
    base = dim / 2
    freqs = 1.0 / (theta ** torch.arange(base).float() / base)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().view(xq.shape[0], xq.shape[1], xq.shape[2], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(xk.shape[0], xk.shape[1], xk.shape[2], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Modules):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_head
        #Had to look up what this was doing but it prob for dist training in MQA, ill look into this further
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim, #is this not the same as dim??
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        #KV-cache
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int):
        batch_size, seq_len = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xv)

        self.cache_k[:batch_size, start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, start_pos + seq_len]
        values = self.cache_v[:batch_size, start_pos + seq_len]

        #skip implementatino of k/v heads logic

        xq = xq.transpose(1,2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1,2)

        scores = torch.matmult(xq, keys.transpose(2,3))/(self.head_dim)**0.5
        mask = torch.full((scores.shape), float("-inf"), device=scores.device)
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask
        scores = F.softmax(scores, dim=-1).as_type(xq)
        output = torch.matmult(scores, values)
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, params: ModelArgs):
        super().__init__()
        self.layer_idx = layer_idx
        self.params = params
        self.dim = params.dim
        self.n_heads = params.n_head
        self.head_dim = params.dim // params.n_head
        self.attention_norm = RMSNorm(params.dim, eps = params.rms_norm_eps)

        self.attention = Attention(params)
        self.feed_forward = FeedForward(params.dim, 4*params.dim)
        self.ffn_norm = RMSNorm(params.dim, eps=params.rms_norm_eps)

    def forward(self, x: torch.Tensor):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.vocab_size = params.vocab_size

        #from the model.py file
        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_idx in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_idx, params))

        self.norm = RMSNorm(params.dim, eps=params.rms_norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_head,
            params.max_seq_len * 2,
            params.rope_theta
        )

    