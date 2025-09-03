import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from torch import Tensor, nn
from typing import Dict, Iterable, Optional, List

def attention_mask(loss_mask, prefix_lm=True):
    """
    Generate the attention mask from the loss mask,
    where the loss mask is in the format [Batch, Length].
    Usually, the loss mask would look like:
      <False> ... <True> ... <False>, which represents the
    prefix, the target sequence and padding respectively.

    This function generates the mask for multi-head attention,
    which is in the shape of [Batch, Length, Length] and features:
    (1) the prefix entries can see all each other, if prefix_lm,
        otherwise causal;
    (2) the target entries are causal to each other and can see all
        prefix entries;
    (3) the padding entries can neither been seen nor see all other
        entries.
    """

    # basic preparation
    device = loss_mask.device
    batch_size, q_len = loss_mask.size()
    axis = torch.arange(q_len).to(device)
    # find the start and end time indices of loss duration
    start = axis.unsqueeze(0).masked_fill(~loss_mask, 1e8).min(dim=1).values
    end = axis.unsqueeze(0).masked_fill(~loss_mask, -1e8).max(dim=1).values
    # we strictly require that there is only one continuous True segment
    # for each example in the loss_mask:
    assert torch.all(end - start == loss_mask.int().sum(dim=-1) - 1)

    # (1) make it causal
    mask = (axis.unsqueeze(1) >= axis.unsqueeze(0)).repeat(batch_size, 1, 1)
    # (2) allow non-causaility in prefix part, if prefix_lm
    if prefix_lm:
        mask = torch.where(start.view(batch_size, 1, 1) > axis.view(1, 1, q_len),
                       True, mask)

    # (3) kill the padding
    mask = torch.where(end.view(batch_size, 1, 1) < axis.view(1, 1, q_len),
                       False, mask)

    return mask

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x, attention_mask=None, prefix_causal_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if prefix_causal_mask is not None:
            # print('prefix_causal_mask ', prefix_causal_mask.shape)
            # print('att ', att.shape)
            att = att.masked_fill(prefix_causal_mask.unsqueeze(1) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalBlock(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout, block_size):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, bias, dropout, block_size)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x, attention_mask=None, prefix_causal_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask, prefix_causal_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class AnyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, dropout, n_head, bias, n_layer):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([CausalBlock(n_embd, n_head, bias, dropout, block_size) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias=bias),
        ))
        self.block_size = block_size
        self.pos_condition = nn.Embedding(block_size, n_embd) #  the positional embedding for condition (pre-fix)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, condition, mask, idx, targets):
        """
        mask: for condition part, 0 表示被mask
        """
        device = idx.device
        b, t = idx.size()
        _, c_t, d = condition.size() # 
        pos_c = torch.arange(0, c_t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        prefix_emb = condition + self.pos_condition(pos_c) # add pre-fix condition position embedding
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        x_combine = torch.cat([prefix_emb, x], dim=1) # combine
        x_mask = torch.ones((b, t)).to(x.device) # 1 present can be seen
        pad_attention_mask = torch.cat([mask, x_mask], dim=1) # the key mask, B,T
        pad_attention_mask = pad_attention_mask.bool()
        #print('pad_attention_mask ', pad_attention_mask.shape)
        #print('attention_mask ', attention_mask.shape)
        loss_causal_mask = torch.cat([torch.zeros((b, c_t)), torch.ones((b, t))], dim=1).to(x.device)
        prefix_causal_mask = attention_mask(loss_causal_mask.bool())
        #print('prefix_causal_mask ', prefix_causal_mask.shape)
        prefix_causal_mask = torch.where(~pad_attention_mask.unsqueeze(1), False, prefix_causal_mask)
        # print('prefix_causal_mask ', prefix_causal_mask)
        # assert 1==2
        for block in self.transformer.h:
            x_combine = block(x_combine, pad_attention_mask, prefix_causal_mask)
        x_combine = x_combine[:,c_t:,:] # only use the non-condition part
        x_combine = self.transformer.ln_f(x_combine)
        logits = self.lm_head(x_combine)
        #print('logits ', logits.shape, targets.shape)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        acc = self.compute_accuracy(logits, targets)
        # print('acc ', acc, loss)
        # assert 1==2
        return logits, loss, acc

    def compute_accuracy(self, lprobs, target):
        n_correct = torch.sum(
            lprobs.argmax(-1).eq(target)
        )
        total = torch.sum(target.ne(-1))
        return n_correct / total
