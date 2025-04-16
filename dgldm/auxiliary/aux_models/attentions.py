from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (:obj:`int`): The number of channels in the query.
        context_dim (:obj:`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (:obj:`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (:obj:`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size).contiguous()
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size).contiguous()
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of

        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        return self.to_out(hidden_states)

    def _attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = torch.matmul(query, key.transpose(-1, -2).contiguous()) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output
        hidden_states = torch.matmul(attention_probs, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2).contiguous()) * self.scale
            )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (:obj:`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        glu (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use GLU activation.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        return self.net(hidden_states)


class OffsetAttC2SInter(nn.Module):
    def __init__(self, c_in_channels=64, s_in_channels=64, n_heads=1, num_groups=32, dropout=0.0, gated_ff=True,):
        super().__init__()
        # style feat projecter
        self.style_proj_in = nn.Conv2d(s_in_channels, s_in_channels, kernel_size=1, stride=1, padding=0)
        self.gnorm_s = torch.nn.GroupNorm(num_groups=num_groups, num_channels=s_in_channels, eps=1e-6, affine=True)
        self.ln_s = nn.LayerNorm(s_in_channels)

        # content feat projecter
        self.content_proj_in = nn.Conv2d(c_in_channels, c_in_channels, kernel_size=1, stride=1, padding=0)
        self.gnorm_c = torch.nn.GroupNorm(num_groups=num_groups, num_channels=c_in_channels, eps=1e-6, affine=True)
        self.ln_c = nn.LayerNorm(c_in_channels)

        # cross-attention
        # dim_head is the middle dealing dimension, output dimension will be change to quert_dim by Linear
        self.cross_attention = CrossAttention(query_dim=c_in_channels, context_dim=s_in_channels, heads=n_heads,
                                              dim_head=s_in_channels, dropout=dropout)

        # FFN
        self.ff = FeedForward(c_in_channels, dropout=dropout, glu=gated_ff)
        self.ln_ff = nn.LayerNorm(c_in_channels)
        self.gnorm_out = torch.nn.GroupNorm(num_groups=num_groups, num_channels=c_in_channels, eps=1e-6, affine=True)
        self.proj_out = nn.Conv2d(c_in_channels, 1 * 2 * 3 * 3, kernel_size=1, stride=1, padding=0)

    def forward(self, content_hidden_states, style_hidden_states):
        B, C_c, H, W = content_hidden_states.shape
        _, C_s, _, _ = style_hidden_states.shape
        # style projecter
        style_hidden_states = self.gnorm_s(style_hidden_states)
        style_hidden_states = self.style_proj_in(style_hidden_states)
        style_hidden_states = style_hidden_states.permute(0, 2, 3, 1).reshape(B, H * W, C_s).contiguous()
        style_hidden_states = self.ln_s(style_hidden_states)

        # content projecter
        content_hidden_states = self.gnorm_c(content_hidden_states)
        content_hidden_states = self.content_proj_in(content_hidden_states)
        content_hidden_states = content_hidden_states.permute(0, 2, 3, 1).reshape(B, H * W, C_c).contiguous()
        content_hidden_states = self.ln_c(content_hidden_states)

        # style and content cross-attention
        hidden_states = self.cross_attention(content_hidden_states, context=style_hidden_states)

        # ffn
        hidden_states = self.ff(self.ln_ff(hidden_states)) + hidden_states

        # reshape
        _, _, c = hidden_states.shape
        reshape_out = hidden_states.permute(0, 2, 1).reshape(B, c, H, W).contiguous()

        # projert out
        reshape_out = self.gnorm_out(reshape_out)
        offset_out = self.proj_out(reshape_out)
        return offset_out


class OffsetAttS2CInter(nn.Module):
    def __init__(self, c_in_channels=64, s_in_channels=64, n_heads=1, num_groups=32, dropout=0.0, gated_ff=True,):
        super().__init__()

        # style feat projecter
        self.style_proj_in = nn.Conv2d(s_in_channels, s_in_channels, kernel_size=1, stride=1, padding=0)
        self.gnorm_s = torch.nn.GroupNorm(num_groups=num_groups, num_channels=s_in_channels, eps=1e-6, affine=True)
        self.ln_s = nn.LayerNorm(s_in_channels)

        # content feat projecter
        self.content_proj_in = nn.Conv2d(c_in_channels, c_in_channels, kernel_size=1, stride=1, padding=0)
        self.gnorm_c = torch.nn.GroupNorm(num_groups=num_groups, num_channels=c_in_channels, eps=1e-6, affine=True)
        self.ln_c = nn.LayerNorm(c_in_channels)

        # cross-attention
        self.cross_attention = CrossAttention(query_dim=s_in_channels, context_dim=c_in_channels, heads=n_heads,
                                              dim_head=c_in_channels, dropout=dropout)

        # FFN
        self.ff = FeedForward(s_in_channels, dropout=dropout, glu=gated_ff)
        self.ln_ff = nn.LayerNorm(s_in_channels)
        self.gnorm_out = torch.nn.GroupNorm(num_groups=num_groups, num_channels=s_in_channels, eps=1e-6,
                                            affine=True)
        self.proj_out = nn.Conv2d(s_in_channels, 1 * 2 * 3 * 3, kernel_size=1, stride=1, padding=0)

    def forward(self, content_hidden_states, style_hidden_states):
        B, C_c, H, W = content_hidden_states.shape
        _, C_s, _, _ = style_hidden_states.shape
        # style projecter
        style_hidden_states = self.gnorm_s(style_hidden_states)
        style_hidden_states = self.style_proj_in(style_hidden_states)
        style_hidden_states = style_hidden_states.permute(0, 2, 3, 1).reshape(B, H * W, C_s).contiguous()
        style_hidden_states = self.ln_s(style_hidden_states)

        # content projecter
        content_hidden_states = self.gnorm_c(content_hidden_states)
        content_hidden_states = self.content_proj_in(content_hidden_states)
        content_hidden_states = content_hidden_states.permute(0, 2, 3, 1).reshape(B, H * W, C_c).contiguous()
        content_hidden_states = self.ln_c(content_hidden_states)

        # style and content cross-attention
        hidden_states = self.cross_attention(style_hidden_states, context=content_hidden_states)

        # ffn
        hidden_states = self.ff(self.ln_ff(hidden_states)) + hidden_states

        # reshape
        _, _, c = hidden_states.shape
        reshape_out = hidden_states.permute(0, 2, 1).reshape(B, c, H, W).contiguous()

        # projert out
        reshape_out = self.gnorm_out(reshape_out)
        offset_out = self.proj_out(reshape_out)
        return offset_out


def tst_OffsetAttS2CInter():
    print('tst_OffsetAttS2CInter')
    device = 'cuda:0'
    B, C, H, W = 8, 128, 48, 48
    content_hidden_states = torch.randn(size=(B, C, H, W)).to(device)
    style_hidden_states = torch.randn(size=(B, C, H, W)).to(device)
    deformable_concat = torch.cat((style_hidden_states, content_hidden_states), 1)
    offsetAttS2CInter = OffsetAttS2CInter(c_in_channels=C, s_in_channels=C*2).to(device)
    offset_out = offsetAttS2CInter(content_hidden_states, deformable_concat)
    print('content', content_hidden_states.shape, 'style', deformable_concat.shape, 'offset', offset_out.shape)


def tst_OffsetAttC2SInter():
    print('tst_OffsetAttC2SInter')
    device = 'cuda:0'
    B, C, H, W = 8, 128, 48, 48
    content_hidden_states = torch.randn(size=(B, C, H, W)).to(device)
    style_hidden_states = torch.randn(size=(B, C, H, W)).to(device)
    deformable_concat = torch.cat((style_hidden_states, content_hidden_states), 1)
    offsetAttC2SInter = OffsetAttC2SInter(c_in_channels=C, s_in_channels=C*2).to(device)
    offset_out = offsetAttC2SInter(content_hidden_states, deformable_concat)
    print('content', content_hidden_states.shape, 'style', deformable_concat.shape, 'offset', offset_out.shape)


if __name__ == '__main__':
    tst_OffsetAttS2CInter()
    tst_OffsetAttC2SInter()
