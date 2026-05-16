"""Transformer Encoder for sequential feature extraction.
Processes time-series of feature vectors into dense embeddings.
Optional Mamba (S4-based) backbone for linear-time sequence modeling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import config

_model_cfg = config.get("models", {}).get("primary", {}).get("transformer", {})


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FeatureTransformerEncoder(nn.Module):
    """Transformer encoder that processes sequences of feature vectors.

    Input:  (batch, seq_len, n_features) — raw feature sequences
    Output: (batch, d_model) — dense embedding for downstream heads
    """

    def __init__(
        self,
        n_features: int = 82,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        seq_len: int = 60,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model

        # Project raw features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for better training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Output: pool over sequence → single vector
        self.output_norm = nn.LayerNorm(d_model)

        # Attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features) feature sequences
            mask: (batch, seq_len) boolean mask, True = padded

        Returns:
            (batch, d_model) pooled embedding
        """
        # Project to d_model
        h = self.input_proj(x)  # (B, T, d_model)
        h = self.pos_enc(h)

        # Transformer encoding
        if mask is not None:
            h = self.transformer(h, src_key_padding_mask=mask)
        else:
            h = self.transformer(h)

        h = self.output_norm(h)

        # Attention-weighted pooling
        attn_weights = self.attn_pool(h).squeeze(-1)  # (B, T)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)  # (B, T, 1)
        pooled = (h * attn_weights).sum(dim=1)  # (B, d_model)

        return pooled


class MambaBlock(nn.Module):
    """Simplified Mamba-style selective state space block.
    Linear-time sequence modeling alternative to attention.
    """

    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projection
        self.in_proj = nn.Linear(d_model, d_model * 2)
        # Selective mechanism
        self.dt_proj = nn.Linear(d_model, d_model)
        self.A = nn.Parameter(torch.randn(d_model, d_state))
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)"""
        residual = x
        x = self.norm(x)

        # Split into two paths
        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)

        # Selective scan (simplified)
        dt = F.softplus(self.dt_proj(x_path))  # (B, T, D)
        B = self.B_proj(x_path)  # (B, T, N)
        C = self.C_proj(x_path)  # (B, T, N)

        # Discretize A
        A = -torch.exp(self.A)  # (D, N)

        # Sequential scan
        batch, seq_len, d = x_path.shape
        h = torch.zeros(batch, d, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            dt_t = dt[:, t].unsqueeze(-1)  # (B, D, 1)
            dA = torch.exp(dt_t * A.unsqueeze(0))  # (B, D, N)
            dB = dt_t * B[:, t].unsqueeze(1)  # (B, D, N)
            h = dA * h + dB * x_path[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, T, D)
        y = y + self.D * x_path
        y = y * F.silu(z)

        y = self.out_proj(y)
        y = self.dropout(y) + residual
        return y


class FeatureMambaEncoder(nn.Module):
    """Mamba-based encoder as alternative to Transformer."""

    def __init__(
        self,
        n_features: int = 82,
        d_model: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.layers = nn.ModuleList([
            MambaBlock(d_model, dropout=dropout) for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.attn_pool = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        h = self.output_norm(h)

        # Attention pooling
        weights = self.attn_pool(h).squeeze(-1)
        if mask is not None:
            weights = weights.masked_fill(mask, float("-inf"))
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)
        return (h * weights).sum(dim=1)
