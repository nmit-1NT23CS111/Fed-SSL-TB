"""MAE model used by the encoder-only federated TB pipeline."""

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import MAEDecoder
from src.models.encoder import get_encoder, ResNet50Encoder
from src.models.proto_head import PrototypicalHead


class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder for unsupervised NIH pretraining. Only the encoder is federated."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        mask_ratio: float = 0.75,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        projection_dim: int = 128,
        proto_head: nn.Module | None = None,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proto_head = proto_head or PrototypicalHead(embed_dim=encoder.embed_dim, projection_dim=projection_dim)
        self.num_patches = (image_size // patch_size) ** 2
        self._is_resnet = isinstance(encoder, ResNet50Encoder)

        if self._is_resnet:
            _patch_dim = patch_size * patch_size * in_channels
            self.patch_embed = nn.Linear(_patch_dim, encoder.embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder.embed_dim), requires_grad=False)
            self._init_pos_embed()
            self.pre_norm = nn.LayerNorm(encoder.embed_dim)
            self.patch_encoder = nn.Sequential(
                nn.Linear(encoder.embed_dim, encoder.embed_dim),
                nn.GELU(),
                nn.Linear(encoder.embed_dim, encoder.embed_dim),
                nn.LayerNorm(encoder.embed_dim),
            )

    def _init_pos_embed(self) -> None:
        from src.models.decoder import _get_sinusoidal_pos_embed

        emb = _get_sinusoidal_pos_embed(self.encoder.embed_dim, self.num_patches)
        self.pos_embed.data.copy_(emb.unsqueeze(0))

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        assert H == W == self.image_size, f"Expected {self.image_size}x{self.image_size}, got {H}x{W}"
        h = w = H // p
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, _ = patches.shape
        p = self.patch_size
        C = self.in_channels
        h = w = int(N ** 0.5)
        patches = patches.reshape(B, h, w, p, p, C)
        patches = patches.permute(0, 5, 1, 3, 2, 4)
        return patches.reshape(B, C, h * p, w * p)

    def random_masking(self, tokens: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = tokens.shape
        num_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        visible_tokens = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=tokens.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return visible_tokens, mask, ids_restore

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.vit.patch_embed(x)
        tokens = tokens + self.encoder.vit.pos_embed[:, 1:, :]
        tokens = self.encoder.vit.pos_drop(tokens)
        return tokens

    def _encode_visible(self, visible_tokens: torch.Tensor) -> torch.Tensor:
        x = visible_tokens
        for block in self.encoder.vit.blocks:
            x = block(x)
        x = self.encoder.vit.norm(x)
        return self.encoder.projection(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(x, (list, tuple)):
            x = x[0]
        target = self.patchify(x)
        tokens = self._tokenize(x)
        visible_tokens, mask, ids_restore = self.random_masking(tokens, self.mask_ratio)
        encoded = self._encode_visible(visible_tokens)
        pred = self.decoder(encoded, ids_restore)
        loss = self._mae_loss(pred, target, mask)
        return loss, pred, mask

    def _mae_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

    def get_encoder_weights(self) -> Dict[str, Any]:
        return {k: v.clone() for k, v in self.encoder.state_dict().items()}

    def get_federated_weights(self) -> Dict[str, Dict[str, Any]]:
        return {"encoder": {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()}}

    def load_federated_weights(self, state: Dict[str, Dict[str, Any]]) -> None:
        if "encoder" in state:
            self.encoder.load_state_dict(state["encoder"])
            return
        self.load_encoder_weights(state)

    def load_encoder_weights(self, state_dict: Dict[str, Any]) -> None:
        self.encoder.load_state_dict(state_dict)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x = x[0]
        embedding = self.encoder(x)
        if embedding.ndim != 2 or embedding.shape[-1] != self.encoder.embed_dim:
            raise ValueError(f"Encoder output contract violated: expected [B, {self.encoder.embed_dim}], got {tuple(embedding.shape)}")
        return embedding


def build_mae(
    backbone: str = "vit_tiny",
    embed_dim: int = 192,
    mask_ratio: float = 0.75,
    decoder_depth: int = 4,
    image_size: int = 224,
    patch_size: int = 16,
    in_channels: int = 3,
    projection_dim: int = 128,
) -> MaskedAutoencoder:
    num_patches = (image_size // patch_size) ** 2
    encoder = get_encoder(backbone=backbone, embed_dim=embed_dim)
    decoder = MAEDecoder(
        embed_dim=embed_dim,
        num_patches=num_patches,
        patch_size=patch_size,
        in_channels=in_channels,
        decoder_embed_dim=256,
        decoder_depth=decoder_depth,
        decoder_num_heads=8,
    )
    return MaskedAutoencoder(
        encoder=encoder,
        decoder=decoder,
        mask_ratio=mask_ratio,
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        projection_dim=projection_dim,
    )

