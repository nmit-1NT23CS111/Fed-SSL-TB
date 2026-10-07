"""Central federated server for the encoder-only FL pipeline."""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.models.mae import MaskedAutoencoder, build_mae
from src.server.aggregator import fedavg, fedprox


class FederatedServer:
    def __init__(self, config, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model: Optional[MaskedAutoencoder] = None

        self.checkpoint_dir = Path(config.logging.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_validation_auc: float = 0.0
        self.best_round: int = -1
        self.best_encoder_weights: Optional[Dict[str, Any]] = None

        self.aggregation = str(config.federated.aggregation).lower()
        self.fedprox_mu = float(getattr(config.federated, "fedprox_mu", 0.01))

    def initialize_global_model(self) -> MaskedAutoencoder:
        self.global_model = build_mae(
            backbone=self.config.model.backbone,
            embed_dim=self.config.model.embed_dim,
            mask_ratio=self.config.model.mask_ratio,
            decoder_depth=self.config.model.decoder_depth,
            image_size=self.config.data.image_size,
            projection_dim=int(getattr(getattr(self.config, "few_shot", None), "projection_dim", getattr(self.config.model, "embed_dim", 192))),
        ).to(self.device)
        print(f"[Server] Global model initialized | Backbone: {self.config.model.backbone} | Embed dim: {self.config.model.embed_dim} | Device: {self.device}")
        return self.global_model

    def broadcast(self) -> Dict[str, Any]:
        assert self.global_model is not None, "Global model not initialized. Call initialize_global_model() first."
        return self.global_model.get_federated_weights()

    def get_global_weights(self) -> Dict[str, Any]:
        return self.broadcast()

    def aggregate(self, received_weights: List[Dict[str, Any]], sample_counts: List[int]) -> Dict[str, Any]:
        if self.aggregation == "fedavg":
            aggregated = fedavg(received_weights, sample_counts)
        elif self.aggregation == "fedprox":
            aggregated = fedprox(self.broadcast(), received_weights, sample_counts, mu=self.fedprox_mu)
        else:
            raise ValueError(f"Unknown aggregation strategy '{self.aggregation}'. Use 'fedavg' or 'fedprox'.")

        if "encoder" not in aggregated:
            raise ValueError("Federated aggregation must return encoder-only payloads.")
        return aggregated

    def update_global_model(self, aggregated_weights: Dict[str, Any]) -> None:
        assert self.global_model is not None
        self.global_model.load_federated_weights(aggregated_weights)

    def save_checkpoint(self, round_num: int, metrics: Optional[Dict[str, Any]] = None) -> str:
        assert self.global_model is not None

        checkpoint = {
            "round": round_num,
            "global_encoder_state_dict": self.global_model.encoder.state_dict(),
            "encoder_state_dict": self.global_model.encoder.state_dict(),
            "config": {
                "backbone": self.config.model.backbone,
                "embed_dim": self.config.model.embed_dim,
                "mask_ratio": self.config.model.mask_ratio,
            },
        }
        if metrics:
            checkpoint["validation_metrics"] = metrics

        ckpt_path = self.checkpoint_dir / f"flame_round_{round_num:03d}.pt"
        torch.save(checkpoint, str(ckpt_path))

        candidate_auc = None
        for key in ("val_auc", "auc", "validation_auc"):
            if key in (metrics or {}):
                candidate_auc = float(metrics[key])
                break
        if candidate_auc is not None and torch.isfinite(torch.tensor(candidate_auc)):
            if candidate_auc > self.best_validation_auc:
                self.best_validation_auc = candidate_auc
                self.best_round = round_num
                self.best_encoder_weights = copy.deepcopy(self.global_model.get_encoder_weights())
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save({**checkpoint, "best_validation_auc": self.best_validation_auc, "best_round": self.best_round}, str(best_path))
                print(f"  [Server] [BEST VALIDATION MODEL] saved | Round {round_num} | AUC={self.best_validation_auc:.4f}")

        return str(ckpt_path)

    def load_checkpoint(self, path: str) -> int:
        assert self.global_model is not None
        ckpt = torch.load(path, map_location=self.device)
        encoder_state = ckpt.get("global_encoder_state_dict", ckpt.get("encoder_state_dict"))
        self.global_model.load_encoder_weights(encoder_state)
        round_num = int(ckpt.get("round", 0))
        print(f"[Server] Loaded checkpoint from round {round_num}: {path}")
        return round_num

    def get_encoder(self) -> nn.Module:
        assert self.global_model is not None
        return self.global_model.encoder

    def get_global_model(self) -> MaskedAutoencoder:
        assert self.global_model is not None
        return self.global_model

    def summary(self) -> str:
        return f"FederatedServer | Aggregation: {self.aggregation} | Best AUC: {self.best_validation_auc:.4f} @ Round {self.best_round}"
