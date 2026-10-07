"""Local hospital update with MAE + prototypical objectives and encoder-only FedProx."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


def _images(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _global_encoder_state(global_weights):
    if isinstance(global_weights, dict) and "encoder" in global_weights:
        return global_weights["encoder"]
    return global_weights


def flame_local_train(
    hospital_id: int,
    model: nn.Module,
    unlabeled_loader: DataLoader,
    support_loader: DataLoader,
    proto_train_query_loader: DataLoader,
    eval_query_loader: DataLoader,
    config,
    global_weights: Optional[Dict[str, Dict[str, Any]]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    loss_cfg = getattr(config, "loss", None)
    lambda_mae = float(getattr(loss_cfg, "lambda_mae", 0.70) if loss_cfg is not None else 0.70)
    lambda_proto = float(getattr(loss_cfg, "lambda_proto", 0.30) if loss_cfg is not None else 0.30)

    optimizer = AdamW(model.parameters(), lr=float(config.ssl.lr), weight_decay=0.05, betas=(0.9, 0.95))
    is_fedprox = getattr(config.federated, "aggregation", "fedavg").lower() == "fedprox" and global_weights is not None
    mu = float(getattr(config.federated, "fedprox_mu", 0.01))
    global_encoder_state = _global_encoder_state(global_weights)

    support_images, support_labels = next(iter(support_loader))
    support_images = support_images.to(device)
    support_labels = torch.as_tensor(support_labels, device=device).long()

    train_query_batches = list(proto_train_query_loader)
    if not train_query_batches:
        raise ValueError(f"Hospital {hospital_id} has no local prototype-train queries.")

    epoch_records = []
    for _ in range(int(config.ssl.epochs_per_round)):
        mae_total = proto_total = grad_total = 0.0
        batches = 0
        for unlabeled_batch in unlabeled_loader:
            optimizer.zero_grad()
            images = _images(unlabeled_batch).to(device)
            mae_loss, _, _ = model(images)

            query_batch = train_query_batches[batches % len(train_query_batches)]
            query_images, query_labels = query_batch
            query_images = query_images.to(device)
            query_labels = torch.as_tensor(query_labels, device=device).long()

            support_embeddings = model.encoder(support_images)
            prototypes = model.proto_head.compute_prototypes(support_embeddings, support_labels)
            query_embeddings = model.encoder(query_images)
            proto_loss, _ = model.proto_head.prototypical_loss(query_embeddings, query_labels, prototypes)

            proximal = torch.zeros((), device=device)
            if is_fedprox and isinstance(global_encoder_state, dict):
                for name, param in model.encoder.named_parameters():
                    if name in global_encoder_state:
                        proximal = proximal + torch.sum((param - global_encoder_state[name].to(device)) ** 2)
                proximal = (mu / 2.0) * proximal

            total_loss = lambda_mae * mae_loss + lambda_proto * proto_loss + proximal
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            mae_total += float(mae_loss.detach())
            proto_total += float(proto_loss.detach())
            grad_total += float(total_loss.detach())
            batches += 1

        divisor = max(batches, 1)
        epoch_records.append({
            "mae_loss": mae_total / divisor,
            "proto_loss": proto_total / divisor,
            "total_loss": grad_total / divisor,
        })

    final = epoch_records[-1]
    print(f"[Hospital {hospital_id}] MAE={final['mae_loss']:.4f} | Proto={final['proto_loss']:.4f} | Total={final['total_loss']:.4f}")

    return {
        "model_weights": model.get_federated_weights(),
        "encoder_weights": model.get_federated_weights()["encoder"],
        "num_samples": len(unlabeled_loader.dataset),
        "epoch_losses": epoch_records,
        "query_metrics": {},
        "mae_loss": final["mae_loss"],
        "proto_loss": final["proto_loss"],
        "total_loss": final["total_loss"],
    }

