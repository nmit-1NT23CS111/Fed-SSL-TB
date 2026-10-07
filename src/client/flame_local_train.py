"""Joint FLAME local update: MAE + prototypical + classification learning."""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.utils.metrics import evaluate


def _images(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _safe_global_encoder_state(global_weights):
    if isinstance(global_weights, dict) and "encoder" in global_weights:
        return global_weights["encoder"]
    return global_weights


def _split_data_loader(loader: DataLoader):
    for batch in loader:
        yield batch


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
    """Run local MAE, prototype, and classifier training for one federated round."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    loss_cfg = getattr(config, "loss", None)
    lambda_mae = float(getattr(loss_cfg, "lambda_mae", 0.5) if loss_cfg is not None else 0.5)
    lambda_proto = float(getattr(loss_cfg, "lambda_proto", 0.25) if loss_cfg is not None else 0.25)
    lambda_ce = float(getattr(loss_cfg, "lambda_ce", 0.25) if loss_cfg is not None else 0.25)

    optimizer = AdamW(model.parameters(), lr=float(config.ssl.lr), weight_decay=0.05, betas=(0.9, 0.95))
    is_fedprox = bool(getattr(config.federated, "aggregation", "fedavg").lower() == "fedprox" and global_weights is not None)
    mu = float(getattr(config.federated, "fedprox_mu", 0.01))
    global_encoder_state = _safe_global_encoder_state(global_weights)

    support_images, support_labels = next(iter(support_loader))
    support_images = support_images.to(device)
    support_labels = torch.as_tensor(support_labels, device=device).long()

    supervised_data = list(_split_data_loader(proto_train_query_loader))
    if not supervised_data:
        raise ValueError(f"Hospital {hospital_id} has no supervised train-query data.")
    batch_cycle = []
    for batch in supervised_data:
        images, labels = batch
        batch_cycle.append((images.to(device), torch.as_tensor(labels, device=device).long()))

    epoch_records = []
    for _ in range(int(config.ssl.epochs_per_round)):
        mae_total = proto_total = ce_total = prox_total = total_total = 0.0
        batches = 0
        query_index = 0

        for unlabeled_batch in unlabeled_loader:
            images = _images(unlabeled_batch).to(device)
            optimizer.zero_grad()
            mae_loss, _, _ = model(images)

            query_images, query_labels = batch_cycle[query_index % len(batch_cycle)]
            query_index += 1
            query_embeddings = model.encoder(query_images)
            prototypes = model.proto_head.compute_prototypes(support_images, support_labels)
            proto_loss, _ = model.proto_head.prototypical_loss(query_embeddings, query_labels, prototypes)

            classifier_logits = model.classifier_head(query_embeddings).squeeze(-1)
            ce_loss = F.binary_cross_entropy_with_logits(classifier_logits, query_labels.float())

            prox_loss = torch.zeros((), device=device)
            if is_fedprox and isinstance(global_encoder_state, dict):
                for name, param in model.encoder.named_parameters():
                    if name in global_encoder_state:
                        prox_loss = prox_loss + torch.sum((param - global_encoder_state[name].to(device)) ** 2)
                prox_loss = (mu / 2.0) * prox_loss

            total_loss = lambda_mae * mae_loss + lambda_proto * proto_loss + lambda_ce * ce_loss + prox_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            mae_total += float(mae_loss.detach())
            proto_total += float(proto_loss.detach())
            ce_total += float(ce_loss.detach())
            prox_total += float(prox_loss.detach())
            total_total += float(total_loss.detach())
            batches += 1

        divisor = max(batches, 1)
        epoch_records.append({
            "mae_loss": mae_total / divisor,
            "proto_loss": proto_total / divisor,
            "classification_loss": ce_total / divisor,
            "fedprox_loss": prox_total / divisor,
            "total_loss": total_total / divisor,
        })

    model.eval()
    eval_probabilities, eval_labels = [], []
    with torch.no_grad():
        support_embedding = model.encoder(support_images)
        prototypes = model.proto_head.compute_prototypes(support_embedding, support_labels)
        for images, labels in eval_query_loader:
            img = images.to(device)
            query_embedding = model.encoder(img)
            logits = model.classifier_head(query_embedding).squeeze(-1)
            proto_logits = model.proto_head.forward(query_embedding, prototypes)
            classifier_prob = torch.sigmoid(logits)
            prototype_prob = torch.softmax(proto_logits, dim=-1)[:, 1]
            combined_prob = 0.5 * (classifier_prob + prototype_prob)
            eval_probabilities.append(combined_prob.cpu())
            eval_labels.append(torch.as_tensor(labels, device="cpu", dtype=torch.long))

    eval_metrics = evaluate(
        torch.cat(eval_labels).numpy(), torch.cat(eval_probabilities).numpy()
    )

    final = epoch_records[-1]
    print(f"[Hospital {hospital_id}] MAE={final['mae_loss']:.4f} | Proto={final['proto_loss']:.4f} | CE={final['classification_loss']:.4f} | FedProx={final['fedprox_loss']:.4f} | Total={final['total_loss']:.4f}")

    return {
        "model_weights": model.get_federated_weights(),
        "encoder_weights": model.get_encoder_weights(),
        "num_samples": len(unlabeled_loader.dataset),
        "epoch_losses": epoch_records,
        "query_metrics": eval_metrics,
        "mae_loss": final["mae_loss"],
        "proto_loss": final["proto_loss"],
        "classification_loss": final["classification_loss"],
        "fedprox_loss": final["fedprox_loss"],
        "total_loss": final["total_loss"],
    }
