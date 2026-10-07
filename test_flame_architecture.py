"""Architecture contract tests for the encoder-only FedSSL pipeline."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.client.flame_local_train import flame_local_train
from src.client.local_train import split_client_support_train_eval_indices
from src.models.mae import build_mae
from src.models.proto_head import PrototypicalHead
from src.server.aggregator import fedavg


class FlameArchitectureTests(unittest.TestCase):
    def test_vit_tiny_encoder_output_dim_is_192(self):
        model = build_mae(backbone="vit_tiny", embed_dim=192, decoder_depth=4)
        x = torch.randn(2, 3, 224, 224)
        embedding = model.get_embedding(x)
        self.assertEqual(tuple(embedding.shape), (2, 192))

    def test_support_images_are_encoded_before_prototype_construction(self):
        encoder = torch.nn.Linear(4, 3)
        support_images = torch.randn(4, 4)
        support_labels = torch.tensor([0, 0, 1, 1])
        head = PrototypicalHead(embed_dim=3, num_classes=2, projection_dim=2)
        support_embeddings = encoder(support_images)
        prototypes = head.compute_prototypes(support_embeddings, support_labels)
        self.assertEqual(tuple(prototypes.shape), (2, 2))

    def test_proto_loss_backpropagates_into_encoder(self):
        encoder = torch.nn.Linear(4, 3)
        head = PrototypicalHead(embed_dim=3, num_classes=2, projection_dim=2)
        support_images = torch.randn(4, 4)
        support_labels = torch.tensor([0, 0, 1, 1])
        query_images = torch.randn(4, 4)
        query_labels = torch.tensor([0, 1, 0, 1])

        support_embeddings = encoder(support_images)
        prototypes = head.compute_prototypes(support_embeddings, support_labels)
        query_embeddings = encoder(query_images)
        loss, _ = head.prototypical_loss(query_embeddings, query_labels, prototypes)
        loss.backward()

        self.assertIsNotNone(encoder.weight.grad)
        self.assertGreater(torch.abs(encoder.weight.grad).sum().item(), 0.0)

    def test_fedavg_only_returns_encoder_payload(self):
        local = [
            {"encoder": {"weight": torch.tensor([1.0])}},
            {"encoder": {"weight": torch.tensor([3.0])}},
        ]
        result = fedavg(local, [1, 3])
        self.assertEqual(set(result), {"encoder"})
        self.assertEqual(result["encoder"]["weight"].item(), 2.5)

    def test_client_support_train_and_eval_splits_are_disjoint(self):
        labels = torch.tensor([0] * 30 + [1] * 30)
        for k in (1, 2, 5):
            with self.subTest(k=k):
                splits = split_client_support_train_eval_indices(labels, num_clients=3, k=k, seed=21, train_queries_per_class=2)
                all_eval = set()
                all_train = set()
                for support, train_query, eval_query in splits:
                    support_set = set(support.tolist())
                    train_query_set = set(train_query.tolist())
                    eval_query_set = set(eval_query.tolist())
                    self.assertFalse(support_set & train_query_set)
                    self.assertFalse(support_set & eval_query_set)
                    self.assertFalse(train_query_set & eval_query_set)
                    self.assertEqual(len(support), 2 * k)
                    self.assertEqual(len(train_query), 2 * 2)
                    self.assertEqual(len(eval_query), 2)
                    self.assertFalse(all_eval & eval_query_set)
                    all_eval.update(eval_query_set)
                    all_train.update(support_set | train_query_set)
                self.assertFalse(all_eval & all_train)

    def test_flame_local_train_uses_encoded_support_and_no_classifier_branch(self):
        class TinyJointModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(4, 3)
                self.decoder = torch.nn.Linear(3, 4)
                self.proto_head = PrototypicalHead(3, 2, projection_dim=2)

            def forward(self, images):
                reconstruction = self.decoder(self.encoder(images))
                return F.mse_loss(reconstruction, images), reconstruction, None

            def get_federated_weights(self):
                return {"encoder": self.encoder.state_dict()}

        nih_images = torch.randn(8, 4)
        support_images = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
        ])
        support_labels = torch.tensor([0, 0, 1, 1])
        query_images = torch.tensor([
            [0.8, 0.2, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
        ])
        query_labels = torch.tensor([0, 1])
        config = SimpleNamespace(
            ssl=SimpleNamespace(lr=1e-2, epochs_per_round=1),
            federated=SimpleNamespace(aggregation="fedavg", fedprox_mu=0.01),
            loss=SimpleNamespace(lambda_mae=0.70, lambda_proto=0.30),
        )

        result = flame_local_train(
            hospital_id=1,
            model=TinyJointModel(),
            unlabeled_loader=DataLoader(TensorDataset(nih_images), batch_size=4),
            support_loader=DataLoader(TensorDataset(support_images, support_labels), batch_size=4),
            proto_train_query_loader=DataLoader(TensorDataset(query_images, query_labels), batch_size=2),
            eval_query_loader=DataLoader(TensorDataset(query_images, query_labels), batch_size=2),
            config=config,
            device=torch.device("cpu"),
        )

        self.assertIn("encoder_weights", result)
        self.assertNotIn("classifier", result)
        self.assertIn("mae_loss", result)
        self.assertIn("proto_loss", result)
        self.assertEqual(set(result["encoder_weights"].keys()), set(TinyJointModel().encoder.state_dict().keys()))


if __name__ == "__main__":
    unittest.main()
