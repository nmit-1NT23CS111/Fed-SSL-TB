"""Focused checks for the federated FLAME contracts."""
import torch
import unittest
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from src.client.flame_local_train import flame_local_train, joint_loss
from src.client.local_train import split_client_support_train_eval_indices
from src.models.proto_head import PrototypicalHead
from src.server.aggregator import fedavg


class FlameArchitectureTests(unittest.TestCase):
    def test_joint_loss_weights(self):
        self.assertTrue(torch.isclose(joint_loss(torch.tensor(2.0), torch.tensor(1.0)), torch.tensor(1.7)))


    def test_proto_gradient_reaches_encoder(self):
        encoder = torch.nn.Linear(4, 3)
        head = torch.nn.Linear(3, 2)
        features = encoder(torch.randn(4, 4))
        loss = torch.nn.functional.cross_entropy(head(features), torch.tensor([0, 1, 0, 1]))
        loss.backward()
        self.assertIsNotNone(encoder.weight.grad)
        self.assertGreater(encoder.weight.grad.abs().sum(), 0)


    def test_full_model_aggregation(self):
        local = [
        {"encoder": {"weight": torch.tensor([1.0])}, "decoder": {"weight": torch.tensor([3.0])}, "proto_head": {"weight": torch.tensor([5.0])}},
        {"encoder": {"weight": torch.tensor([3.0])}, "decoder": {"weight": torch.tensor([5.0])}, "proto_head": {"weight": torch.tensor([7.0])}},
    ]
        result = fedavg(local, [1, 3])
        self.assertEqual(set(result), {"encoder", "decoder", "proto_head"})
        self.assertEqual(result["encoder"]["weight"].item(), 2.5)

    def test_client_support_train_and_eval_splits_are_disjoint_for_k_values(self):
        labels = torch.tensor([0] * 30 + [1] * 30)

        for k in (1, 2, 5):
            with self.subTest(k=k):
                client_splits = split_client_support_train_eval_indices(
                    labels, num_clients=3, k=k, seed=21
                )
                all_eval_indices = set()
                all_training_indices = set()
                for support, train_query, eval_query in client_splits:
                    support_set = set(support.tolist())
                    train_query_set = set(train_query.tolist())
                    eval_query_set = set(eval_query.tolist())
                    self.assertFalse(support_set & train_query_set)
                    self.assertFalse(support_set & eval_query_set)
                    self.assertFalse(train_query_set & eval_query_set)
                    self.assertEqual(len(support), 2 * k)
                    self.assertEqual(len(train_query), 2)
                    self.assertEqual(len(eval_query), 2)
                    self.assertEqual(set(labels[support].tolist()), {0, 1})
                    self.assertEqual(set(labels[train_query].tolist()), {0, 1})
                    self.assertEqual(set(labels[eval_query].tolist()), {0, 1})
                    self.assertFalse(all_eval_indices & eval_query_set)
                    all_eval_indices.update(eval_query_set)
                    all_training_indices.update(support_set | train_query_set)
                self.assertFalse(all_eval_indices & all_training_indices)

    def test_eval_query_contents_do_not_change_local_training(self):
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
                return {
                    "encoder": self.encoder.state_dict(),
                    "decoder": self.decoder.state_dict(),
                    "proto_head": self.proto_head.state_dict(),
                }

        def run_with_eval_images(eval_images):
            model = TinyJointModel()
            initial_encoder = {
                name: value.detach().clone()
                for name, value in model.encoder.state_dict().items()
            }
            nih_images = torch.randn(4, 4)
            support_images = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.8, 0.0, 0.0],
            ])
            support_labels = torch.tensor([0, 0, 1, 1])
            train_query_images = torch.tensor([
                [0.9, 0.0, 0.1, 0.0],
                [0.0, 0.9, 0.1, 0.0],
            ])
            train_query_labels = torch.tensor([0, 1])
            eval_labels = torch.tensor([0, 1])
            result = flame_local_train(
                hospital_id=1,
                model=model,
                unlabeled_loader=DataLoader(TensorDataset(nih_images), batch_size=4),
                support_loader=DataLoader(
                    TensorDataset(support_images, support_labels), batch_size=4
                ),
                proto_train_query_loader=DataLoader(
                    TensorDataset(train_query_images, train_query_labels), batch_size=2
                ),
                eval_query_loader=DataLoader(
                    TensorDataset(eval_images, eval_labels), batch_size=2
                ),
                config=SimpleNamespace(
                    ssl=SimpleNamespace(lr=1e-2, epochs_per_round=1),
                    federated=SimpleNamespace(aggregation="fedavg"),
                    finetuning=SimpleNamespace(alpha=0.7),
                ),
                device=torch.device("cpu"),
            )
            self.assertTrue(any(
                not torch.equal(initial_encoder[name], value)
                for name, value in model.encoder.state_dict().items()
            ))
            self.assertIn("mae_loss", result)
            self.assertIn("proto_loss", result)
            return model.state_dict()

        torch.manual_seed(17)
        first_eval = torch.tensor([[0.2, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0]])
        second_eval = torch.tensor([[9.0, 0.0, 0.0, 0.0], [0.0, 9.0, 0.0, 0.0]])
        first_state = run_with_eval_images(first_eval)
        torch.manual_seed(17)
        second_state = run_with_eval_images(second_eval)
        for name in first_state:
            self.assertTrue(torch.equal(first_state[name], second_state[name]), name)
