import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from incident_embeddings.cli import load_model
from incident_embeddings.data import ClassBatchSampler, collate, read_records, forest_levels
from incident_embeddings.model import IncidentEncoder, ModelConfig
from incident_embeddings.training import retrieval_metrics, supervised_contrastive_loss
from dataclasses import asdict


def record(i=0, label="a"):
    return {"incident_id": str(i), "equivalence_id": label, "graphs": [
        {"node_features": [[1., 0.], [0., 2.], [3., 1.]], "edges": [[0, 1], [1, 2]], "weight": 0.7},
        {"node_features": [[2., 1.]], "edges": [], "weight": 0.3}]}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(1)
        self.model = IncidentEncoder(ModelConfig(2, 16, 8, 2))

    def test_node_and_set_permutation(self):
        a = record()
        b = copy.deepcopy(a)
        # New order [2, 0, 1], with endpoints relabeled consistently.
        b["graphs"][0]["node_features"] = [a["graphs"][0]["node_features"][i] for i in [2, 0, 1]]
        b["graphs"][0]["edges"] = [[1, 2], [2, 0]]
        b["graphs"].reverse()
        for pool in ["deepsets", "attention"]:
            model = IncidentEncoder(ModelConfig(2, 16, 8, 2, pool)).eval()
            torch.testing.assert_close(model(collate([a])), model(collate([b])), atol=1e-6, rtol=1e-5)

    def test_batch_independence_and_direction(self):
        a, b = record(), record(1)
        b["graphs"][0]["edges"] = [[1, 0], [2, 1]]
        single = self.model(collate([a]))
        batched = self.model(collate([a, b]))
        torch.testing.assert_close(single[0], batched[0])
        self.assertFalse(torch.allclose(batched[0], batched[1]))

    def test_loss_and_gradients_reach_both_models(self):
        examples = [record(i, str(i // 2)) for i in range(4)]
        examples[2]["graphs"][0]["node_features"][0] = [8., -3.]
        examples[3]["graphs"][0]["node_features"][0] = [7., -2.]
        loss = supervised_contrastive_loss(self.model(collate(examples)), torch.tensor([0, 0, 1, 1]))
        loss.backward()
        for module in [self.model.graph_encoder, self.model.set_encoder]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        with self.assertRaises(ValueError):
            supervised_contrastive_loss(torch.randn(3, 8), torch.arange(3))

    def test_sampler_distinct_positives(self):
        examples = [record(i, str(i // 3)) for i in range(6)]
        indices = ClassBatchSampler(examples, 2, 2, 1).sample()
        self.assertEqual(len(set(indices)), 4)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model_config": asdict(self.model.config), "model": self.model.state_dict()}, path)
            other = load_model(path, "cpu")
            torch.testing.assert_close(self.model(collate([record()])), other(collate([record()])))

    def test_retrieval_excludes_self_and_reports_coverage(self):
        gallery = [record(0, "a"), record(1, "b"), record(2, "a")]
        queries = [record(0, "a"), record(3, "missing")]
        metrics = retrieval_metrics(torch.tensor([[1., 0.], [0., 1.]]), queries,
                                    torch.tensor([[1., 0.], [0., 1.], [.9, .1]]), gallery)
        self.assertEqual(metrics["hit_at_1"], 1.)
        self.assertEqual(metrics["query_coverage"], .5)

    def test_zero_weight_attention(self):
        sample = record()
        sample["graphs"][0]["weight"] = 0.
        model = IncidentEncoder(ModelConfig(2, 16, 8, 2, "attention"))
        self.assertTrue(torch.isfinite(model(collate([sample]))).all())

    def test_schema_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            good = record()
            path.write_text(json.dumps(good) + "\n")
            records, dimension = read_records(path)
            self.assertEqual(dimension, 2)
            self.assertEqual(len(records), 1)
            bad_cases = []
            bad = copy.deepcopy(good)
            bad["graphs"][0]["edges"] = [[0, 99]]
            bad_cases.append(bad)
            bad = copy.deepcopy(good)
            bad["graphs"][0]["node_features"][0] = [float("nan"), 1.]
            bad_cases.append(bad)
            bad = copy.deepcopy(good)
            for graph in bad["graphs"]:
                graph["weight"] = 0.
            bad_cases.append(bad)
            for bad in bad_cases:
                path.write_text(json.dumps(bad) + "\n")
                with self.assertRaises(ValueError):
                    read_records(path)

    def test_separable_batch_can_be_learned(self):
        samples = [record(i, str(i // 2)) for i in range(4)]
        for sample in samples[2:]:
            for graph in sample["graphs"]:
                graph["node_features"] = [[-v - 2 for v in row] for row in graph["node_features"]]
        batch = collate(samples)
        labels = torch.tensor([0, 0, 1, 1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        initial = supervised_contrastive_loss(self.model(batch), labels).item()
        for _ in range(20):
            optimizer.zero_grad()
            loss = supervised_contrastive_loss(self.model(batch), labels)
            loss.backward()
            optimizer.step()
        final = supervised_contrastive_loss(self.model(batch), labels).item()
        self.assertLess(final, initial)

    def test_reject_non_forests(self):
        for edges in [[[0, 1], [1, 0]], [[0, 2], [1, 2]], [[0, 1], [0, 1]], [[0, 0]]]:
            with self.assertRaises(ValueError):
                forest_levels(3, edges)

    def test_forest_and_unequal_branch_depths(self):
        sample = {"incident_id": "forest", "graphs": [{
            "node_features": [[float(i), 1.] for i in range(6)],
            "edges": [[0, 1], [0, 2], [2, 3], [3, 4]]}]}
        encoder = self.model.graph_encoder
        batch = collate([sample])
        actual = encoder(batch)
        x = encoder.input(batch["x"])
        children = {0: [1, 2], 2: [3], 3: [4]}
        def visit(node):
            descendants = children.get(node, [])
            total = sum((encoder.child_message(visit(child)) for child in descendants), torch.zeros_like(x[node]))
            return encoder.update(x[node][None], total[None])[0]
        h = torch.stack([visit(i) for i in range(6)])
        root_sum = h[[0, 5]].sum(0)
        expected = encoder.readout(torch.cat([root_sum, root_sum / 2, h.mean(0),
                                               torch.tensor([7., 3.]).log()])[None])
        torch.testing.assert_close(actual, expected)

    def test_deep_leaf_reaches_root_through_autograd(self):
        sample = {"incident_id": "chain", "graphs": [{
            "node_features": [[1., float(i)] for i in range(9)],
            "edges": [[i, i + 1] for i in range(8)]}]}
        batch = collate([sample])
        batch["x"].requires_grad_(True)
        states = []
        handle = self.model.graph_encoder.update.register_forward_hook(lambda module, args, output: states.append(output))
        try:
            self.model.graph_encoder(batch)
        finally:
            handle.remove()
        self.assertEqual(len(states), 9)
        states[-1].sum().backward()
        self.assertGreater(batch["x"].grad[-1].abs().sum().item(), 0.)

    def test_optimizer_updates_both_encoders(self):
        samples = [record(i, str(i // 2)) for i in range(4)]
        samples[2]["graphs"][0]["node_features"][0] = [-8., 2.]
        samples[3]["graphs"][0]["node_features"][0] = [-7., 3.]
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        loss = supervised_contrastive_loss(self.model(collate(samples)), torch.tensor([0, 0, 1, 1]))
        loss.backward()
        optimizer.step()
        for prefix in ["graph_encoder.", "set_encoder."]:
            self.assertTrue(any(not torch.equal(before[name], p) for name, p in self.model.named_parameters() if name.startswith(prefix)))

    def test_legacy_checkpoint(self):
        model = IncidentEncoder(ModelConfig(2, 16, 8, 2, encoder="directed"))
        config = asdict(model.config)
        del config["encoder"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({"model_config": config, "model": model.state_dict()}, path)
            restored = load_model(path, "cpu")
            torch.testing.assert_close(model(collate([record()])), restored(collate([record()])))


if __name__ == "__main__":
    unittest.main()
