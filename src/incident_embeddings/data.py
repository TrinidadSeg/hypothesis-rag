import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch


def forest_levels(node_count, edges):
    """Validate parent->child forest and group nodes by height above leaves."""
    parents = [-1] * node_count
    remaining = [0] * node_count
    heights = [0] * node_count
    for parent, child in edges:
        if not (0 <= parent < node_count and 0 <= child < node_count):
            raise ValueError("Tree edge endpoint out of range")
        if parents[child] != -1:
            raise ValueError("Trees require at most one parent per node; duplicate edges are invalid")
        parents[child] = parent
        remaining[parent] += 1
    queue = [v for v in range(node_count) if remaining[v] == 0]
    for child in queue:
        parent = parents[child]
        if parent != -1:
            heights[parent] = max(heights[parent], heights[child] + 1)
            remaining[parent] -= 1
            if remaining[parent] == 0:
                queue.append(parent)
    if len(queue) != node_count:
        raise ValueError("Explanation must be an acyclic directed tree or forest")
    return heights, [v for v, parent in enumerate(parents) if parent == -1]


def read_records(path, node_dim=None, labeled=True):
    records, ids = [], set()
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            prefix = f"{path}:{line_number}"
            incident_id = record.get("incident_id")
            if not isinstance(incident_id, str) or not incident_id or incident_id in ids:
                raise ValueError(f"{prefix}: incident_id must be a unique nonempty string")
            ids.add(incident_id)
            if labeled and (not isinstance(record.get("equivalence_id"), str) or not record["equivalence_id"]):
                raise ValueError(f"{prefix}: equivalence_id must be a nonempty string")
            graphs = record.get("graphs")
            if not isinstance(graphs, list) or not graphs:
                raise ValueError(f"{prefix}: graphs must be nonempty")
            for graph in graphs:
                x = graph.get("node_features")
                if not isinstance(x, list) or not x or not isinstance(x[0], list) or not x[0]:
                    raise ValueError(f"{prefix}: node_features must be a nonempty matrix")
                if node_dim is None:
                    node_dim = len(x[0])
                if any(not isinstance(row, list) or len(row) != node_dim or
                       any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in row)
                       for row in x):
                    raise ValueError(f"{prefix}: inconsistent or nonfinite node features")
                edges = graph.get("edges")
                if not isinstance(edges, list) or any(
                    not isinstance(e, list) or len(e) != 2 or
                    any(type(v) is not int or not 0 <= v < len(x) for v in e) for e in edges
                ):
                    raise ValueError(f"{prefix}: edges must be [source, target] index pairs")
                try:
                    forest_levels(len(x), edges)
                except ValueError as error:
                    raise ValueError(f"{prefix}: {error}") from error
                weight = graph.get("weight", 1.0)
                if not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight < 0:
                    raise ValueError(f"{prefix}: weights must be finite and nonnegative")
            if sum(g.get("weight", 1.0) for g in graphs) <= 0:
                raise ValueError(f"{prefix}: at least one graph must have positive weight")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records, node_dim


def collate(records, device="cpu"):
    xs, edges, node_graph, graph_incident, weights = [], [], [], [], []
    offset = 0
    heights, roots = [], []
    for incident_index, record in enumerate(records):
        for graph in record["graphs"]:
            x = torch.tensor(graph["node_features"], dtype=torch.float32)
            edge = torch.tensor(graph["edges"], dtype=torch.long).reshape(-1, 2).T
            xs.append(x)
            edges.append(edge + offset)
            local_heights, local_roots = forest_levels(len(x), graph["edges"])
            heights.extend(local_heights)
            roots.extend(offset + v for v in local_roots)
            node_graph.extend([len(weights)] * len(x))
            graph_incident.append(incident_index)
            weights.append(graph.get("weight", 1.0))
            offset += len(x)
    batch = {
        "x": torch.cat(xs), "edges": torch.cat(edges, dim=1),
        "node_graph": torch.tensor(node_graph, dtype=torch.long),
        "graph_incident": torch.tensor(graph_incident, dtype=torch.long),
        "weights": torch.tensor(weights, dtype=torch.float32),
        "graph_count": len(weights), "incident_count": len(records),
        "roots": torch.tensor(roots, dtype=torch.long),
    }
    # Precompute execution groups on CPU. Children may lie at different heights.
    nodes_by_level = [[] for _ in range(max(heights) + 1)]
    positions = [0] * offset
    for node, height in enumerate(heights):
        positions[node] = len(nodes_by_level[height])
        nodes_by_level[height].append(node)
    links = [defaultdict(lambda: ([], [])) for _ in nodes_by_level]
    for parent, child in batch["edges"].T.tolist():
        child_positions, parent_positions = links[heights[parent]][heights[child]]
        child_positions.append(positions[child])
        parent_positions.append(positions[parent])
    batch["tree_schedule"] = [
        (torch.tensor(nodes, dtype=torch.long, device=device), [
            (child_level, torch.tensor(children, dtype=torch.long, device=device),
             torch.tensor(parents, dtype=torch.long, device=device))
            for child_level, (children, parents) in groups.items()])
        for nodes, groups in zip(nodes_by_level, links)
    ]
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class ClassBatchSampler:
    """P equivalence classes, K distinct incidents per class; no self-positive copies."""

    def __init__(self, records, classes_per_batch, samples_per_class, seed):
        self.groups = defaultdict(list)
        for i, record in enumerate(records):
            self.groups[record["equivalence_id"]].append(i)
        self.p, self.k = classes_per_batch, samples_per_class
        if self.p < 2 or self.k < 2:
            raise ValueError("Need >=2 classes per batch and >=2 samples per class")
        small = {c: len(v) for c, v in self.groups.items() if len(v) < self.k}
        if small:
            raise ValueError(f"Classes need >= {self.k} distinct training incidents: {small}")
        if len(self.groups) < self.p:
            raise ValueError("Too few equivalence classes for classes_per_batch")
        self.rng = random.Random(seed)

    def sample(self):
        return [i for c in self.rng.sample(list(self.groups), self.p)
                for i in self.rng.sample(self.groups[c], self.k)]
