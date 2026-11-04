from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    node_dim: int
    hidden_dim: int = 128
    embedding_dim: int = 128
    graph_layers: int = 3
    set_pool: str = "deepsets"
    encoder: str = "tree"


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                         nn.Linear(hidden_dim, output_dim))


class DirectedLayer(nn.Module):
    """Separate incoming/outgoing sums preserve edge orientation."""

    def __init__(self, width):
        super().__init__()
        self.incoming = nn.Linear(width, width, bias=False)
        self.outgoing = nn.Linear(width, width, bias=False)
        self.update = mlp(3 * width, width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x, edges):
        src, dst = edges
        incoming, outgoing = torch.zeros_like(x), torch.zeros_like(x)
        incoming.index_add_(0, dst, self.incoming(x[src]))
        outgoing.index_add_(0, src, self.outgoing(x[dst]))
        return self.norm(x + self.update(torch.cat([x, incoming, outgoing], -1)))


class GraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.input = nn.Linear(config.node_dim, d)
        self.layers = nn.ModuleList([DirectedLayer(d) for _ in range(config.graph_layers)])
        self.readout = mlp(2 * d + 1, d, config.embedding_dim)

    def forward(self, batch):
        x = self.input(batch["x"])
        for layer in self.layers:
            x = layer(x, batch["edges"])
        count = batch["graph_count"]
        sums = x.new_zeros(count, x.shape[-1])
        sums.index_add_(0, batch["node_graph"], x)
        sizes = torch.bincount(batch["node_graph"], minlength=count).to(x.dtype).unsqueeze(-1)
        return self.readout(torch.cat([sums, sums / sizes, sizes.log1p()], -1))


class TreeEncoder(nn.Module):
    """Shared bottom-up recurrence over all depths; no fixed hop cutoff.

    Child order is immaterial. Forest roots are pooled within each explanation.
    This is a custom gated tree encoder, not a TreeLSTM or a causal simulator.
    """

    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.input = nn.Linear(config.node_dim, d)
        self.child_message = mlp(d, d, d)
        self.update = nn.GRUCell(d, d)
        self.readout = mlp(3 * d + 2, d, config.embedding_dim)

    def forward(self, batch):
        inputs = self.input(batch["x"])
        states, node_order = [], []
        for nodes, groups in batch["tree_schedule"]:
            child_sum = inputs.new_zeros(len(nodes), inputs.shape[-1])
            for child_level, child_positions, parent_positions in groups:
                messages = self.child_message(states[child_level][child_positions])
                child_sum = child_sum.index_add(0, parent_positions, messages)
            states.append(self.update(inputs[nodes], child_sum))
            node_order.append(nodes)
        # Reassemble node-index order while retaining the full autograd graph.
        order = torch.cat(node_order)
        h = torch.cat(states)[torch.argsort(order)]
        count = batch["graph_count"]
        sums = h.new_zeros(count, h.shape[-1]).index_add(0, batch["node_graph"], h)
        sizes = torch.bincount(batch["node_graph"], minlength=count).to(h.dtype)[:, None]
        roots = batch["roots"]
        root_graph = batch["node_graph"][roots]
        root_sum = h.new_zeros(count, h.shape[-1]).index_add(0, root_graph, h[roots])
        root_count = torch.bincount(root_graph, minlength=count).to(h.dtype)[:, None]
        return self.readout(torch.cat([root_sum, root_sum / root_count, sums / sizes,
                                       sizes.log1p(), root_count.log1p()], -1))


class SetEncoder(nn.Module):
    """Weighted DeepSets, optionally with learned scalar attention."""

    def __init__(self, config):
        super().__init__()
        d, h = config.embedding_dim, config.hidden_dim
        self.phi = mlp(d, h, h)
        self.score = nn.Linear(h, 1) if config.set_pool == "attention" else None
        self.rho = mlp(h, h, d)

    def forward(self, graphs, batch):
        features = self.phi(graphs)
        pooled = []
        for i in range(batch["incident_count"]):
            mask = batch["graph_incident"] == i
            values, weights = features[mask], batch["weights"][mask]
            if self.score is not None:
                logits = self.score(values).squeeze(-1) + weights.log()
                weights = logits.softmax(0)
            else:
                weights = weights / weights.sum()
            pooled.append((values * weights[:, None]).sum(0))
        return F.normalize(self.rho(torch.stack(pooled)), dim=-1)


class IncidentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if min(config.node_dim, config.hidden_dim, config.embedding_dim, config.graph_layers) < 1:
            raise ValueError("Model dimensions and graph_layers must be positive")
        if config.set_pool not in {"deepsets", "attention"}:
            raise ValueError("set_pool must be deepsets or attention")
        if config.encoder not in {"tree", "directed"}:
            raise ValueError("encoder must be tree or directed")
        self.config = config
        self.graph_encoder = TreeEncoder(config) if config.encoder == "tree" else GraphEncoder(config)
        self.set_encoder = SetEncoder(config)

    def forward(self, batch):
        return self.set_encoder(self.graph_encoder(batch), batch)
