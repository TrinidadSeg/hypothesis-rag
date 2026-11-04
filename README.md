# Incident explanation embeddings

Train **two neural models jointly**: a bottom-up tree encoder maps each explanation tree/forest to a vector; a permutation-invariant set encoder maps an incident's explanation collection to a unit-normalized retrieval vector. Incident equivalence IDs supervise both models through a supervised contrastive loss. This package does not compute equivalent splits or generate explanations.

## Version 0.2 architecture

```text
Explanation trees/forests
    -> shared bottom-up tree encoder (leaves to roots)
    -> one embedding per explanation
    -> weighted DeepSets or attention pooling
    -> one normalized incident embedding
    -> supervised contrastive loss using equivalence IDs
    -> backward() through BOTH encoders
    -> optimizer.step() updates BOTH encoders
```

For node v, first project its features to x_v. Recursively sum learned child messages, c_v = sum(phi(h_child)), then compute h_v = GRUCell(x_v, c_v). Leaves use a zero child sum. The GRU is a shared gated update over tree structure, not over incident time; this is a custom tree encoder, not a TreeLSTM or a causal simulator. Nodes at the same height across the batch are processed together. No fixed hop cutoff is imposed, although deep trees can still suffer vanishing gradients and compressed subtree information.

Readout combines the sum and mean of root states, mean of all node states, log node count, and log root count. Multiple roots support forests left after splitting. Roots summarize descendants; the node mean supplies a shorter gradient path and direct access to local evidence. Root sum and mean are redundant for single-root trees. Counts may become shortcuts and should be ablated.

Siblings are treated as unordered. If split branches have semantic roles, put these roles in the child node features. Edge attributes and split histories are not automatically consumed. The model learns supplied equivalences statistically; it does not guarantee exact invariance to them.

The graph embeddings remain in PyTorch's autograd computation graph during training: no detach, NumPy conversion, or export occurs between encoders. A single optimizer owns all parameters. Embedding export is a separate inference operation after training.

Use `--encoder tree` (default) for the new encoder. `--encoder directed` retains the previous message-passing baseline. `--graph-layers` controls only that baseline; tree depth comes from the input. Old 0.1 checkpoints load as directed models, but both input paths now require trees/forests. To use the new architecture, train a fresh checkpoint and regenerate historical vectors.

## Install

Python 3.10+ and PyTorch 2.2+ are required. On your training machine, install the PyTorch build appropriate for its GPU/CUDA environment, then:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The implementation uses native PyTorch scatter operations, with no PyTorch Geometric dependency. It uses one device, float32, variable-size disjoint graph batches, and in-memory JSONL data. A 512 GB GPU needs no special code path. Capacity does not imply all data should be one contrastive batch: batch class composition changes the learning objective. Distributed training, automatic resume, and mixed precision are not implemented.

## Data contract: JSONL, one incident per line

This pretty-printed example represents **one line** in the actual file:

```json
{
  "incident_id": "game-2026-10-01-incident-017",
  "equivalence_id": "opaque-equivalence-class-42",
  "leakage_group": "original-incident-017",
  "graphs": [
    {
      "node_features": [[1.0, 0.0, 0.3], [0.0, 1.0, 0.8]],
      "edges": [[0, 1]],
      "weight": 0.8
    },
    {
      "node_features": [[1.0, 0.0, 0.6]],
      "edges": [],
      "weight": 0.2
    }
  ]
}
```

- `incident_id`: unique nonempty string. Training and validation IDs must be disjoint.
- `equivalence_id`: nonempty string supplied by your coworker's equivalence pipeline. Same ID means a positive pair; different IDs mean negatives. Omitted for embedding export. Unknown or unlabeled pairs are **not supported** as negatives: do not assign each uncertain incident a fresh class and train on that assumption.
- `leakage_group`: optional string identifying all variants/windows/augmentations of one source incident. If provided, its values must not cross training/validation partitions. This is different from equivalence class: equivalent but independently occurring incidents should be present in both query and historical gallery data.
- `graphs`: nonempty list of candidate explanation graphs. Deduplicate semantically equivalent explanations upstream.
- `node_features`: nonempty `[nodes, features]` finite numeric matrix. All graphs use the same feature schema and dimension. Supply meaningful features such as component type, anomaly evidence, observed-failure indicator, temporal lag, and candidate-cause indicator. These are application choices, not inferred by the package. Normalize numeric features using training statistics and reuse those statistics at inference.
- `edges`: list of directed `[parent, child]` pairs, zero-based local node indices. `[]` is valid, including isolated nodes. Each node has at most one parent. Cycles, self-loops, duplicate edges, and multiple-parent nodes are rejected. Multiple roots are allowed within one explanation, representing a forest. If your trees use child-to-parent orientation, reverse the edges upstream. General DAGs require a different encoder/contract.
- `weight`: optional nonnegative finite evidence weight, default 1. At least one weight per incident must be positive. Weights are normalized within the incident and need not be probabilities.

**Adapter boundary:** convert your split-derived explanation into the node-feature/edge representation, then attach your externally computed incident equivalence ID. If your coworker's labels describe individual splits rather than whole incidents, you must first define the incident-level positive relation. The package deliberately does not invent that mapping. Both encoders receive incident-level supervision; there is no separate graph-level label requirement or graph pretraining objective.

No edge features are consumed. Encode essential mechanism/lag information in node features or extend the graph model explicitly; adding an unused JSON field will not teach the model about it. Never include equivalence IDs, known postmortem root causes, or successful remediations as input features unless those are actually available at retrieval time.

## Train

```bash
incident-embeddings train \
  --train /data/train.jsonl --val /data/val.jsonl \
  --output runs/deepsets-001 --device cuda \
  --encoder tree --hidden-dim 256 --embedding-dim 128 \
  --classes-per-batch 16 --samples-per-class 4 \
  --epochs 50 --steps-per-epoch 200
```

Each batch samples P distinct equivalence classes and K distinct incidents in each. Every training class must have at least K incidents; P and K must both be at least 2. An epoch is the configured number of sampled steps, not a guaranteed traversal of every incident. Sampling balances equivalence classes rather than their empirical frequencies.

The optional `--encoder directed` baseline uses separate incoming/outgoing neighbor sums, residual updates, and LayerNorm. Its readout combines node sum, node mean, and log node count. Both encoders preserve node-index permutation invariance. Neither is guaranteed injective or a complete graph-isomorphism test. The default tree encoder's recursive computation reaches every ancestor, but finite hidden states do not guarantee retention of every descendant detail.

The default set model is weighted DeepSets, `normalize(rho(sum(weight * phi(graph))))`. Set order does not matter; weight scale does not matter. Normalized pooling discards absolute total evidence mass and collection count. Finite-vector compression may merge different explanation sets. Use `--set-pool attention` for learned scalar attention combined with the supplied evidence weights. This is attention pooling, **not** a Set Transformer. Both options train the graph encoder and set encoder end to end.

Normalized incident vectors use supervised contrastive learning: for each anchor, average the log probabilities of all other same-class incidents in the batch; the denominator contains every other incident. Training reports mean loss and embedding standard deviation as a basic collapse diagnostic.

Output includes `config.json`, `metrics.jsonl`, `last.pt`, and `best.pt` selected by validation MRR. Checkpoints contain both encoder weights, model configuration, optimizer state, epoch, and metrics. Use a fresh output directory for every run.

## Export and evaluate

```bash
incident-embeddings embed --checkpoint runs/deepsets-001/best.pt \
  --input /data/new_incidents.jsonl --output runs/new_vectors.jsonl --device cuda

incident-embeddings evaluate --checkpoint runs/deepsets-001/best.pt \
  --queries /data/test.jsonl --gallery /data/history.jsonl --device cuda
```

Export writes one `incident_id` and normalized `embedding` per JSONL line. Rank vectors by dot product (equal to cosine similarity here), then fetch incident records and explanation evidence for the RAG prompt. Keep historical vectors and query vectors on the same checkpoint and feature schema; rebuild the index after changing either. Retain original graphs separately for explanation and reranking.

Evaluation uses exact cosine ranking with one query's scores in memory at a time. Matching incident IDs are excluded. It reports Hit@1, Hit@5, and MRR conditional on having at least one relevant gallery incident, plus query coverage. Hit@K means at least one relevant result in the top K; it is not fraction-of-all-positives Recall@K. Queries with no positive in the gallery are excluded from those ranking metrics and exposed through coverage. Ties use the tensor sorting order; do not interpret tie-heavy scores as diagnostic evidence.

Use chronological partitions for historical retrieval, grouping all variants of one incident together. Equivalence classes may overlap across partitions. For unseen-class evaluation, provide a separate historical gallery containing examples of those unseen classes; otherwise class-relevance retrieval is undefined. Freeze a final test set and never use it to select checkpoints. The training command checks IDs and optional leakage groups, but cannot detect semantically duplicated incidents or postmortem feature leakage.

## Synthetic smoke run

```bash
python examples/make_demo.py
incident-embeddings train --train examples/data/train.jsonl \
  --val examples/data/val.jsonl --output runs/smoke --device cpu \
  --classes-per-batch 4 --samples-per-class 3 \
  --encoder tree --hidden-dim 32 --embedding-dim 16 \
  --epochs 2 --steps-per-epoch 5
incident-embeddings evaluate --checkpoint runs/smoke/best.pt \
  --queries examples/data/test.jsonl --gallery examples/data/train.jsonl --device cpu
```

The synthetic classes deliberately have separable features. Success verifies plumbing, not RCA usefulness. Before using production retrieval, compare against mean node-feature pooling, explicit split matching, and symptom-only retrieval on the same partitions. Test graph size, explanation count, missing evidence, and changed component naming as potential shortcuts or shift sources.

## Python access

```python
from incident_embeddings.cli import load_model
from incident_embeddings.data import collate, read_records
import torch

model = load_model("runs/deepsets-001/best.pt", "cuda").eval()
records, _ = read_records("/data/new_incidents.jsonl", model.config.node_dim, labeled=False)
batch = collate(records[:32], "cuda")
with torch.no_grad():
    graph_vectors = model.graph_encoder(batch)
    incident_vectors = model.set_encoder(graph_vectors, batch)
```

Graph vectors are intermediate learned representations, not independently calibrated graph distances. Incident vectors are the metric-learning output.
