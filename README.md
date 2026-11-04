# Hypothesis RAG

### Learning representations of hypothesis spaces for root cause retrieval

**Hypothesis RAG investigates whether sets of causal explanations can provide useful representations for retrieving related historical failures.** Given a failure, an upstream tree-structured causal-model workflow produces alternative explanations. We represent each explanation as a tree, embed the trees with a shared encoder, and aggregate their embeddings into one representation of the entire hypothesis space. Both encoders learn jointly from externally supplied equivalence labels.

The motivating application is root cause analysis for video-streaming failures: when a new failure occurs, retrieve historical failures with related explanation structures and use their supporting evidence in a downstream retrieval-augmented generation (RAG) workflow.

**Current status: research implementation, version 0.2.** Tree encoding, set encoding, supervised contrastive training, exact retrieval evaluation, and vector export are implemented. Synthetic runs exercise the complete learning pipeline. Effectiveness on real incidents, the contribution of each architectural component, and downstream RAG utility remain open experimental questions.

## Research objective

The project asks:

> Can an embedding of a failure's hypothesis space retrieve relevant historical failures, including cases whose observed symptoms differ but whose underlying explanations are related?

A symptom-only representation can overlook alternative explanations. Representing the full hypothesis space retains competing explanations at the input stage and lets the retrieval objective determine which structural and evidential features matter. Whether the final compressed vector preserves enough of that information is a hypothesis to test, not an established result.

The intended system has three parts:

1. **Construct hypothesis spaces upstream.** Generate explanation trees from tree-structured causal models and determine equivalences using the project's split and equivalence definitions.
2. **Learn and retrieve hypothesis-space embeddings.** Train a tree encoder and a set encoder, then compare new spaces with a historical gallery. This is the scope implemented here.
3. **Use retrieved evidence for RCA assistance.** Retrieve the original explanations and incident records, then supply them to a RAG system. Vector-index deployment and answer generation are future integration work.

## Terminology and problem formulation

**A hypothesis space is a set of trees, not one tree.** Each tree represents a different candidate explanation of the same failure. A single explanation may involve one cause or several causes. Forests are also supported when a split leaves multiple disconnected tree components within one explanation.

| Term | Meaning |
| --- | --- |
| Hypothesis | One candidate causal explanation, represented by a tree or forest. |
| `hypothesis_id` | Identifier for an individual hypothesis. It is metadata, not a similarity label or model feature. |
| Hypothesis space | The unordered collection of candidate explanation trees for a failure. |
| Hypothesis embedding | The vector produced by the first encoder for one tree or forest. |
| Hypothesis-space embedding | The vector produced by the set encoder for the entire collection. |
| Equivalence label | An externally supplied class label supervising relationships between whole hypothesis spaces. |

For failure $i$, let

$$
\mathcal H_i = \{T_{i1},\ldots,T_{im_i}\}
$$

be its hypothesis space, where $T_{ik}$ is an explanation tree or forest. The learned representation is

$$
g_{ik}=f_\theta(T_{ik}), \qquad
z_i=S_\psi\bigl(\{(g_{ik},w_{ik})\}_{k=1}^{m_i}\bigr).
$$

Here $f_\theta$ is the tree encoder, $S_\psi$ is the set encoder, and $w_{ik}$ is an optional supplied explanation weight. Retrieval ranks historical spaces by cosine similarity between their normalized vectors $z$.

**Supervision boundary.** The current loss requires equivalence labels for whole hypothesis spaces. Equivalence between individual splits or individual explanation trees does not automatically define equivalence between their collections. That mapping must be supplied upstream. There is no separate tree-level contrastive objective in this version.

## What is implemented

| Component | Current capability | Boundary |
| --- | --- | --- |
| Structured input | Validates node features and directed trees/forests. | Does not generate causal models or compute equivalent splits. |
| Hypothesis encoder | Shared, gated bottom-up tree recurrence. | Custom architecture; no tree attention or explicit edge-feature encoder. |
| Graph readout | Combines root summaries, a node-state mean, and size features. | Finite-vector compression can merge distinct explanations. |
| Hypothesis-space encoder | Weighted DeepSets by default, optional scalar attention pooling. | No Set Transformer or information-gain estimator. |
| Joint learning | Supervised contrastive loss with class-balanced batches and one optimizer. | Requires positive and negative class relationships for complete spaces. |
| Alternative graph encoder | Fixed-depth directed message passing for comparison. | Still uses the tree/forest input contract. |
| Evaluation and export | Exact cosine retrieval, checkpoint selection, JSONL vector export. | No deployed vector database or RAG answer-generation service. |

## Method and design decisions

### 1. Bottom-up tree encoding

For node $v$, let $x_v$ be its input feature vector. The encoder computes

$$
a_v=W_xx_v+b_x, \qquad
c_v=\sum_{u\in\operatorname{Children}(v)}\phi_\theta(h_u), \qquad
h_v=\operatorname{GRUCell}_\theta(a_v,c_v).
$$

Leaves use $c_v=0$. Nodes are processed in reverse topological order, grouped by height across the batch. The child MLP and gated update share parameters across nodes, depths, and hypotheses.

**Why this encoder?** The input structure supports recursive subtree aggregation. A root has a computational dependency on every descendant without a fixed message-passing radius. The learned child transformation followed by summation preserves sensitivity to repeated child messages while remaining invariant to sibling ordering.

The GRU receives the node projection as its input and the child-message sum as its hidden argument. It provides a learned gate for combining those quantities. This is a custom gated tree encoder, not a GAT, TreeLSTM, or causal simulator. Attention does not operate inside the tree encoder.

**Limitations.** Deep recurrences can weaken gradients, high branching factors can change message scales, and finite hidden states can discard information. Sibling order is ignored; meaningful branch roles must be encoded in features. A path from every descendant is not a guarantee of injectivity or lossless structural representation.

### 2. Graph-level readout

Each explanation becomes one vector by pooling its completed node states. For node set $V$ and root set $R$:

$$
r_\Sigma=\sum_{v\in R}h_v,\qquad
r_\mu=\frac{r_\Sigma}{|R|},\qquad
m=\frac{1}{|V|}\sum_{v\in V}h_v,
$$

$$
g=f_{\mathrm{readout}}\left([r_\Sigma,r_\mu,m,\log(1+|V|),\log(1+|R|)]\right).
$$

Brackets denote concatenation. The readout is a two-layer MLP with GELU activation.

**Why these summaries?** Root states carry recursively aggregated subtree information. The all-node mean gives local node states a direct route to the output and a shorter gradient path. Root aggregation supports forests, and counts expose structural size. For a single tree, root sum and root mean are identical. Counts may become shortcuts, so root-only and count-free readouts are planned ablations.

### 3. Weighted DeepSets for the set of trees

The default set encoder computes

$$
\alpha_{ik}=\frac{w_{ik}}{\sum_j w_{ij}},\qquad
u_i=\sum_k\alpha_{ik}\phi_\psi(g_{ik}),\qquad
z_i=\operatorname{normalize}\bigl(\rho_\psi(u_i)\bigr).
$$

Both $\phi_\psi$ and $\rho_\psi$ are MLPs. With equal weights, this is a **mean of transformed tree embeddings**. It is distinct from the unnormalized child-message sum inside each tree.

**Why weighted averaging?** It supports variable numbers of explanations, removes ordering dependence, and keeps aggregate scale from growing directly with the number of hypotheses. We retain it as the current baseline while investigating whether hypothesis-space cardinality carries useful signal or mainly reflects candidate enumeration.

Normalization discards absolute weight mass and explicit hypothesis count. Replicating the entire collection leaves the representation unchanged, while duplicating only one member generally changes it. Deduplicate unintended repeated explanations upstream. Weighted averaging is not claimed to be as expressive as an injective multiset encoding; sum pooling and mean-plus-count remain future comparisons.

With `--set-pool attention`, a learned scalar score modifies the supplied weights:

$$
\alpha_{ik}=\operatorname{softmax}_k\left(s_\psi(\phi_\psi(g_{ik}))+\log w_{ik}\right).
$$

This is attention pooling across hypotheses, with each score computed independently before normalization. It is not pairwise self-attention or a Set Transformer.

### 4. End-to-end equivalence learning

Let $c_i$ be the equivalence class of space $i$ and $P(i)$ the distinct same-class spaces in the batch. The supervised contrastive objective is

$$
\mathcal L=-\frac{1}{B}\sum_{i=1}^{B}\frac{1}{|P(i)|}
\sum_{p\in P(i)}
\log\frac{\exp(z_i^\top z_p/\tau)}{\sum_{a\ne i}\exp(z_i^\top z_a/\tau)}.
$$

Here $B$ is batch size and $\tau>0$ is the temperature. The training loop keeps both encoders in the same autograd graph:

```python
optimizer.zero_grad(set_to_none=True)
hypothesis_vectors = model.graph_encoder(batch)
space_vectors = model.set_encoder(hypothesis_vectors, batch)
loss = supervised_contrastive_loss(space_vectors, labels, temperature)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
optimizer.step()
```

**Why joint training?** The retrieval loss updates both the subtree features and the collection representation. Tree embeddings are recomputed at every step rather than frozen between stages. They receive supervision through the space-level objective, so their standalone distances are not separately calibrated.

Matching labels define positives and different labels define negatives. Missing or uncertain relationships cannot safely be represented by assigning arbitrary different classes. Binary equivalence also does not specify a calibrated similarity scale between nonequivalent spaces. A cosine score is not a probability of a shared root cause.

## Evidence and current limitations

The repository includes automated checks for permutation invariance, direction sensitivity, forest validation, batching, gradient flow through deep trees, parameter updates in both encoders, checkpoint loading, and retrieval behavior. Synthetic runs have exercised training, evaluation, and normalized vector export for both set-pooling options.

These checks support **implementation correctness and trainability**. The synthetic generator deliberately makes classes separable through node features. Successful retrieval on that data does not demonstrate that the encoder uses topology, captures causal equivalence, or outperforms simpler baselines. Joint loss reduction also does not isolate the benefit of either encoder.

There are currently no real-incident benchmark results or downstream RAG results established by this repository. CUDA execution and large-scale throughput still require verification on the target training environment. The model learns supplied equivalences statistically; it does not enforce exact invariance to every equivalence transformation.

## Installation and quick start

Requires **Python 3.10+** and **PyTorch >=2.2,<3**. Install a PyTorch build compatible with the target machine's CUDA environment, then install this package from the repository root:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The implementation uses native PyTorch operations without a PyTorch Geometric dependency. It loads JSONL data into memory and trains on one device in float32. Distributed training, mixed precision, and automatic checkpoint resume are not implemented.

### Run the synthetic example

```bash
python examples/make_demo.py

incident-embeddings train \
  --train examples/data/train.jsonl \
  --val examples/data/val.jsonl \
  --output runs/smoke \
  --device cpu \
  --encoder tree --set-pool deepsets \
  --classes-per-batch 4 --samples-per-class 3 \
  --hidden-dim 32 --embedding-dim 16 \
  --epochs 2 --steps-per-epoch 5

incident-embeddings evaluate \
  --checkpoint runs/smoke/best.pt \
  --queries examples/data/test.jsonl \
  --gallery examples/data/train.jsonl \
  --device cpu
```

Use a fresh output directory for each run. This example verifies the workflow; it is not an RCA benchmark.

## Dataset format

Use **JSONL with one hypothesis space per line**. The current parser retains the field names `incident_id` and `graphs` for compatibility. The following formatted object must occupy one line in an actual JSONL file:

```json
{
  "incident_id": "failure-017",
  "equivalence_id": "space-equivalence-42",
  "leakage_group": "source-failure-017",
  "graphs": [
    {
      "hypothesis_id": "failure-017-h1",
      "node_features": [[1.0, 0.0, 0.3], [0.0, 1.0, 0.8]],
      "edges": [[0, 1]],
      "weight": 0.8
    },
    {
      "hypothesis_id": "failure-017-h2",
      "node_features": [[1.0, 0.0, 0.6]],
      "edges": [],
      "weight": 0.2
    }
  ]
}
```

| Field | Requirements and interpretation |
| --- | --- |
| `incident_id` | Unique nonempty string identifying the record for one failure's hypothesis space. |
| `equivalence_id` | Nonempty string labeling the entire space. Required for training and evaluation, optional for embedding export. |
| `leakage_group` | Optional string grouping windows, variants, or augmentations from the same source failure. The training command rejects overlap across training and validation when supplied. |
| `graphs` | Nonempty list of hypothesis trees or forests. |
| `hypothesis_id` | Optional member metadata in the current parser. Preserved in loaded dictionaries but not validated, embedded, or exported by the existing CLI. It does not replace `equivalence_id`. |
| `node_features` | Nonempty matrix of finite numeric values with the same feature width and semantics across all trees. |
| `edges` | Directed `[parent, child]` integer pairs using local zero-based node indices. Empty lists and isolated nodes are valid. |
| `weight` | Optional finite nonnegative explanation weight, default 1. At least one weight per hypothesis space must be positive. Weights need not be probabilities. |

Every node may have at most one parent. Cycles, self-loops, duplicate edges, and multiple-parent nodes are rejected. Forests are supported, but general DAGs require a different contract. Supply parent-to-child orientation only when it correctly represents the intended rooted structure; changing orientation changes the representation.

Feature engineering and normalization remain upstream responsibilities. Fit any feature normalization on training data and reuse it for validation and inference. Node roles, anomaly evidence, and branch attributes are possible features, not fields the package constructs automatically. Edge attributes and split histories are not consumed. Restrict query features to information available at retrieval time; do not leak postmortem causes, equivalence labels, or successful remediation outcomes into those features.

The batch builder preserves the mappings **node to hypothesis** and **hypothesis to hypothesis space**. It combines trees as disjoint components and creates a bottom-up execution schedule. It does not add edges between alternative explanations.

## Training and configuration

```bash
incident-embeddings train \
  --train /data/train.jsonl --val /data/val.jsonl \
  --output runs/tree-deepsets-001 --device cuda \
  --encoder tree --set-pool deepsets \
  --hidden-dim 256 --embedding-dim 128 \
  --classes-per-batch 16 --samples-per-class 4 \
  --epochs 50 --steps-per-epoch 200 --seed 42
```

These are example settings, not tuned recommendations.

| Option | Default | Meaning |
| --- | --- | --- |
| `--encoder` | `tree` | Bottom-up recurrence. `directed` selects the fixed-depth message-passing baseline. |
| `--hidden-dim` | 128 | Internal node-state and MLP hidden width. |
| `--embedding-dim` | 128 | Output width for both hypotheses and hypothesis spaces. |
| `--graph-layers` | 3 | Number of layers for `directed`; unused by the tree recurrence. |
| `--set-pool` | `deepsets` | Normalized supplied weights; `attention` additionally learns scalar scores. |
| `--classes-per-batch` | 8 | Number of distinct equivalence classes sampled per step. |
| `--samples-per-class` | 4 | Distinct hypothesis spaces sampled from each selected class. |
| `--temperature` | 0.1 | Contrastive softmax temperature. |
| `--lr` / `--weight-decay` | 0.0003 / 0.0001 | AdamW settings. |
| `--epochs` / `--steps-per-epoch` | 50 / 100 | Number of epochs and sampled optimization steps per epoch. |
| `--seed` | 42 | Python and PyTorch random seed; not a guarantee of cross-device determinism. |

A batch samples $P$ classes and $K$ distinct spaces per class. Both must be at least two, every training class must contain at least $K$ records, and there must be at least $P$ classes. Uniform class sampling balances classes rather than matching their production frequencies. An epoch is a fixed number of sampled steps, not a full traversal of every record.

Training writes `config.json`, `metrics.jsonl`, `last.pt`, and `best.pt`. The best checkpoint maximizes validation MRR. Checkpoints contain both encoders, model configuration, optimizer state, epoch, and metrics. Although optimizer state is saved, the CLI does not implement training resume.

The alternative directed encoder uses separate incoming/outgoing sums, residual updates, and LayerNorm. Version 0.1 checkpoints load as directed models. Switching architecture requires a new checkpoint and regenerated historical embeddings.

## Evaluation and embedding export

```bash
incident-embeddings evaluate \
  --checkpoint runs/tree-deepsets-001/best.pt \
  --queries /data/test.jsonl --gallery /data/history.jsonl \
  --device cuda

incident-embeddings embed \
  --checkpoint runs/tree-deepsets-001/best.pt \
  --input /data/new_spaces.jsonl \
  --output runs/new_space_vectors.jsonl --device cuda
```

Export writes `incident_id` and `embedding` for each hypothesis space. It does not export member hypotheses or their identifiers. Retain those original records separately. The output file must not already exist.

During training, validation spaces query a gallery of training spaces. The standalone evaluation command accepts a separate gallery. Both use exact cosine ranking and exclude matching record IDs.

| Metric | Definition |
| --- | --- |
| Hit@1 / Hit@5 | Fraction of eligible queries with at least one same-equivalence result in the first 1 or 5 positions. |
| MRR | Mean reciprocal rank of the first same-equivalence result. |
| Query coverage | Fraction of queries with at least one relevant, non-self gallery member. |
| Embedding standard deviation | Training log diagnostic for representation variation; not a complete collapse test. |

Ranking metrics exclude queries without a relevant gallery member, and evaluation fails if no queries are eligible. Always report coverage alongside ranking results. Hit@K is not fraction-of-all-positives Recall@K, and these metrics do not measure correct rejection of novel failures. Tie-heavy rankings should not be interpreted as diagnostic evidence.

Use chronological partitions when evaluating historical retrieval. Keep all variants of a source failure within one partition, while allowing equivalent independent failures across partitions. For unseen-class evaluation, supply historical examples from those classes in a separate held-out gallery. Reserve a final test set for reporting, not checkpoint selection. ID checks cannot detect semantic duplicates or feature leakage.

For downstream RAG, retrieve by vector similarity, fetch the corresponding explanation trees and records, and ground the response in that evidence. Query and gallery vectors must use the same checkpoint and feature schema. A close embedding alone is not evidence of an identical root cause.

## Future objectives and evaluation plan

| Objective | Planned work | Evidence needed |
| --- | --- | --- |
| Integrate the causal-model outputs | Map real explanation trees and `hypothesis_id` metadata into the data contract; confirm whole-space equivalence supervision. | Reviewed end-to-end examples with consistent node semantics and labels. |
| Establish retrieval usefulness | Train on domain data and evaluate on chronological held-out failures. | Hit@1, Hit@5, MRR, coverage, uncertainty across seeds, and inspected retrieval errors. |
| Measure the value of structure | Compare against symptom-only retrieval, mean node-feature pooling, and direct equivalence/split matching where available. | Gains that persist after controlling for size and feature shortcuts. |
| Isolate encoder contributions | Compare directed versus recursive tree encoders, root-only versus combined readout, and frozen-encoder variants. | Ablations showing which components contribute to held-out performance. |
| Test set aggregation | Compare weighted mean, sum, mean-plus-count, and scalar attention; consider Set Transformer if interactions matter. | Performance across hypothesis counts, redundancy, and missing explanations. |
| Study informative weighting | Compare uniform, evidence-based, and expected-information-gain weights. | A defined diagnostic query model and improvement over uniform weighting. |
| Evaluate robustness | Test deeper trees, changed branching factors, missing evidence, component renaming, and unseen families. | Retrieval and rejection behavior under controlled distribution shifts. |
| Integrate RAG | Connect vector retrieval to source records and evidence-grounded generation. | Relevant supporting evidence and useful RCA responses beyond retrieval metrics alone. |

### Expected information gain as a possible weight

Entropy-reduction weighting is a research extension, not an implemented estimator. Let $C$ denote the uncertain causal explanation, $D$ the available evidence, $q$ a diagnostic query or test, and $Y$ its outcome. Expected information gain is

$$
\operatorname{IG}(q\mid D)=H(C\mid D)-\mathbb E_{Y\mid D,q}[H(C\mid D,Y,q)].
$$

Assigning this utility to hypothesis $k$ requires an associated query $q_k$ or another explicitly defined evidence-acquisition operation. One candidate weighting is $\alpha_k\propto\exp(\lambda\operatorname{IG}(q_k\mid D))$, with $\lambda\ge0$. Greater expected entropy reduction would increase pooling weight.

This weights diagnostic utility, which differs from the probability that a hypothesis is correct. It needs a probabilistic outcome model, must use only evidence available at retrieval time, and may change the meaning of similarity. Weighted DeepSets can consume supplied weights but does not learn or guarantee entropy reduction by itself.

**Immediate next milestone:** validate one complete hypothesis-space example and its equivalence label with the upstream pipeline, then run a controlled real-data comparison of the tree-plus-DeepSets model against the directed baseline and a feature-only baseline.

## Repository structure

```text
src/incident_embeddings/
  model.py       Tree encoder, graph readout, set encoder, directed baseline
  data.py        JSONL validation, forest scheduling, batching, class sampler
  training.py    Contrastive loss, embedding inference, retrieval metrics
  cli.py         Training loop, checkpoint loading, evaluate/embed commands
  __init__.py    Public model imports
examples/
  make_demo.py   Synthetic tree datasets for workflow checks
tests/
  test_pipeline.py
pyproject.toml   Package metadata and command-line entry point
```

## Related work

These works motivate the architecture and evaluation questions. This repository does not claim to reproduce their full methods or establish a new state-of-the-art result.

- Zaheer et al. **Deep Sets.** NeurIPS 2017. [Paper](https://arxiv.org/abs/1703.06114). Permutation-invariant transform–aggregate–transform learning.
- Xu et al. **How Powerful Are Graph Neural Networks?** ICLR 2019. [Paper](https://arxiv.org/abs/1810.00826). Multiset aggregation and message-passing expressivity; its injectivity results do not automatically apply to this custom encoder.
- Wagstaff et al. **On the Limitations of Representing Functions on Sets.** ICML 2019. [Paper](https://proceedings.mlr.press/v97/wagstaff19a.html). Limits of continuous finite-dimensional set representations.
- Lee et al. **Set Transformer.** ICML 2019. [Paper](https://proceedings.mlr.press/v97/lee19d.html). A potential alternative for interactions among hypotheses; not implemented here.
- Khosla et al. **Supervised Contrastive Learning.** NeurIPS 2020. [Paper](https://arxiv.org/abs/2004.11362). Label-supervised representation learning objective.
- Foster et al. **A Unified Stochastic Gradient Approach to Designing Bayesian-Optimal Experiments.** AISTATS 2020. [Paper](https://proceedings.mlr.press/v108/foster20a.html). Expected information gain as an experimental-design objective.
