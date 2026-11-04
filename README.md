# Hypothesis RAG

**Hierarchical representation learning over sets of attributed causal explanation trees**

## Abstract

This project studies metric learning for retrieval of historical failures from structured hypothesis spaces. Each failure is represented by a finite collection of candidate causal explanations, each encoded as an attributed directed tree or forest derived from an upstream tree-structured causal model. A shared recursive graph encoder maps each explanation to a vector; a permutation-invariant set encoder maps the collection to a hypothesis-space representation. Both encoders are optimized jointly using supervised contrastive learning with externally supplied hypothesis-space equivalence labels. The intended application is retrieval of structurally and semantically related failures for evidence-grounded root cause analysis (RCA) in video-streaming systems.

**Implementation status — version 0.2.0:** input validation, recursive tree encoding, weighted set encoding, joint optimization, exact vector retrieval, and embedding export are implemented. Causal-model construction, equivalence computation, information-gain estimation, and retrieval-augmented generation (RAG) are outside the current implementation. Synthetic execution checks establish pipeline functionality; domain-level retrieval performance and causal validity remain unestablished.

## 1. Problem formulation

### 1.1 Attributed explanation trees

Represent candidate explanation $k$ for failure $i$ by

$$
T_{ik}=(V_{ik},E_{ik},X_{ik}),
\qquad X_{ik}\in\mathbb R^{|V_{ik}|\times F},
$$

where $V_{ik}$ is a nonempty node set, $E_{ik}$ contains directed parent-to-child edges, and $F$ is the node-feature dimension. The graph is acyclic and each node has in-degree at most one. A connected graph satisfying these constraints is a rooted tree; disconnected components form a rooted forest. Each explanation can encode multiple causes. Neither a node nor a complete tree is assumed to correspond to exactly one cause.

The architecture consumes the supplied graph structure and features. It does not estimate structural equations, perform interventions, or establish causal identifiability. The interpretation of edges and node attributes is part of the upstream data specification.

### 1.2 Hypothesis spaces and identifiers

The hypothesis space for failure $i$ is

$$
\mathcal H_i=\{T_{i1},\ldots,T_{im_i}\},\qquad m_i\geq 1.
$$

**A hypothesis space is a set of alternative explanation trees, not an individual tree.** Optional member weights satisfy $w_{ik}\geq0$ and $\sum_k w_{ik}>0$.

| Concept | Mathematical object | Current schema |
| --- | --- | --- |
| Candidate hypothesis | Attributed tree or forest $T_{ik}$ | One entry in `graphs` |
| Hypothesis identifier | Metadata identifying $T_{ik}$ | `hypothesis_id` |
| Hypothesis space | Collection $\mathcal H_i$ | One JSONL record |
| Record identifier | Metadata identifying the space record | `incident_id` |
| Equivalence class | Label $y_i$ for the complete space | `equivalence_id` |

The input is serialized as a list and is not deduplicated. Consequently, the implemented aggregation operates on a **weighted multiset**, or equivalently a normalized empirical measure, rather than enforcing mathematical set uniqueness:

$$
\mu_i=\sum_{k=1}^{m_i}\alpha_{ik}\delta_{T_{ik}},
\qquad \alpha_{ik}=\frac{w_{ik}}{\sum_j w_{ij}}.
$$

Here $\delta_T$ denotes a point mass at explanation $T$. Repeated members contribute repeated mass. The upstream pipeline must specify whether repeated explanations are distinct candidates or duplicates to remove.

### 1.3 Equivalence supervision and retrieval

Let $\sim$ denote the externally defined equivalence relation on complete hypothesis spaces, with labels satisfying

$$
y_i=y_j\iff\mathcal H_i\sim\mathcal H_j
$$

within the labeled dataset. Equivalence between splits or individual trees does not by itself specify equivalence between their collections; the space-level labeling rule must be defined upstream.

We learn

$$
g_{ik}=f_\theta(T_{ik})\in\mathbb R^p,\qquad
z_i=S_\psi\bigl(\{(g_{ik},w_{ik})\}_{k=1}^{m_i}\bigr)\in\mathbb R^p.
$$

For query space $\mathcal H_q$ and historical gallery $\mathcal G$, retrieval orders records by

$$
s(q,j)=z_q^\top z_j,\qquad j\in\mathcal G.
$$

The implementation normalizes outputs using $N_\varepsilon(v)=v/\max(\lVert v\rVert_2,\varepsilon)$. For nondegenerate vectors with norm at least $\varepsilon$, this score is cosine similarity. A zero pre-normalization vector remains zero; normalization alone does not prevent representation collapse.

The objective encourages equivalent spaces to have similar representations. It does **not** impose the exact quotient-space condition $z(\mathcal H)=z(\mathcal H')$ whenever $\mathcal H\sim\mathcal H'$, and it does not define a calibrated distance between distinct equivalence classes. Similarity scores are neither causal probabilities nor evidence of identical root causes.

**Research question.** Does hierarchical encoding of $(E,X)$ and the collection of candidate explanations improve held-out equivalence retrieval over representations that omit adjacency, omit alternative hypotheses, or use fixed structural descriptors?

## 2. Implemented architecture

The default computation is

$$
\{(V_{ik},E_{ik},X_{ik},w_{ik})\}_{k=1}^{m_i}
\xrightarrow{\ f_\theta\ }
\{(g_{ik},w_{ik})\}_{k=1}^{m_i}
\xrightarrow{\ S_\psi\ }z_i.
$$

Let $d$ denote `hidden_dim`, $p$ denote `embedding_dim`, and $F$ denote `node_dim`. Node states have dimension $d$; both explanation embeddings and hypothesis-space embeddings have dimension $p$. Unless specified otherwise, each MLP is $\mathrm{Linear}\rightarrow\mathrm{GELU}\rightarrow\mathrm{Linear}$ with hidden width $d$.

### 2.1 Recursive bottom-up graph encoder

For node $v$, let $\mathrm{Ch}(v)=\{u:(v,u)\in E\}$. The encoder computes

$$
a_v=W_xx_v+b_x\in\mathbb R^d,
$$

$$
c_v=\sum_{u\in\mathrm{Ch}(v)}\phi_\theta(h_u),\qquad
h_v=\mathrm{GRUCell}_\theta(a_v,c_v)\in\mathbb R^d,
$$

where $\phi_\theta:\mathbb R^d\rightarrow\mathbb R^d$ is the child-message MLP and $c_v=0$ at leaves. In the PyTorch call, $a_v$ is the GRU input and $c_v$ is its hidden-state argument. Parameters are shared across nodes, heights, and explanations.

The batch scheduler processes nodes by increasing subtree height. Therefore every descendant has a computational path to its ancestor's representation, without choosing a fixed number of message-passing layers. This is a custom gated recursive graph encoder, not GAT or TreeLSTM. **The tree encoder does not use attention.**

**Design rationale.** The recurrence matches the rooted input domain. Summation gives a permutation-invariant aggregation of transformed child states and permits sensitivity to child multiplicity. The GRU supplies a learnable nonlinear update combining local attributes and descendant information. Unlike a fixed-depth local GNN, the recurrence covers complete subtrees irrespective of depth.

**Expressivity and stability.** Summation does not make this finite-width architecture injective. Different child multisets can produce the same sum, and the gated update can introduce further collisions. Deep computational paths can attenuate gradients; large branching factors can increase message magnitude. Full descendant dependence is not a guarantee against over-squashing. Claims of GIN-level or Weisfeiler–Leman expressivity would require additional assumptions and analysis.

### 2.2 Explanation-level graph readout

For a tree or forest with root set $R$, define $n=|V|$, $r=|R|$, and

$$
b_\Sigma=\sum_{v\in R}h_v,\qquad
b_\mu=\frac{1}{r}\sum_{v\in R}h_v,\qquad
b_V=\frac{1}{n}\sum_{v\in V}h_v.
$$

The graph embedding is

$$
g=\rho_\theta^{\mathrm{graph}}
\left(b_\Sigma\mathbin\Vert b_\mu\mathbin\Vert b_V
\mathbin\Vert\log(1+n)\mathbin\Vert\log(1+r)\right),
$$

where $\Vert$ denotes concatenation and $\rho_\theta^{\mathrm{graph}}:\mathbb R^{3d+2}\rightarrow\mathbb R^p$ is an MLP.

| Readout term | Architectural purpose | Limitation |
| --- | --- | --- |
| Root sum $b_\Sigma$ | Aggregates recursive component representations across a forest. | Magnitude can depend on component count. |
| Root mean $b_\mu$ | Provides a component-count-normalized root summary. | Equals the root sum for a single tree. |
| Node-state mean $b_V$ | Provides direct output dependence on every completed node state. | Does not identify the multiset of node states uniquely. |
| Log counts | Explicitly expose node and component counts. | May support size-based shortcuts under biased sampling. |

This readout combines recursive component summaries and a global node-state statistic. It is an implemented design choice whose benefit requires comparison with root-only and count-free readouts. It is not a lossless encoding of the input forest.

### 2.3 Hypothesis-space encoder: weighted DeepSets

The default `deepsets` option computes

$$
v_{ik}=\phi_\psi(g_{ik})\in\mathbb R^d,\qquad
u_i=\sum_{k=1}^{m_i}\alpha_{ik}v_{ik},\qquad
z_i=N_\varepsilon\bigl(\rho_\psi(u_i)\bigr),
$$

with MLPs $\phi_\psi:\mathbb R^p\rightarrow\mathbb R^d$ and $\rho_\psi:\mathbb R^d\rightarrow\mathbb R^p$. Thus $u_i$ is a learned finite-dimensional expectation under $\mu_i$. Equal supplied weights yield mean pooling of transformed explanation embeddings.

**Design rationale.** The transform–aggregate–transform structure supports variable collection cardinality and permutation invariance. Normalized weighting separates aggregate scale from total supplied weight and permits upstream confidence or utility weights. It provides a baseline for learning over collections before introducing pairwise interactions between hypotheses.

**Information discarded.** For any $a>0$, replacing all weights by $aw_{ik}$ leaves the result unchanged. Replicating the complete weighted collection also leaves it unchanged. Absolute weight mass and explicit collection cardinality are therefore absent. Duplicating only one member generally changes the empirical measure and the embedding. Different measures can have identical finite-dimensional expectations.

Unnormalized sum pooling can distinguish some multisets that mean pooling cannot: $\{v\}$ and $\{v,v\}$ have equal means but different sums when $v\ne0$. This does not establish universal superiority of sum pooling or injectivity of arbitrary learned sums. Whether multiplicity is relevant depends on the semantics of candidate enumeration. Sum pooling and mean-plus-count are planned ablations, not current CLI options.

The optional `attention` encoder uses

$$
\widetilde\alpha_{ik}
=\frac{w_{ik}\exp(s_\psi(v_{ik}))}
{\sum_j w_{ij}\exp(s_\psi(v_{ij}))},
\qquad u_i=\sum_k\widetilde\alpha_{ik}v_{ik},
$$

where $s_\psi$ is a learned affine scalar score. This is equivalent to the implemented softmax of $s_\psi(v_{ik})+\log w_{ik}$; zero-weight members receive zero mass. Scores are computed independently per hypothesis before normalization. This is scalar attention pooling, not pairwise self-attention or Set Transformer.

### 2.4 Invariances and their scope

For a node permutation $\pi$, apply the permutation consistently to features and edge indices. The shared recurrence is node-permutation equivariant, and its readout satisfies

$$
f_\theta(\pi\cdot T)=f_\theta(T).
$$

The argument follows by induction over subtree height: corresponding leaves receive equal inputs; commutative child summation preserves corresponding parent states; symmetric root and node pooling removes node-index dependence.

For any permutation $\sigma$ of hypothesis entries,

$$
S_\psi\bigl(\{(g_{i\sigma(k)},w_{i\sigma(k)})\}_k\bigr)
=S_\psi\bigl(\{(g_{ik},w_{ik})\}_k\bigr).
$$

These identities hold in exact arithmetic; floating-point reductions can introduce numerical differences. They establish invariance to indexing and ordering, not invariance to edge reversal, feature changes, arbitrary candidate duplication, or upstream causal equivalence transformations. Structural roles that must be distinguished require corresponding attributes or an extended edge representation.

### 2.5 Fixed-depth directed GNN baseline

The implemented `directed` option applies $L$ layers with separate incoming and outgoing transformations:

$$
m_v^{\mathrm{in},\ell}=\sum_{u:(u,v)\in E}W_{\mathrm{in}}^\ell h_u^\ell,
\qquad
m_v^{\mathrm{out},\ell}=\sum_{u:(v,u)\in E}W_{\mathrm{out}}^\ell h_u^\ell,
$$

$$
h_v^{\ell+1}=\mathrm{LayerNorm}\left(
 h_v^\ell+U_\ell(h_v^\ell\Vert m_v^{\mathrm{in},\ell}\Vert m_v^{\mathrm{out},\ell})
\right).
$$

Its readout concatenates the sum and mean of final node states with $\log(1+n)$, followed by an MLP. Here $L$ is `graph_layers`; this option does not control recursive tree depth. Each final node state has a bounded $L$-hop receptive field, whereas the recursive encoder aggregates complete descendant subtrees. The existing data validator restricts both options to trees and forests.

## 3. Joint optimization

For a batch of $B$ hypothesis spaces, define the positive index set

$$
P(i)=\{j\ne i:y_j=y_i\}.
$$

The supervised contrastive objective is

$$
\mathcal L=-\frac{1}{B}\sum_{i=1}^B\frac{1}{|P(i)|}
\sum_{p\in P(i)}
\log\frac{\exp(z_i^\top z_p/\tau)}
{\sum_{a\ne i}\exp(z_i^\top z_a/\tau)},\qquad\tau>0.
$$

The implementation normalizes embeddings again within the loss. Class-balanced batches contain at least two classes and at least two distinct spaces per selected class, ensuring positive and negative examples for every anchor.

### Gradient propagation through both encoders

Writing Jacobians as $J$, the chain rule gives

$$
\nabla_\theta\mathcal L
=\sum_i\sum_k (J_\theta g_{ik})^{\top}
(J_{g_{ik}}z_i)^{\top}\nabla_{z_i}\mathcal L,
\qquad
\nabla_\psi\mathcal L
=\sum_i (J_\psi z_i)^{\top}\nabla_{z_i}\mathcal L.
$$

Here $J_\theta g_{ik}$ denotes the Jacobian of $g_{ik}$ with respect to $\theta$, and similarly for the other terms. The graph encoder, graph readout, and set encoder remain in one autograd computation graph. One AdamW optimizer updates all trainable parameters after gradient-norm clipping.

```python
optimizer.zero_grad(set_to_none=True)
hypothesis_vectors = model.graph_encoder(batch)
space_vectors = model.set_encoder(hypothesis_vectors, batch)
loss = supervised_contrastive_loss(space_vectors, labels, temperature)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
optimizer.step()
```

Joint optimization allows the space-level objective to adapt both subtree representations and set aggregation. It establishes a training mechanism, not an empirical guarantee that either encoder improves retrieval. There is no independent tree-level loss; distances between individual hypothesis embeddings are not directly supervised.

Different labels are treated as negatives. Incomplete equivalence labeling can therefore introduce false negatives. The objective does not encode graded similarity between nonequivalent classes, asymmetric transfer utility, or uncertainty in equivalence assignments.

## 4. Implementation scope and evidence

| Component | Implemented | Not established or not implemented |
| --- | --- | --- |
| Input processing | Feature validation, forest validation, disjoint batching, height scheduling | Causal-model estimation and equivalence computation |
| Graph representation | Recursive gated encoder, combined graph readout, directed GNN baseline | Injectivity, causal-equivalence invariance, general DAG support |
| Set representation | Weighted DeepSets and scalar attention pooling | Pairwise attention, automatic deduplication, information-gain estimation |
| Optimization | Supervised contrastive loss, balanced sampling, joint AdamW updates | Separate tree supervision, graded relevance objectives |
| Retrieval | Exact dot-product ranking of normalized outputs, evaluation, vector export | Approximate index deployment, novelty rejection, RAG generation |

Automated checks cover permutation behavior, edge-direction sensitivity, forest validation, batching, gradient flow, updates to both encoders, checkpoint loading, and retrieval behavior. Synthetic runs exercise training, evaluation, and export. The synthetic generator includes class-dependent node features, so success on these data does not isolate structural reasoning or causal equivalence learning.

No real-incident benchmark, architectural advantage, or downstream RCA benefit is established. Target-device CUDA behavior and large-scale throughput require measurement. Empirical evaluation must distinguish software functionality from representation quality and downstream diagnostic utility.

## 5. Installation and quick start

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

Use a fresh output directory for each run. This synthetic dataset evaluates pipeline execution, not domain-level RCA performance.

## 6. Dataset format

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

## 7. Training and configuration

```bash
incident-embeddings train \
  --train /data/train.jsonl --val /data/val.jsonl \
  --output runs/tree-deepsets-001 --device cuda \
  --encoder tree --set-pool deepsets \
  --hidden-dim 256 --embedding-dim 128 \
  --classes-per-batch 16 --samples-per-class 4 \
  --epochs 50 --steps-per-epoch 200 --seed 42
```

The configuration illustrates CLI usage; no hyperparameter search establishes these settings as optimal.

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

## 8. Evaluation and embedding export

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

For query $q$, let $\mathcal G_q$ exclude gallery records with the same `incident_id`, and define

$$
\mathcal R_q=\{j\in\mathcal G_q:y_j=y_q\},\qquad
\mathcal Q_+=\{q:\mathcal R_q\ne\varnothing\}.
$$

If $r_q$ is the rank of the first relevant gallery record, the reported ranking metrics are

$$
\mathrm{Hit@K}=\frac{1}{|\mathcal Q_+|}\sum_{q\in\mathcal Q_+}\mathbf 1[r_q\le K],
\qquad
\mathrm{MRR}=\frac{1}{|\mathcal Q_+|}\sum_{q\in\mathcal Q_+}\frac{1}{r_q},
$$

$$
\mathrm{Coverage}=\frac{|\mathcal Q_+|}{|\mathcal Q|}.
$$

The evaluator reports Hit@1 and Hit@5. Training also logs the mean coordinate-wise standard deviation of gallery embeddings as a representation-variation diagnostic; this statistic alone cannot establish absence of collapse.

Ranking metrics exclude queries without a relevant gallery member, and evaluation fails if no queries are eligible. Always report coverage alongside ranking results. Hit@K is not fraction-of-all-positives Recall@K, and these metrics do not measure correct rejection of novel failures. Tie-heavy rankings should not be interpreted as diagnostic evidence.

Use chronological partitions when evaluating historical retrieval. Keep all variants of a source failure within one partition, while allowing equivalent independent failures across partitions. For unseen-class evaluation, supply historical examples from those classes in a separate held-out gallery. Reserve a final test set for reporting, not checkpoint selection. ID checks cannot detect semantic duplicates or feature leakage.

For downstream RAG, retrieve by vector similarity, fetch the corresponding explanation trees and records, and ground the response in that evidence. Query and gallery vectors must use the same checkpoint and feature schema. A close embedding alone is not evidence of an identical root cause.

## 9. Research objectives and evaluation plan

| Objective | Planned work | Evidence needed |
| --- | --- | --- |
| Integrate the causal-model outputs | Map real explanation trees and `hypothesis_id` metadata into the data contract; confirm whole-space equivalence supervision. | Reviewed end-to-end examples with consistent node semantics and labels. |
| Establish retrieval usefulness | Train on domain data and evaluate on chronological held-out failures. | Hit@1, Hit@5, MRR, coverage, uncertainty across seeds, and inspected retrieval errors. |
| Measure the value of structure | Compare against attribute-only encoders that omit $E$, pooled node-feature statistics, and structural or equivalence descriptors defined upstream. | Held-out retrieval gains after controlling for graph size, class-correlated attributes, and hypothesis count. |
| Isolate encoder contributions | Compare directed versus recursive tree encoders, root-only versus combined readout, and frozen-encoder variants. | Ablations showing which components contribute to held-out performance. |
| Test set aggregation | Compare weighted mean, sum, mean-plus-count, and scalar attention; consider Set Transformer if interactions matter. | Performance across hypothesis counts, redundancy, and missing explanations. |
| Study informative weighting | Compare uniform, evidence-based, and expected-information-gain weights. | A defined diagnostic query model and improvement over uniform weighting. |
| Evaluate robustness | Test deeper trees, changed branching factors, missing evidence, component renaming, and unseen families. | Retrieval and rejection behavior under controlled distribution shifts. |
| Integrate RAG | Connect vector retrieval to source records and evidence-grounded generation. | Relevant supporting evidence and useful RCA responses beyond retrieval metrics alone. |

### Expected information gain as a possible weight

Entropy-reduction weighting is a research extension, not an implemented estimator. Let $C$ denote the uncertain causal explanation, $D$ the available evidence, $q$ a diagnostic query or test, and $Y$ its outcome. Expected information gain is

$$
\mathrm{IG}(q\mid D)=H(C\mid D)-\mathbb E_{Y\mid D,q}[H(C\mid D,Y,q)].
$$

Assigning this utility to hypothesis $k$ requires an associated query $q_k$ or another explicitly defined evidence-acquisition operation. One candidate weighting is $\alpha_k\propto\exp(\lambda\mathrm{IG}(q_k\mid D))$, with $\lambda\ge0$. Greater expected entropy reduction would increase pooling weight.

This weights diagnostic utility, which differs from the probability that a hypothesis is correct. It needs a probabilistic outcome model, must use only evidence available at retrieval time, and may change the meaning of similarity. Weighted DeepSets can consume supplied weights but does not learn or guarantee entropy reduction by itself.

### Falsifiable hypotheses and next experiment

1. **Structural contribution:** the recursive encoder improves held-out MRR relative to an attribute-only baseline using the same features, labels, partitions, and set aggregation. Failure to improve provides no evidence that the tree structure benefits this task.
2. **Collection contribution:** encoding the complete hypothesis space improves retrieval relative to a single-hypothesis representation selected by a fixed, label-independent rule.
3. **Aggregation sensitivity:** differences between normalized mean, sum, and mean-plus-count pooling persist when evaluation controls for hypothesis count and duplicate enumeration.

The principal unresolved questions are the semantics of whole-space equivalence, the information available at retrieval time, and whether hypothesis multiplicity represents evidence or enumeration artifacts.

**Next experiment:** after validating the upstream schema and labels, compare the recursive encoder with an attribute-only baseline under identical weighted set aggregation and chronological, source-group-disjoint partitions. Report MRR, Hit@1, Hit@5, coverage, variation across seeds, and results stratified by tree size and hypothesis count. This directly tests whether the graph structure contributes retrieval information beyond the supplied attributes.

## 10. Repository structure

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

## 11. Related work

These works motivate the architecture and evaluation questions. This repository does not claim to reproduce their full methods or establish a new state-of-the-art result.

- Zaheer et al. **Deep Sets.** NeurIPS 2017. [Paper](https://arxiv.org/abs/1703.06114). Permutation-invariant transform–aggregate–transform learning.
- Xu et al. **How Powerful Are Graph Neural Networks?** ICLR 2019. [Paper](https://arxiv.org/abs/1810.00826). Multiset aggregation and message-passing expressivity; its injectivity results do not automatically apply to this custom encoder.
- Wagstaff et al. **On the Limitations of Representing Functions on Sets.** ICML 2019. [Paper](https://proceedings.mlr.press/v97/wagstaff19a.html). Limits of continuous finite-dimensional set representations.
- Lee et al. **Set Transformer.** ICML 2019. [Paper](https://proceedings.mlr.press/v97/lee19d.html). A potential alternative for interactions among hypotheses; not implemented here.
- Khosla et al. **Supervised Contrastive Learning.** NeurIPS 2020. [Paper](https://arxiv.org/abs/2004.11362). Label-supervised representation learning objective.
- Foster et al. **A Unified Stochastic Gradient Approach to Designing Bayesian-Optimal Experiments.** AISTATS 2020. [Paper](https://proceedings.mlr.press/v108/foster20a.html). Expected information gain as an experimental-design objective.
