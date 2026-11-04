import torch
from torch.nn import functional as F

from .data import collate


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = F.normalize(embeddings.float(), dim=-1)
    logits = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    if not positives.any(1).all() or not ((labels[:, None] != labels[None, :]).any(1)).all():
        raise ValueError("Every anchor requires a distinct positive and a negative")
    denominator = logits.masked_fill(eye, -torch.inf).logsumexp(1, keepdim=True)
    log_probability = logits - denominator
    return -(log_probability.masked_fill(~positives, 0).sum(1) / positives.sum(1)).mean()


@torch.no_grad()
def embed(model, records, device, batch_size=32):
    model.eval()
    return torch.cat([model(collate(records[i:i + batch_size], device)).cpu()
                      for i in range(0, len(records), batch_size)])


def retrieval_metrics(query_vectors, query_records, gallery_vectors, gallery_records, k=5):
    """Exact cosine retrieval; exclude matching incident IDs, report eligible coverage."""
    hits1, hitsk, reciprocal, eligible = 0, 0, 0.0, 0
    for vector, query in zip(query_vectors, query_records):
        candidates = [j for j, r in enumerate(gallery_records) if r["incident_id"] != query["incident_id"]]
        relevant = {j for j in candidates if gallery_records[j]["equivalence_id"] == query["equivalence_id"]}
        if not relevant:
            continue
        eligible += 1
        scores = gallery_vectors[candidates] @ vector
        ranking = [candidates[j] for j in scores.argsort(descending=True).tolist()]
        rank = next(i + 1 for i, j in enumerate(ranking) if j in relevant)
        hits1 += rank == 1
        hitsk += rank <= k
        reciprocal += 1 / rank
    if not eligible:
        raise ValueError("No validation query has a relevant gallery incident")
    return {"hit_at_1": hits1 / eligible, f"hit_at_{k}": hitsk / eligible,
            "mrr": reciprocal / eligible, "eligible_queries": eligible,
            "query_coverage": eligible / len(query_records)}
