import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .data import ClassBatchSampler, collate, read_records
from .model import IncidentEncoder, ModelConfig
from .training import embed, retrieval_metrics, supervised_contrastive_loss


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["model_config"])
    config.setdefault("encoder", "directed")  # Version 0.1 checkpoints used message passing.
    model = IncidentEncoder(ModelConfig(**config))
    model.load_state_dict(checkpoint["model"])
    return model.to(device)


def train(args):
    if min(args.epochs, args.steps_per_epoch, args.eval_batch_size) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Epochs, steps, batch size and learning rate must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records, node_dim = read_records(args.train)
    validation, _ = read_records(args.val, node_dim)
    overlap = {r["incident_id"] for r in records} & {r["incident_id"] for r in validation}
    if overlap:
        raise ValueError(f"Training/validation incident IDs overlap: {sorted(overlap)[:5]}")
    # Reject common upstream leakage when callers supply family/group metadata.
    groups = {r["leakage_group"] for r in records if r.get("leakage_group") is not None}
    if groups & {r["leakage_group"] for r in validation if r.get("leakage_group") is not None}:
        raise ValueError("Training/validation leakage_group values overlap")
    sampler = ClassBatchSampler(records, args.classes_per_batch, args.samples_per_class, args.seed)
    if not set(sampler.groups) & {r["equivalence_id"] for r in validation}:
        raise ValueError("Validation needs at least one equivalence class present in the training gallery")
    class_ids = {c: i for i, c in enumerate(sampler.groups)}
    config = ModelConfig(node_dim, args.hidden_dim, args.embedding_dim, args.graph_layers, args.set_pool, args.encoder)
    model = IncidentEncoder(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new run directory")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.steps_per_epoch):
            selected = [records[i] for i in sampler.sample()]
            labels = torch.tensor([class_ids[r["equivalence_id"]] for r in selected], device=args.device)
            optimizer.zero_grad(set_to_none=True)
            batch = collate(selected, args.device)
            # One differentiable forward path: trees -> explanations -> incidents.
            explanation_vectors = model.graph_encoder(batch)
            incident_vectors = model.set_encoder(explanation_vectors, batch)
            loss = supervised_contrastive_loss(incident_vectors, labels, args.temperature)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; inspect feature scales and learning rate")
            loss.backward()  # Gradients flow through set pooling into the tree encoder.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()
        gallery = embed(model, records, args.device, args.eval_batch_size)
        queries = embed(model, validation, args.device, args.eval_batch_size)
        metrics = retrieval_metrics(queries, validation, gallery, records)
        metrics.update(epoch=epoch, loss=total / args.steps_per_epoch,
                       embedding_std=gallery.std(dim=0, unbiased=False).mean().item())
        print(json.dumps(metrics), flush=True)
        with (output / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(metrics) + "\n")
        checkpoint = {"model_config": asdict(config), "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics}
        torch.save(checkpoint, output / "last.pt")
        if metrics["mrr"] > best:
            best = metrics["mrr"]
            torch.save(checkpoint, output / "best.pt")


def export(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = load_model(args.checkpoint, args.device)
    records, _ = read_records(args.input, model.config.node_dim, labeled=False)
    vectors = embed(model, records, args.device, args.batch_size)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as out:
        for record, vector in zip(records, vectors):
            out.write(json.dumps({"incident_id": record["incident_id"], "embedding": vector.tolist()}) + "\n")


def evaluate(args):
    model = load_model(args.checkpoint, args.device)
    queries, _ = read_records(args.queries, model.config.node_dim)
    gallery, _ = read_records(args.gallery, model.config.node_dim)
    print(json.dumps(retrieval_metrics(
        embed(model, queries, args.device), queries,
        embed(model, gallery, args.device), gallery), indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train graph + explanation-set encoders")
    commands = parser.add_subparsers(dest="command", required=True)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    fit = commands.add_parser("train")
    fit.add_argument("--train", required=True)
    fit.add_argument("--val", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", default=default_device)
    fit.add_argument("--hidden-dim", type=int, default=128)
    fit.add_argument("--embedding-dim", type=int, default=128)
    fit.add_argument("--graph-layers", type=int, default=3)
    fit.add_argument("--encoder", choices=["tree", "directed"], default="tree",
                     help="tree: bottom-up recurrence; directed: previous fixed-depth baseline")
    fit.add_argument("--set-pool", choices=["deepsets", "attention"], default="deepsets")
    fit.add_argument("--classes-per-batch", type=int, default=8)
    fit.add_argument("--samples-per-class", type=int, default=4)
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--steps-per-epoch", type=int, default=100)
    fit.add_argument("--eval-batch-size", type=int, default=32)
    fit.add_argument("--lr", type=float, default=3e-4)
    fit.add_argument("--weight-decay", type=float, default=1e-4)
    fit.add_argument("--temperature", type=float, default=0.1)
    fit.add_argument("--seed", type=int, default=42)
    encoding = commands.add_parser("embed")
    encoding.add_argument("--checkpoint", required=True)
    encoding.add_argument("--input", required=True)
    encoding.add_argument("--output", required=True)
    encoding.add_argument("--batch-size", type=int, default=32)
    encoding.add_argument("--device", default=default_device)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--queries", required=True)
    evaluation.add_argument("--gallery", required=True)
    evaluation.add_argument("--device", default=default_device)
    args = parser.parse_args()
    {"train": train, "embed": export, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
