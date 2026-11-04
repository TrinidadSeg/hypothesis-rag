"""Generate deliberately easy synthetic data for exercising the pipeline, not RCA evidence."""
import json
import random
from pathlib import Path

rng = random.Random(7)
destination = Path(__file__).parent / "data"
destination.mkdir(exist_ok=True)
for split, count in [("train", 12), ("val", 4), ("test", 4)]:
    with (destination / f"{split}.jsonl").open("w") as output:
        for label in range(4):
            for i in range(count):
                graphs = []
                for view in range(rng.randint(1, 3)):
                    features = [[float(j == label) + rng.gauss(0, 0.1) for j in range(4)]
                                for _ in range(rng.randint(3, 7))]
                    graphs.append({"node_features": features,
                                   "edges": [[rng.randrange(j), j] for j in range(1, len(features))],
                                   "weight": 1.0})
                output.write(json.dumps({"incident_id": f"{split}-{label}-{i}",
                                         "equivalence_id": str(label), "graphs": graphs}) + "\n")
print(destination)
