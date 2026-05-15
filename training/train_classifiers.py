import argparse
import csv
from pathlib import Path

from ml.classifiers import DEFAULT_FEATURE_COLUMNS


TARGETS = {
    "fatigue": "fatigue_state",
    "attention": "attention_state",
    "tension": "tension_state",
}


def load_rows(paths):
    rows = []
    for root in paths:
        for path in Path(root).glob("dataset_*.csv"):
            with path.open(newline="") as f:
                rows.extend(csv.DictReader(f))
    return rows


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def train(args):
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    rows = load_rows(args.inputs)
    if not rows:
        raise SystemExit("No dataset_*.csv rows found.")

    feature_columns = [col for col in DEFAULT_FEATURE_COLUMNS if any(col in row for row in rows)]
    x = [[as_float(row.get(col)) for col in feature_columns] for row in rows]
    models = {}
    for signal, target in TARGETS.items():
        labeled = [(features, row.get(target)) for features, row in zip(x, rows) if row.get(target)]
        labels = sorted({label for _, label in labeled})
        if len(labels) < 2:
            continue
        train_x = [features for features, _ in labeled]
        train_y = [label for _, label in labeled]
        model = RandomForestClassifier(n_estimators=args.trees, random_state=42, class_weight="balanced")
        model.fit(train_x, train_y)
        models[signal] = model

    if not models:
        raise SystemExit("No target had at least two classes; collect more labeled data.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"feature_columns": feature_columns, "models": models}, out)
    print(f"Saved {len(models)} classifier(s) to {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Human Signal classifiers from datasets/dataset_*.csv files.")
    parser.add_argument("inputs", nargs="*", default=["datasets"], help="Dataset directories.")
    parser.add_argument("--output", default="models/classifiers.joblib")
    parser.add_argument("--trees", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
