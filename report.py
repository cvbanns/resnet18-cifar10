"""Create a results table and figures from completed runs."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog",
           "frog", "horse", "ship", "truck")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score(values):
    return f"{statistics.mean(values):.2%} ± {statistics.stdev(values) * 100:.2f} pp"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    selected = read(args.run_dir / "selection.json")
    winner = selected["experiment"]
    summary = read(args.run_dir / "test-summary.json")
    args.output.mkdir(parents=True, exist_ok=True)
    lines = ["# ResNet-18 results", "", f"Selected configuration: `{winner}`.",
             "", "| Configuration | Best validation | Final validation |",
             "| --- | ---: | ---: |"]
    figure, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for name in selected["validation_means"]:
        runs = [read(args.run_dir / name / str(seed) / "validation.json")
                for seed in (17, 29, 43, 61, 83)]
        best = score([run["validation_accuracy"] for run in runs])
        final = score([run["final_validation_accuracy"] for run in runs])
        lines.append(f"| {name} | {best} | {final} |")
        values = np.mean([run["history"]["val_accuracy"] for run in runs], axis=0)
        ax.plot(np.arange(1, len(values) + 1), 100 * values, label=name)
    ax.set(xlabel="Epoch", ylabel="Mean validation accuracy (%)")
    ax.legend(fontsize=8)
    figure.savefig(args.output / "validation-curves.png", dpi=160)
    plt.close(figure)

    lines.extend(["", "| Seed | Test accuracy | Flip TTA accuracy |",
                  "| ---: | ---: | ---: |"])
    matrix = np.zeros((10, 10), dtype=np.int64)
    for seed in (17, 29, 43, 61, 83):
        row = read(args.run_dir / winner / str(seed) / "test.json")
        lines.append(f"| {seed} | {row['test_accuracy']:.2%} | {row['flip_tta_accuracy']:.2%} |")
        matrix += np.asarray(row["confusion_matrix"])
    for name, label in (("test_accuracy", "Mean standard test accuracy"),
                        ("flip_tta_accuracy", "Mean flip TTA accuracy")):
        result = summary[name]
        lines.extend(["", f"{label} is **{result['mean']:.2%} ± {result['sd'] * 100:.2f} pp**."])
    delta = (summary["test_accuracy"]["mean"] - 0.8751) * 100
    lines.extend(["", f"The standard test score is {delta:+.2f} percentage points relative "
                  "to the earlier plain CNN (87.51%).", "", "| Class | Mean test accuracy |",
                  "| --- | ---: |"])
    for name, accuracy in zip(CLASSES, matrix.diagonal() / matrix.sum(axis=1)):
        lines.append(f"| {name} | {accuracy:.2%} |")
    lines.extend(["", "![Validation curves](validation-curves.png)", "",
                  "![Test confusion matrix](confusion-matrix.png)"])
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    image = ax.imshow(matrix / matrix.sum(axis=1, keepdims=True), vmin=0, vmax=1)
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="True class", title="Test confusion across five seeds")
    ax.tick_params(axis="x", labelrotation=45)
    figure.colorbar(image, ax=ax, label="Fraction of true class")
    figure.savefig(args.output / "confusion-matrix.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
