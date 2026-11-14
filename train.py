"""Run the ResNet-18 study with epoch-level restart checkpoints."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from tensorflow import keras

import data
import resnet

SEEDS = (17, 29, 43, 61, 83)
EPOCHS = 200
DEFAULTS = dict(stem="cifar", skip=True, augment=True, cutout=False,
                label_smoothing=0.0, weight_decay=0.0005, optimizer="sgd",
                schedule="cosine", learning_rate=0.1)
CONFIGS = {name: {**DEFAULTS, **overrides} for name, overrides in json.loads(
    Path(__file__).with_name("experiments.json").read_text(encoding="utf-8")).items()}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def compile_model(config, epochs, steps):
    model = resnet.build(config)
    peak = config["learning_rate"]
    schedules = keras.optimizers.schedules
    if config["schedule"] == "step":
        lr = schedules.PiecewiseConstantDecay([100 * steps - 1, 150 * steps - 1],
                                                [peak, peak / 10, peak / 100])
    else:
        lr = schedules.CosineDecay(peak, epochs * steps)
    if config["optimizer"] == "adam":
        optimizer = keras.optimizers.Adam(lr)
    else:
        optimizer = keras.optimizers.SGD(lr, momentum=0.9)
    smoothing = config["label_smoothing"]
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if smoothing:
        categorical_loss = keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=smoothing)

        def loss(labels, logits):
            labels = tf.one_hot(tf.cast(tf.reshape(labels, [-1]), tf.int32), 10)
            return categorical_loss(labels, logits)

    model.compile(optimizer, loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy(
        name="accuracy")], steps_per_execution=16)
    return model


def fit(root, name, seed, epochs, train_data, val_data, stats, fingerprint):
    run = root / name / str(seed)
    run.mkdir(parents=True, exist_ok=True)
    config = CONFIGS[name]
    steps = (len(train_data[1]) + data.BATCH - 1) // data.BATCH
    model = compile_model(config, epochs, steps)
    parameters = sum(int(tf.size(weight)) for weight in model.trainable_weights)
    if name == "resnet18" and parameters != 11_173_962:
        raise RuntimeError(f"Expected 11,173,962 trainable parameters, got {parameters}")
    metadata = dict(experiment=name, seed=seed, config=config, epochs=epochs,
                    split_seed=data.SPLIT_SEED, validation_indices_sha256=fingerprint,
                    normalization=stats, batch_size=data.BATCH, parameters=parameters,
                    precision=keras.mixed_precision.global_policy().name,
                    tensorflow=tf.__version__,
                    gpu=[tf.config.experimental.get_device_details(gpu).get("device_name")
                         for gpu in tf.config.list_physical_devices("GPU")])
    manifest = run / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError(f"Existing run settings differ in {run}")
    write_json(manifest, metadata)
    if (run / "validation.json").exists():
        return
    validation = data.batches(*val_data, stats)
    state = tf.Variable("[]", dtype=tf.string, trainable=False)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer, history=state)
    manager = tf.train.CheckpointManager(checkpoint, str(run / "resume"), max_to_keep=2)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
    history = json.loads(state.numpy())
    best = max(range(len(history)), key=lambda i: history[i]["val_accuracy"]) if history else None
    if best is not None and not (run / f"best-{best + 1}.h5").exists():
        raise FileNotFoundError(f"Missing best checkpoint in {run}")
    for epoch in range(len(history), epochs):
        start = perf_counter()
        training = data.batches(*train_data, stats, config, seed, epoch)
        metrics = model.fit(training, initial_epoch=epoch, epochs=epoch + 1,
                            validation_data=validation, verbose=2).history
        if int(model.optimizer.iterations) != (epoch + 1) * steps:
            raise RuntimeError(f"Unexpected optimizer step count in {run}")
        row = {key: float(values[0]) for key, values in metrics.items()}
        row["seconds"] = round(perf_counter() - start, 2)
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"Non-finite metrics in {name}, seed {seed}, epoch {epoch + 1}")
        history.append(row)
        if best is None or row["val_accuracy"] > history[best]["val_accuracy"]:
            best = epoch
            model.save_weights(str(run / f"best-{best + 1}.h5"))
        state.assign(json.dumps(history))
        manager.save(checkpoint_number=epoch + 1)
        for path in run.glob("best-*.h5"):
            if path.name != f"best-{best + 1}.h5":
                path.unlink()
    write_json(run / "validation.json", {
        **metadata, "best_epoch": best + 1, "validation_accuracy": history[best]["val_accuracy"],
        "final_validation_accuracy": history[-1]["val_accuracy"],
        "seconds": round(sum(row["seconds"] for row in history), 2),
        "history": {key: [row[key] for row in history] for key in history[0]},
    })


def evaluate(root, archive):
    rows = {name: [json.loads((root / name / str(seed) / "validation.json").read_text())
                   for seed in SEEDS] for name in CONFIGS}
    for name, group in rows.items():
        for seed, row in zip(SEEDS, group):
            if (row["config"], row["seed"], row["epochs"]) != (CONFIGS[name], seed, EPOCHS):
                raise ValueError(f"Incompatible metrics for {name} seed {seed}")
    means = {name: statistics.mean(row["validation_accuracy"] for row in group)
             for name, group in rows.items()}
    name = max(means, key=means.get)
    selected = dict(experiment=name, validation_means=means)
    manifest = root / "selection.json"
    if manifest.exists() and json.loads(manifest.read_text()) != selected:
        raise ValueError("Existing test selection differs")
    write_json(manifest, selected)
    images, labels = data.load(archive, test=True)
    results = []
    for seed, row in zip(SEEDS, rows[name]):
        run = root / name / str(seed)
        target = run / "test.json"
        if not target.exists():
            keras.backend.clear_session()
            model = resnet.build(CONFIGS[name])
            model.load_weights(str(run / f"best-{row['best_epoch']}.h5"))
            batches = data.batches(images, labels, row["normalization"])
            plain, augmented = [], []
            for pixels, _ in batches:
                logits = model(pixels, training=False)
                flipped = model(tf.reverse(pixels, [2]), training=False)
                plain.extend(tf.argmax(logits, axis=1).numpy())
                probabilities = tf.nn.softmax(logits) + tf.nn.softmax(flipped)
                augmented.extend(tf.argmax(probabilities, axis=1).numpy())
            matrix = tf.math.confusion_matrix(labels, plain, num_classes=10).numpy()
            write_json(target, dict(
                experiment=name, seed=seed, best_epoch=row["best_epoch"],
                test_accuracy=float(np.trace(matrix) / matrix.sum()),
                per_class_accuracy=(matrix.diagonal() / matrix.sum(axis=1)).tolist(),
                confusion_matrix=matrix.tolist(),
                flip_tta_accuracy=float(np.mean(labels == augmented)),
            ))
        results.append(json.loads(target.read_text()))
    summary = {}
    for metric in ("test_accuracy", "flip_tta_accuracy"):
        values = [row[metric] for row in results]
        summary[metric] = {"mean": statistics.mean(values), "sd": statistics.stdev(values)}
    write_json(root / "test-summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "test", "grid"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--experiment", choices=CONFIGS, default="resnet18")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--float32", action="store_true")
    args = parser.parse_args()
    if args.mode == "grid":
        command = [sys.executable, "-u", __file__]
        common = ["--archive", str(args.archive), "--run-dir", str(args.run_dir)]
        if args.float32:
            common.append("--float32")
        for name in CONFIGS:
            for seed in SEEDS:
                completed = args.run_dir / name / str(seed) / "validation.json"
                if completed.exists():
                    row = json.loads(completed.read_text())
                    if (row["config"], row["seed"], row["epochs"]) != (CONFIGS[name], seed, EPOCHS):
                        raise ValueError(f"Incompatible completed run in {completed}")
                    continue
                subprocess.run([*command, "train", *common, "--experiment", name,
                                "--seed", str(seed)], check=True)
        subprocess.run([*command, "test", *common], check=True)
        return
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if not args.float32:
        keras.mixed_precision.set_global_policy("mixed_float16")
    if args.mode == "test":
        evaluate(args.run_dir, args.archive)
        return
    images, labels = data.load(args.archive)
    train, val = data.split(labels)
    train_data, val_data = (images[train], labels[train]), (images[val], labels[val])
    stats = data.normalize_stats(train_data[0])
    fingerprint = hashlib.sha256(val.astype("<i8").tobytes()).hexdigest()
    keras.utils.set_random_seed(args.seed)
    print(f"Training {args.experiment}, seed {args.seed}", flush=True)
    smoke = args.mode == "smoke"
    root = args.run_dir / "smoke" if smoke else args.run_dir
    fit(root, args.experiment, args.seed, 2 if smoke else EPOCHS,
        train_data, val_data, stats, fingerprint)


if __name__ == "__main__":
    main()
