# ResNet-18 on CIFAR-10

A standalone controlled ablation study of CIFAR-style ResNet-18, trained from scratch.
The earlier [plain-CNN study](https://github.com/cvbanns/cifar10-cnn-ablation-study/pull/1)
achieved 87.51% mean test accuracy. The same validation split is reused here, and
three of the five training seeds overlap.

## Status

The two-epoch smoke test passed on an NVIDIA RTX 2070 Max-Q with 11,173,962
trainable parameters. The 50-run grid is in progress. Test metrics will be
available after configuration selection.

## Reproduce

Use Python 3.9 and install the dependencies. For native Windows GPU execution,
the pinned TensorFlow build uses CUDA 11.2 and cuDNN 8.1. Mixed precision is
enabled by default, with float32 logits. TensorFlow operation determinism is off.

Download the [official Python archive](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
as `cifar-10-python.tar.gz`. The loader verifies the archive checksum.

```sh
python -m pip install -r requirements.txt
python train.py smoke --archive cifar-10-python.tar.gz
python train.py grid --archive cifar-10-python.tar.gz
python report.py --run-dir runs
```

Restart the grid with the same command to resume from the last complete epoch.
For multiple GPUs, distribute separate `train --experiment NAME --seed SEED`
jobs and run `test` after all training metrics and checkpoints are collected in
one run directory. Use a single worker for each experiment and seed.

The grid uses seeds 17, 29, 43, 61, and 83 and runs every configuration for 200
epochs. The [protocol](docs/protocol.md) lists the controls and scoring rules.
`report.py` generates the measured results table and figures from the run files.
