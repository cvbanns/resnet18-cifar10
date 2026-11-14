# ResNet-18 on CIFAR-10

A standalone controlled ablation study of CIFAR-style ResNet-18, trained from scratch.
The earlier [plain-CNN study](https://github.com/cvbanns/cifar10-cnn-ablation-study/pull/1)
achieved 87.51% mean test accuracy. The same validation split is reused here, and
three of the five training seeds overlap.

## Results

All 50 runs completed 200 epochs. Validation selection chose ResNet-18 with
Cutout and label smoothing, which achieved **95.90% ± 0.13 percentage points**
mean official test accuracy across five seeds. Horizontal-flip test-time
augmentation achieved **96.27% ± 0.08 points**.

The standard test result improves on the earlier plain CNN's 87.51% by
**8.39 percentage points**. These studies use different training recipes and
seed counts. The first three seeds are shared.

| Configuration | Mean best validation accuracy |
| --- | ---: |
| Standard ResNet-18 | 95.32% |
| Step schedule | 95.08% |
| Adam | 94.07% |
| No residual connections | 94.91% |
| ImageNet stem | 87.86% |
| No augmentation | 88.84% |
| Weight decay 0.0001 | 94.80% |
| Cutout | 96.08% |
| Label smoothing | 95.24% |
| Cutout + label smoothing | **96.14%** |

See the [full results](results/README.md) for standard deviations, per-seed
test scores, per-class accuracy, curves, and confusion matrices. The
[run records](results/runs) include all training histories and test evaluations.

Runs used an NVIDIA RTX 2070 Max-Q. System memory exhaustion interrupted two
runs, which resumed from their last complete epoch checkpoints. TensorFlow
operation determinism was off, so exact bitwise reproduction is not guaranteed.

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
