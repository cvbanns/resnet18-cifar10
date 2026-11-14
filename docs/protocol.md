# Evaluation protocol

## Dataset and model

Use the official CIFAR-10 training set. The stratified split selects 500
validation images per class with NumPy generator seed 2026. The remaining
45,000 images supply the per-channel mean and standard deviation.

Train with four pixels of zero padding, a per-image random 32×32 crop,
and a random horizontal flip. Normalize after augmentation. Cutout fills
a clipped 16×16 patch centered at a random pixel with normalized zeros.
Validation and standard test inference use normalization alone.

The reference architecture uses a 3×3 stride-one stem and eight two-convolution
residual blocks across widths 64, 128, 256, and 512. The first block of each
new width downsamples using stride two and a projection shortcut. Global average
pooling feeds a ten-class float32 logit layer. Batch normalization uses momentum
0.9 and epsilon 1e-5. The convolutions use He normal initialization.

## Training matrix

All configurations run 200 epochs at batch size 128 with seeds 17, 29, 43,
61, and 83. The reference uses SGD with 0.9 momentum, cosine decay from 0.1,
and kernel L2 of 0.00025, equivalent to a coupled decay coefficient of 0.0005.

| Experiment | Change from reference |
| --- | --- |
| resnet18 | Reference |
| resnet18_step | Learning-rate drops by 10× at epochs 100 and 150 |
| resnet18_adam | Adam at 0.001 with cosine decay |
| resnet18_no_skip | Remove identity and projection shortcuts |
| resnet18_imagenet_stem | 7×7 stride-two convolution and stride-two max pooling |
| resnet18_no_aug | Disable crop and flip |
| resnet18_wd1e4 | Weight decay 0.0001 |
| resnet18_cutout | Add Cutout |
| resnet18_ls | Label smoothing 0.1 |
| resnet18_cutout_ls | Combine Cutout and label smoothing |

The combined row also supports comparisons with each of the two single-factor
regularization rows. The earlier CNN score is an external comparison across
training recipes. The optional plain-CNN SGD bridge is outside this grid.

## Selection and reporting

Keep the earliest checkpoint attaining the highest validation accuracy and the
final epoch checkpoint. Select the configuration by mean best validation
accuracy across five seeds. Evaluate its five best checkpoints on the official
test set. Compute horizontal-flip test-time augmentation in the same pass,
using mean probabilities and reporting its score separately.

Each `validation.json` stores epoch metrics, timings, hardware, validation
split fingerprint, and the best and final validation scores. Each selected
`test.json` stores standard and flip-augmented accuracy, per-class accuracy,
and a confusion matrix in CIFAR-10 class order.

Architecture reference: [He et al., Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
