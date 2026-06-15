# Project 2 Report: Neural Network and Deep Learning

**Name:** TODO  
**Student ID:** TODO  
**Code link:** TODO  
**Dataset link:** TODO  
**Trained model weights link:** TODO

## 1. Task Overview

This project trains neural networks on CIFAR-10 and studies the effect of Batch
Normalization. CIFAR-10 contains 60,000 color images with size 32x32 from 10
classes. I implemented and compared a baseline VGG-A model and a VGG-A model
with Batch Normalization.

## 2. Network Structure

The baseline model is VGG-A adapted for CIFAR-10. It contains 2D convolutional
layers, ReLU activations, max pooling layers, and fully connected layers. The
BatchNorm model inserts a `BatchNorm2d` layer after each convolution and before
the ReLU activation.

| Model | Extra component | Parameter count | Best validation accuracy | Test error |
| --- | --- | ---: | ---: | ---: |
| VGG-A | None | Fill from `reports/tables/summary.csv` | TODO | TODO |
| VGG-A-BN | BatchNorm | Fill from `reports/tables/summary.csv` | TODO | TODO |

## 3. Training Settings

The models are trained with cross entropy loss and Adam optimizer. Weight decay
is used as L2 regularization. The default learning rates are `1e-3`, `2e-3`,
`1e-4`, and `5e-4`. Batch size is 128 and the default number of epochs is 20.

To reproduce the experiment:

```bash
cd codes/VGG_BatchNorm
python VGG_Loss_Landscape.py --epochs 20 --batch-size 128 --device cuda:0
```

## 4. CIFAR-10 Performance

Insert `codes/VGG_BatchNorm/reports/figures/training_curves.png` here.

Discuss:

- Which model obtains the best validation/test error.
- Whether BatchNorm improves convergence speed.
- Whether BatchNorm improves final accuracy.
- How learning rate affects stability and final performance.

## 5. Batch Normalization Comparison

Batch Normalization normalizes channel-wise activations during training and
learns affine scale and shift parameters. In the VGG-A-BN model, this usually
stabilizes the distribution of hidden activations and makes optimization less
sensitive to the learning rate.

From the training curves, compare VGG-A and VGG-A-BN:

- VGG-A-BN should have smoother loss decrease.
- VGG-A-BN should usually reach good accuracy earlier.
- VGG-A may be more sensitive to larger learning rates.

Replace these statements with the actual observations after running the script.

## 6. Loss Landscape

Insert `codes/VGG_BatchNorm/reports/figures/loss_landscape_comparison.png` here.

The loss landscape is visualized by training the same architecture with several
learning rates and plotting the minimum and maximum loss at each training step.
The filled region shows the variation of the loss across runs. A narrower and
smoother region indicates a more stable optimization landscape.

Discuss:

- Whether the BatchNorm model has a smaller loss variation band.
- Whether large learning rates cause spikes for the baseline model.
- Whether gradient norm records in `reports/tables/*_grad_norms.txt` support the
  same conclusion.

## 7. Additional Optimization Attempts

The required optimization attempts are addressed as follows:

The standalone runner writes these results to
`pj2_outputs/tables/ablation_summary.csv`.

- Different filters/architectures: compare `filters_width_0.5` and
  `filters_width_1.0`.
- Different regularization/loss: compare `regularization_no_weight_decay` and
  `loss_label_smoothing_0.1`.
- Different activations: compare `relu`, `leaky_relu`, and `elu`.
- Different optimizers: compare Adam and SGD with momentum.

## 8. Conclusion

Summarize the best model, best test error, and the main insight from BatchNorm.
The expected conclusion is that BatchNorm improves optimization stability and
often speeds up convergence on CIFAR-10. Use the actual numbers from the final
run in the submitted PDF.
