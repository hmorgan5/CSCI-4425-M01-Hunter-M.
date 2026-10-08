# Oxford-IIIT Pet Classification and Segmentation

## 1. Setup, Models, and Training

### Installation and Execution

Create and activate a Python virtual environment:

```bash
python3 -m venv assignment3-env
source assignment3-env/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and prepare the Oxford-IIIT Pet dataset using the instructor-provided script:

```bash
python data.py
```

Run the dataset exploration code:

```bash
python explore_data.py
```

Train and evaluate the regular CNN, ResNet-18, and small U-Net:

```bash
python train_models.py
```

Generate the training-loss and segmentation figures without retraining the models:

```bash
python analyze_results.py
```

### Hardware and Device Configuration

Training and evaluation were performed on an NVIDIA H100 NVL GPU. The program checked GPU availability with `torch.cuda.is_available()` and selected the device with `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

The final run reported:

```text
CUDA available: True
Selected device: cuda
GPU: NVIDIA H100 NVL
```

Each model was moved to the GPU with `model.to(device)`. During classification, the image tensors and cat-or-dog labels were moved to the same device. During segmentation, the image tensors and pixel masks were moved to the same device. Moving the model and required tensors to the same device is necessary because PyTorch cannot directly perform calculations between CPU tensors and CUDA tensors.

### Reproducibility and Training Settings

All models were trained from scratch. ResNet-18 used `weights=None`, so no pretrained ImageNet parameters were loaded.

- Random seed: `0`
- Epochs: `100`
- Batch size: `16`
- Optimizer: Adam
- Learning rate: `0.001`
- Loss function: `nn.CrossEntropyLoss()`
- Processed image size: `128 × 128`
- Classification classes: `0 = Cat`, `1 = Dog`
- Segmentation classes: `0 = Pet`, `1 = Background`, `2 = Boundary`

The regular CNN and ResNet-18 used the same optimizer, learning rate, batch size, number of epochs, training images, and testing images. This gave the two classifiers comparable training budgets. U-Net used the same optimizer settings and epoch count but solved a different task using pixel-level masks as targets.

Raw model scores were passed directly to `CrossEntropyLoss`. Softmax was not applied before the loss because cross-entropy expects unnormalized logits.

### Regular CNN Architecture

The regular CNN contains three convolutional layers. The first convolution converts the three RGB channels into 16 feature maps. The second increases the channel count from 16 to 32, and the third increases it from 32 to 64. Each convolution uses a `3 × 3` kernel and padding of 1, which preserves the image height and width before pooling.

A ReLU activation follows each convolution. Max pooling after the first two convolutional stages reduces the spatial resolution from `128 × 128` to `64 × 64`, and then to `32 × 32`. Adaptive average pooling reduces each final feature map to one value. The resulting 64 values are flattened and passed to a linear layer that produces two classification logits.

```text
Input                         [B, 3, 128, 128]
Conv2d(3, 16) + ReLU          [B, 16, 128, 128]
MaxPool2d(2)                  [B, 16, 64, 64]
Conv2d(16, 32) + ReLU         [B, 32, 64, 64]
MaxPool2d(2)                  [B, 32, 32, 32]
Conv2d(32, 64) + ReLU         [B, 64, 32, 32]
AdaptiveAvgPool2d((1, 1))     [B, 64, 1, 1]
Flatten                       [B, 64]
Linear(64, 2)                 [B, 2]
```

Global average pooling keeps the final classifier relatively small because it avoids connecting every `32 × 32` spatial location to a large fully connected layer. This design reduces the number of trainable parameters, but it also gives the model less representational capacity than ResNet-18.

### ResNet-18 Architecture

ResNet-18 was created with `torchvision.models.resnet18(weights=None)`. Its original fully connected output layer was replaced before creating the optimizer:

```python
model.fc = nn.Linear(model.fc.in_features, 2)
```

The replacement changes the output to two logits, one for cat and one for dog. All parameters remained trainable. Unlike the regular CNN, ResNet-18 contains multiple residual blocks. The residual connections provide direct paths for information and gradients through the network, allowing the deeper architecture to learn more complex image features.

### Small U-Net Architecture

The U-Net performs semantic segmentation rather than whole-image classification. Its output contains three logits at every `128 × 128` pixel location.

Each `DoubleConv` block contains two `3 × 3` convolutions with padding 1 and a ReLU after each convolution. The encoder saves feature maps before downsampling so they can be reused by the decoder through skip connections.

```text
Input                                      [B, 3, 128, 128]
Encoder 1: DoubleConv(3, 16)               [B, 16, 128, 128]
MaxPool2d(2)                               [B, 16, 64, 64]
Encoder 2: DoubleConv(16, 32)              [B, 32, 64, 64]
MaxPool2d(2)                               [B, 32, 32, 32]
Bottleneck: DoubleConv(32, 64)             [B, 64, 32, 32]
Transpose convolution 64 to 32             [B, 32, 64, 64]
Concatenate with Encoder 2                 [B, 64, 64, 64]
Decoder 1: DoubleConv(64, 32)              [B, 32, 64, 64]
Transpose convolution 32 to 16             [B, 16, 128, 128]
Concatenate with Encoder 1                 [B, 32, 128, 128]
Decoder 2: DoubleConv(32, 16)              [B, 16, 128, 128]
Output convolution 16 to 3                 [B, 3, 128, 128]
```

The encoder captures increasingly abstract features while pooling reduces spatial resolution. The decoder restores the original resolution. The two encoder-to-decoder skip connections preserve higher-resolution location and edge information that could otherwise be lost during pooling. The final `1 × 1` convolution converts the 16 decoder features at each pixel into three class logits for pet, background, and boundary.

## 2. Results and Analysis

### Dataset Information

The Oxford-IIIT Pet dataset was downloaded and loaded with the provided `data.py` file.

- Training samples: `3680`
- Testing samples: `3669`
- First image dimensions: `(394, 500)` in `(width, height)` order
- First image classification label: `0`, which represents cat
- First mask dimensions: `(394, 500)` in `(width, height)` order
- First raw mask unique values: `[1, 2, 3]`

The raw images and masks are PIL objects and can have different original dimensions. For training and evaluation, `get_loaders(batch_size=16)` resizes images and masks to `128 × 128`. The images become floating-point RGB tensors, while the masks become integer class-index tensors with values 0, 1, and 2.

### Sample Training Images and Masks

![First three training images and segmentation masks](figures/training_samples.png)

The figure shows the first three raw training images beside their corresponding segmentation masks. All three examples contain cats, but the cats appear at different scales, positions, and orientations. This variation requires the models to learn features that remain useful even when the animal does not occupy the same part of every image.

The first image contains a full cat positioned near the center and lower portion of the image. The corresponding mask follows the visible silhouette, including the ears, back, legs, and tail. The second image is a closer view in which a green foreground object blocks part of the cat's lower face and body. Its mask focuses on the visible head and ears instead of attempting to reconstruct the hidden portion. The third image contains a larger cat whose body extends beyond the right edge of the photograph. Its mask also reaches the image boundary, showing that the dataset contains partially cropped animals.

In the displayed masks, purple marks the pet region, teal marks the background, and yellow marks the boundary surrounding the pet. Treating the boundary as a separate class makes the segmentation task more difficult because U-Net must learn not only the pet's location, but also the transition between the animal and the surrounding scene.

These examples demonstrate why segmentation contains more information than classification. All three images have the same cat label for the classification task, but their masks preserve differences in location, visible shape, cropping, occlusion, and boundary structure.

### Processed Batch and Model Output Shapes

One processed training batch produced these shapes:

```text
Images: torch.Size([16, 3, 128, 128])
Labels: torch.Size([16])
Masks: torch.Size([16, 128, 128])
```

The image tensor contains 16 RGB images. The label tensor contains one cat-or-dog target per image. The mask tensor contains one pet, background, or boundary target for each pixel.

Before training, the same image batch was passed through all three models:

```text
Regular CNN: torch.Size([16, 2])
ResNet-18:   torch.Size([16, 2])
Small U-Net: torch.Size([16, 3, 128, 128])
```

The classifier outputs contain two logits per image. The U-Net output contains three logits per pixel and preserves the input height and width.

### Classification Accuracy

| Model | Test classification accuracy | Difference from baseline |
|---|---:|---:|
| Always-Dog baseline | 67.76% | 0.00 percentage points |
| Regular CNN | 75.44% | +7.68 percentage points |
| ResNet-18 | 81.28% | +13.52 percentage points |

ResNet-18 was the better classifier in this experiment. Its test accuracy was 81.28%, compared with 75.44% for the regular CNN, so ResNet-18 performed better by **5.84 percentage points**.

Both learned classifiers performed better than the always-Dog baseline. The baseline reached 67.76% because 67.76% of the test labels were dogs. The regular CNN exceeded the baseline by 7.68 percentage points, while ResNet-18 exceeded it by 13.52 percentage points. These improvements show that both classifiers learned visual patterns beyond simply predicting the more common class.

The more complex model performed better in this run. ResNet-18's deeper residual architecture improved test accuracy by 5.84 points over the smaller CNN. However, its training loss became almost zero while test accuracy remained 81.28%. The greater model capacity therefore improved generalization relative to the regular CNN, but also produced a much larger difference between training performance and test performance.

### U-Net Pixel Accuracy

The trained U-Net achieved a test pixel accuracy of **84.05%**.

This means that 84.05% of all evaluated test pixels received the same class index as the ground-truth mask. Pixel accuracy measures overall agreement, but it can be influenced by the large background regions in many images. A model can classify much of the background correctly while still producing imprecise pet boundaries or missing smaller pet regions. The segmentation examples are therefore necessary for interpreting this numerical result.

The U-Net pixel accuracy should not be ranked directly against the CNN and ResNet-18 classification accuracies. Classification accuracy counts correctly classified images, whereas U-Net pixel accuracy counts correctly classified pixels. The models also predict different target types.

### Classification Training-Loss Analysis

![Regular CNN and ResNet-18 training-loss curves](figures/classification_training_loss.png)

The regular CNN and ResNet-18 curves follow substantially different patterns. The regular CNN begins with a loss of 0.6355 and decreases gradually to 0.4037 by epoch 100, a reduction of 0.2318. The blue curve has a mostly steady downward slope with small temporary increases. The decline becomes somewhat steeper after approximately epoch 55, but the curve never approaches zero and does not reach a completely flat plateau. The final value is also the lowest reported CNN loss, showing that the model was still improving at the end of training.

ResNet-18 begins slightly higher at 0.6606, but the orange curve falls much faster. Its loss drops below the regular CNN during the first few epochs, reaches 0.3503 by epoch 10, falls below 0.10 by epoch 17, and reaches very small values shortly afterward. Between roughly epochs 20 and 65, the curve contains several sharp but temporary increases. For example, the loss rises to 0.0554 at epoch 48 and 0.0339 at epoch 64 after previously reaching smaller values. These spikes show that the epoch loss did not decrease monotonically, but ResNet-18 repeatedly returned to values near zero.

From approximately epoch 72 through epoch 100, the orange curve visually overlaps the horizontal axis because the reported loss rounds to 0.0000. This indicates that ResNet-18 almost perfectly fit the training samples. In contrast, the regular CNN still had a training loss near 0.40 after the same number of epochs. The large separation between the curves demonstrates the additional fitting capacity of the deeper residual network.

The test results provide important context for the near-zero ResNet-18 training loss. ResNet-18 achieved 81.28% test accuracy, which was 5.84 percentage points better than the regular CNN, but the improvement in test accuracy was much smaller than the difference in final training losses. ResNet-18 therefore benefited from its additional capacity, but the near-perfect fit to the training data did not transfer completely to unseen test images. This gap is evidence of overfitting.

The regular CNN's gradually decreasing loss and 75.44% test accuracy suggest underfitting relative to ResNet-18. The regular CNN continued to improve but lacked enough capacity, training time, or both to fit the training set as closely. However, only training loss was recorded. Without validation-loss curves or validation accuracy after every epoch, the exact point when either model achieved its best generalization cannot be identified.

### U-Net Training-Loss Analysis

![Small U-Net training-loss curve](figures/unet_training_loss.png)

The U-Net training loss decreased from 0.8619 in the first epoch to 0.2992 in the final epoch, a total reduction of 0.5627. The curve is steepest during the early epochs. By epoch 10, the loss had decreased to 0.5795, and by epoch 20 it had reached 0.4707. The model therefore made a large portion of its total improvement during the first 20 epochs.

After the initial rapid decline, the loss continued to decrease more gradually. It reached approximately 0.3854 at epoch 40, 0.3485 at epoch 60, and 0.3190 at epoch 80. The decreasing slope indicates diminishing improvements, but the model did not completely stop learning. Between epochs 60 and 100, the loss still decreased by approximately 0.0493.

The curve contains several small fluctuations during the later epochs. For example, the loss increased from 0.3666 at epoch 49 to 0.3724 at epoch 50, and from 0.3133 at epoch 84 to 0.3228 at epoch 85. These increases were temporary, and the overall direction remained downward. Such fluctuations are consistent with mini-batch training because shuffled batches differ in difficulty from epoch to epoch.

The shape of the curve suggests that U-Net first learned broad differences between pet and background regions and then made slower improvements on harder pixels. Difficult areas likely include thin tails, narrow legs, ears, occluded areas, and the relatively small boundary class. The final loss remaining above zero is consistent with the visible errors in the predicted masks.

Only training loss was recorded, so the curve cannot determine whether test performance improved throughout all 100 epochs. A validation-loss curve would be required to identify when U-Net achieved its best generalization or whether later training caused overfitting.

### Segmentation Examples

![Original test images, ground-truth masks, and U-Net predictions](figures/segmentation_results.png)

The figure compares the first three test images with their ground-truth masks and the masks predicted by the trained U-Net. Purple represents the pet class, teal represents the background class, and yellow represents the boundary class. The ground-truth and predicted masks use the same colormap with `vmin=0` and `vmax=2`, allowing the shapes and class assignments to be compared directly.

The three predictions differ noticeably in quality. The third prediction captures the pet's main body most successfully, the second preserves a recognizable seated silhouette but contains several errors, and the first is highly fragmented. This variation demonstrates why the 84.05% pixel accuracy must be interpreted together with the visual results.

#### Test Image 1

The first test image contains a cat standing in a cluttered indoor environment. The ground-truth mask forms one continuous pet region and outlines the head, back, body, legs, and raised tail. The prediction identifies portions of the cat, but the predicted pet region is divided into several disconnected areas.

The prediction also assigns pet and boundary classes to background objects, especially toward the left, lower, and upper-right areas. Parts of the actual cat are interrupted by pixels classified as background. Although portions of the body and tail are detected, the prediction does not preserve a clean or continuous silhouette.

This image is visually difficult because the cat has dark and patterned fur, while the room contains furniture, shadows, and objects with similar colors and textures. The result shows that U-Net has difficulty separating the pet from clutter when foreground and background features are visually similar.

#### Test Image 2

The second test image contains a cat sitting upright against a building. The ground-truth mask has a clear vertical shape with two ears, a narrow upper body, a wider lower body, and visible legs. The prediction captures the approximate position and seated posture of the cat, making the central pet region more recognizable than in the first example.

However, the prediction contains missing areas within the head and upper body. Disconnected pet and boundary pixels also appear outside the main silhouette, especially above and to the left of the cat. The lower legs and feet are less cleanly separated than in the ground truth, and the yellow boundary is irregular rather than continuous.

The background contains strong vertical edges, dark window regions, and contrasting architectural features. Some of these patterns were incorrectly assigned to the pet or boundary classes. Even with these errors, the main predicted region roughly follows the cat's torso and seated shape.

#### Test Image 3

The third test image produces the strongest segmentation of the three examples. The prediction captures the main body, both ears, and most of the legs. The pet region is largely continuous and matches the general position and scale of the ground-truth silhouette.

The largest error is the tail. In the ground-truth mask, the tail extends clearly toward the left side of the image, but the prediction misses almost the entire tail. The boundary around the legs is also thicker and less precise than the ground truth, and a few isolated pixels appear outside the main pet region.

This example suggests that U-Net identifies large, connected body regions more reliably than narrow structures. The torso occupies a large and visually consistent region, while the thin tail, leg gaps, and detailed edges are more difficult to preserve after resizing and convolutional processing.

#### Overall Segmentation Interpretation

Across the three examples, U-Net can identify the approximate location and broad shape of a pet, but performance decreases with cluttered backgrounds, overlapping objects, dark regions, and thin body parts. Large connected regions such as the torso are predicted more consistently than tails, legs, small gaps, and detailed boundaries.

The boundary class appears especially difficult. The ground-truth masks contain a relatively continuous yellow outline, while the predicted boundaries are often uneven, interrupted, too thick, or placed around incorrect regions. The boundary occupies fewer pixels than the pet and background classes and requires greater spatial precision, making it more difficult to learn.

The visual results help explain how U-Net achieved 84.05% pixel accuracy despite the poor first prediction. Background pixels occupy a large portion of each image, and many background pixels are correct. Overall pixel accuracy can therefore remain high even when visible mistakes occur in the smaller pet and boundary regions. The second and third examples show that U-Net learned meaningful pet shapes, but the fragmented first result demonstrates that the score does not guarantee consistently accurate segmentation.

Segmentation reveals information that classification cannot provide. The CNN and ResNet-18 return only one cat-or-dog prediction for an entire image. U-Net shows where the model believes the pet is located, whether the body forms a continuous shape, and which features are missed or confused with the background. The segmentation masks expose errors involving tails, legs, boundaries, clutter, and disconnected regions that would remain hidden in image-level classification accuracy.

### Training Time

The measured training times were:

- Regular CNN: `18.83 minutes`
- ResNet-18: `27.30 minutes`
- Small U-Net: `21.22 minutes`
- Total model-training time: approximately `67.35 minutes`

ResNet-18 took the longest because it contains a deeper sequence of residual blocks and performs more operations per image than the regular CNN. U-Net performs dense pixel-level prediction and maintains high-resolution decoder features, but its small channel configuration kept its measured training time below ResNet-18 in this experiment.

### Saved Outputs

The completed experiment produced:

```text
cnn_model.pth
resnet18_model.pth
unet_model.pth
training_results.pth
figures/training_samples.png
figures/classification_training_loss.png
figures/unet_training_loss.png
figures/segmentation_results.png
```

The three model files contain the final trained parameters. `training_results.pth` stores the training-loss histories, evaluation values, classifier predictions, test labels, random seed, batch size, learning rate, and epoch count. The analysis script loaded the saved outputs, confirmed that each loss history contained 100 epochs, and generated the analysis figures successfully.

### Generative AI Acknowledgment

I used Microsoft 365 Copilot, based on the GPT-5 reasoning model, to help interpret the assignment requirements, structure portions of the PyTorch model, training, evaluation, and plotting code, debug implementation and environment errors, and organize portions of this README. I reviewed and tested the code, downloaded and explored the assigned dataset, trained the models, generated the figures, and based the reported measurements and interpretations on my actual experiment outputs.
