# Training Guide for 3D U-Net Cell Detection

## Overview

This guide covers training the 3D U-Net model for cell center detection using the Biohub zebrafish embryo dataset.

## Installation

Install the deep learning dependencies:

```bash
cd biohub-cell-tracking
python -m pip install -e ".[dl]"
```

This installs:
- PyTorch and torchvision
- MONAI (for 3D U-Net)
- TorchIO (for medical augmentations)
- Optuna (for hyperparameter optimization)

## Architecture

### 3D U-Net Model

The `CellDetectionUNet` uses MONAI's 3D U-Net with:
- **Input**: Single-channel 3D volumes (Z, Y, X)
- **Output**: 3-channel probability maps (background, center, periphery)
- **Architecture**: 5-level encoder-decoder with residual connections
- **Normalization**: Instance normalization
- **Activation**: ReLU

### Label Generation

The `WeightedLabelGenerator` converts sparse GEFF centroids to training labels:
- **Center sphere**: 2-voxel radius around each centroid (label=1)
- **Periphery sphere**: 4-voxel radius excluding center (label=2)
- **Background**: All other voxels (label=0)
- **Class weights**: Center=10.0, Periphery=5.0, Background=1.0

### Anisotropic Augmentation

The `TrainingAugmentation` pipeline respects physical anisotropy (Z=1.625μm, XY=0.40625μm):
- **XY plane**: Rotations (±15°), flips, elastic deformations
- **Z axis**: Flips only (no rotation to preserve anisotropy)
- **Intensity**: Contrast (0.8-1.2), brightness (±0.1), Gaussian noise
- **Blur**: Anisotropic Gaussian blur (more in Z)
- **Z-scale**: Random scaling (0.9-1.1) to simulate focus variation

## Training Configuration

Edit `configs/training.yaml` to customize training:

```yaml
# Model architecture
model:
  in_channels: 1
  out_channels: 3
  channels: [32, 64, 128, 256, 512]
  strides: [2, 2, 2, 2]
  num_res_units: 2
  dropout: 0.1

# Training parameters
training:
  device: "auto"
  epochs: 100
  batch_size: 2
  learning_rate: 1e-4
  weight_decay: 1e-5
  grad_clip: 1.0
  num_workers: 4
  use_amp: true

# Loss function
loss:
  dice_weight: 0.5
  focal_weight: 0.5
  focal_gamma: 2.0

# Augmentation
augmentation:
  scale: [1.625, 0.40625, 0.40625]
  xy_rotation_range: 15.0
  crop_size: [32, 128, 128]
```

## Running Training

### Basic Training

```bash
python scripts/train_unet.py \
    --config configs/training.yaml \
    --data-root ../train
```

### Resume from Checkpoint

```bash
python scripts/train_unet.py \
    --config configs/training.yaml \
    --data-root ../train \
    --resume outputs/checkpoints/best_model.pt
```

### Custom Validation Split

Edit `configs/training.yaml`:

```yaml
validation:
  val_split: 0.2  # Use 20% for validation
  # OR specify specific samples
  val_samples: ["6bba_fe670320", "44b6_0113de3b"]
```

## Training Pipeline

The training script automatically:
1. Splits samples into train/validation sets
2. Creates lazy-loading datasets with LRU caching
3. Applies anisotropic augmentations during training
4. Uses combined Dice + Focal loss for imbalanced classes
5. Implements mixed precision training (AMP)
6. Saves checkpoints every 5 epochs
7. Implements early stopping with patience=15
8. Reduces learning rate on plateau

## Monitoring Training

Training progress is logged with:
- **Loss**: Combined Dice + Focal loss per epoch
- **Learning rate**: Current learning rate
- **Checkpoints**: Saved to `outputs/checkpoints/`
- **History**: Training history saved to `training_history.yaml`

## Using Trained Model

After training, use the model in the pipeline:

```yaml
# configs/unet_pipeline.yaml
detection:
  method: "unet"
  unet:
    model_path: "outputs/checkpoints/best_model.pt"
    device: "cuda"
    center_threshold: 0.5
    periphery_threshold: 0.3
    min_distance: 3
    use_refinement: true

tracking:
  method: "hungarian"
  max_distance_um: 7.0
```

Run inference:

```bash
biohub run classical_pipeline \
    --config configs/unet_pipeline.yaml \
    --data-root ../train \
    --sample 6bba_fe670320
```

## Memory Optimization

For limited GPU memory:

1. **Reduce batch size**: Set `batch_size: 1` in config
2. **Smaller crop size**: Reduce `crop_size` to `[16, 64, 64]`
3. **Fewer channels**: Reduce `channels` to `[16, 32, 64, 128]`
4. **Disable AMP**: Set `use_amp: false` (slower but less memory)

## Performance Tips

1. **Start with classical baseline**: Verify classical pipeline works first
2. **Small model first**: Train with smaller architecture to debug
3. **Monitor overfitting**: Check train vs val loss gap
4. **Adjust class weights**: If detection is poor, increase center_weight
5. **Tune augmentation**: If overfitting, increase augmentation strength
6. **Use pretrained weights**: Initialize from similar tasks if available

## Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce crop size
- Reduce model channels
- Enable gradient checkpointing

### Poor Detection
- Increase center_radius in label generation
- Increase center_weight in loss
- Reduce augmentation strength
- Train for more epochs

### Overfitting
- Increase augmentation
- Add dropout
- Reduce model capacity
- Use more training samples

## Next Steps

After training the detector:
1. **Integrate with tracking**: Use U-Net detections with Hungarian tracker
2. **Optical flow linking**: Implement flow-based linking (Phase 5A)
3. **Division detection**: Add mitosis classifier (Phase 6)
4. **Ensemble**: Combine with classical detectors
