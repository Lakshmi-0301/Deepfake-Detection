# Deepfake Detection Project

## Overview

This project implements a deepfake detection system using a custom-built Convolutional Neural Network (CNN) trained on GPU with CuPy acceleration. The model is designed to classify images as either "real" or "fake" (deepfake) with an accuracy of approximately 69.5%.

## Features

- **Custom CNN Architecture**: Built from scratch with convolutional layers, batch normalization, ReLU activation, max pooling, and dense layers
- **GPU Acceleration**: Utilizes CuPy for GPU-accelerated computations
- **Data Preprocessing**: Automated image loading, resizing, normalization, and batch preparation
- **Training Pipeline**: Complete training loop with progress tracking, learning rate scheduling, and model checkpointing
- **Validation & Testing**: Comprehensive evaluation with accuracy metrics, confusion matrices, and ROC curves
- **Model Persistence**: Save and load model weights for inference
- **Visualization**: Sample prediction displays and performance metrics visualization

## Model Architecture

The CNN consists of:
- 3 Convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool)
- Flattening layer
- Fully connected layer
- Softmax output for binary classification

## Dataset Structure

The project expects the following directory structure:
```
Dataset/
├── Train/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

## Usage

### Training
```python
# Configure paths
fake_dir = "/path/to/train/fake"
real_dir = "/path/to/train/real"

# Start training
model = train_optimized(
    fake_dir=fake_dir,
    real_dir=real_dir,
    epochs=30,
    batch_size=128,
    lr=0.003,
    batches_per_epoch=300
)
```

### Validation
```python
validate_model(fake_dir, real_dir, "models/best_model.pkl")
```

### Testing
```python
# Run quick test on saved model
metrics = run_quick_test(
    model_path="models/best_model.pkl",
    fake_dir="/path/to/test/fake",
    real_dir="/path/to/test/real",
    num_test_images=200
)
```

## Performance

- **Training Accuracy**: ~69.5% (best model)
- **Validation Accuracy**: ~69.4%
- **Test Accuracy**: ~59.5% (on 200 sample images)

## Key Components

### Data Loading
- `load_images_shuffled_prefetch()`: Efficient batch loading with prefetching
- Automatic class balancing through undersampling
- Image resizing to 128×128 pixels

### Neural Network Layers
- `Conv2D_GPU`: Custom convolutional layer with im2col implementation
- `BatchNorm_GPU`: Batch normalization for training stability
- `MaxPool2D_GPU`: Max pooling implementation
- `Dense_GPU`: Fully connected layers

### Training Features
- Cross-entropy loss with gradient clipping
- Learning rate scheduling (exponential decay)
- Automatic model checkpointing
- Memory management with periodic cleanup

## Requirements

- Python 3.7+
- CuPy (for GPU acceleration)
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- tqdm
- Gradio (for potential web interface)

## File Structure

- `Deepfake_Detection.ipynb`: Main notebook containing all code
- `models/`: Directory for saved model weights
- Generated plots: `mini_test_results.png`, `sample_predictions_mini.png`

## Notes

- The model achieves moderate accuracy but could benefit from:
  - More training data
  - Data augmentation
  - Architecture optimization
  - Hyperparameter tuning
- The current implementation focuses on educational value with custom-built components rather than using high-level deep learning frameworks

## Future Improvements

- Integration with more sophisticated architectures (EfficientNet, Vision Transformers)
- Real-time detection capabilities
- Video frame analysis for temporal consistency
- Web interface for easy model deployment
- Transfer learning from pre-trained models
