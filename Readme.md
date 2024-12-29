## Overview
This project implements a deep learning pipeline for analyzing skin cancer images from the HAM10000 dataset. It uses a combination of MobileNetV2-based feature extraction and autoencoder-based dimensionality reduction to process and visualize skin lesion images.

## Features
- Feature extraction using MobileNetV2 architecture
- Autoencoder-based dimensionality reduction
- Multiple visualization techniques (PCA and t-SNE)
- Data augmentation for improved model robustness
- Comprehensive feature visualization pipeline

## Prerequisites
```
- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn
- OpenCV (cv2)
- Matplotlib
```

## Dataset
The project uses the HAM10000 dataset, which should be organized as follows:
```
skin-cancer-mnist-ham10000/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
└── HAM10000_metadata.csv
```

## Project Structure
The pipeline consists of several key components:

1. **Data Loading and Preprocessing**
   - Image loading from multiple directories
   - Label encoding
   - Data splitting into training and validation sets
   - Image augmentation using ImageDataGenerator

2. **Feature Extraction**
   - MobileNetV2-based U-Net architecture
   - Extraction of features from multiple network layers

3. **Autoencoder Architecture**
   - Dense layers with dropout for regularization
   - Encoding dimension of 128
   - Symmetric decoder structure

4. **Visualization Components**
   - PCA-based dimensionality reduction
   - t-SNE visualization
   - Side-by-side comparison capabilities

## Usage

1. **Data Preparation**
```python
# Load and preprocess data
metadata = pd.read_csv(metadata_path)
# Image paths and labels are automatically processed
```

2. **Feature Extraction**
```python
# Build and use feature extractor
feature_extractor = build_mobilenet_unet(input_shape=(*IMAGE_SIZE, 3))
train_features, train_labels = extract_features(train_generator, feature_extractor, train_steps)
```

3. **Autoencoder Training**
```python
# Build and train autoencoder
autoencoder, encoder_model = build_autoencoder(input_dim=input_dim, encoding_dim=128)
history = autoencoder.fit(train_features_scaled, train_features_scaled, epochs=50)
```

4. **Visualization**
```python
# Visualize features
visualize_features(train_features_scaled, train_labels, method='tsne')
visualize_side_by_side(train_features_scaled, train_labels, 
                      encoded_train_features, train_labels,
                      method='pca')
```

## Model Architecture Details

### MobileNetV2 Feature Extractor
- Uses pretrained MobileNetV2 weights
- Extracts features from 5 different layers:
  - block_1_expand_relu (112x112)
  - block_3_expand_relu (56x56)
  - block_6_expand_relu (28x28)
  - block_13_expand_relu (14x14)
  - block_16_project (7x7)

### Autoencoder
- Input layer → Dense(512) → Dense(256) → Dense(128) → Dense(256) → Dense(512) → Output
- Dropout layers (0.2) for regularization
- ReLU activation for hidden layers
- Sigmoid activation for output layer

## Visualization Options
The project provides multiple visualization techniques:
- Basic feature visualization using t-SNE or PCA
- Side-by-side comparison of features before and after autoencoder
- Original vs. reduced dimension comparisons
- Customizable plot parameters and configurations

## Performance Monitoring
- Training history visualization
- Feature space visualization before and after dimensionality reduction
- Comparison of different reduction techniques (PCA vs. t-SNE)

## Notes
- Image size is set to 32x32 pixels for processing
- Batch size is set to 4 for memory efficiency
- The autoencoder uses an encoding dimension of 128
- All visualizations include class labels and appropriate legends

## Future Improvements
- Implementation of additional visualization techniques
- Support for larger image sizes
- Integration of additional feature extraction architectures
- Enhanced data augmentation techniques
- Support for custom dataset structures
