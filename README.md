# Facial Emotion Recognition (EfficientNetB0 + Transfer Learning)

This repository contains the implementation of a facial emotion recognition system that classifies six basic facial expressions using deep learning and transfer learning techniques.

## Approach

The project uses a two-stage transfer learning pipeline built on EfficientNetB0:

- Face detection using Haar cascade classifiers (Viola-Jones framework)
- Preprocessing with resizing to 224x224 and ImageNet normalisation
- Data augmentation with random horizontal flipping and rotation up to 10 degrees
- **Stage 1:** Frozen EfficientNetB0 feature extractor with a custom classification head trained at lr=0.001
- **Stage 2:** Full fine-tuning of all layers at a reduced lr=0.0001 to avoid catastrophic forgetting
- Class-weighted cross-entropy loss to address class imbalance
- Best model weights saved based on test accuracy

## Datasets

- **JAFFE** — 183 greyscale images of Japanese female subjects, 6 emotion classes
- **CK (Cohn–Kanade)** — 325 images with greater subject and expression variability, 6 emotion classes
