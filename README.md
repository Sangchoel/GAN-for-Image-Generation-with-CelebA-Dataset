# CelebA Face Generation using DCGAN

This project implements a Deep Convolutional GAN (DCGAN) using PyTorch to generate face images based on the CelebA dataset.

## 🔧 Features
- Generator & Discriminator based on CNN architecture
- Image resolution: 64x64 RGB
- Stable training using Adam optimizer & BCELoss
- Image normalization and preprocessing included
- Automatic sample image saving every epoch

## 🧠 Architecture

### Generator
- Input: 100-dim latent vector (noise)
- Output: 3x64x64 RGB image

### Discriminator
- Input: 3x64x64 image
- Output: Binary classification (Real/Fake)

## 📂 Dataset
- CelebA dataset should be placed at: `your_path/celeba`
- Make sure it's organized under an `ImageFolder`-style structure.

## 🚀 Usage
1. Install dependencies
```bash
pip install torch torchvision matplotlib
