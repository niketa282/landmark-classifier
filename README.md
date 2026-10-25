# Landmark Image Classifier

[![Launch demo on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/niketa282/landmark-classifier/HEAD?urlpath=voila%2Frender%2Fapp.ipynb)

A PyTorch image classifier that looks at a photo and predicts which of **50 famous landmarks** it shows, from Machu Picchu to the Eiffel Tower. It can tag photos that have no location data attached, which is useful for photo-sharing and photo-storage services.

I built and compared two models:

| Model | Approach | Test accuracy |
|---|---|---|
| CNN from scratch | Custom 5-block convolutional network | **61%** |
| Transfer learning | Pretrained ResNet18 with a new classification layer | **75%** |

The better model (transfer learning) is deployed as an interactive web app. Upload a photo and it returns the top 5 most likely landmarks with their probabilities.

**[Try the live demo](https://mybinder.org/v2/gh/niketa282/landmark-classifier/HEAD?urlpath=voila%2Frender%2Fapp.ipynb)**. It runs on Binder, a free service, so it can take a minute or two to start.

---

## Dataset

The dataset is a subset of the [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark): about 6,000 images across 50 landmark classes. The test set has 1,250 images (25 per landmark), and 20% of the training data was held out for validation.

## Model 1: CNN from scratch

**Architecture**
- 5 convolutional blocks (16 → 32 → 64 → 128 → 256 filters), each with Conv2d → **BatchNorm** → ReLU → MaxPool
- **Global average pooling**, which compresses each feature map to a single value
- Classifier: Linear(256 → 256) → BatchNorm → ReLU → Dropout(0.4) → Linear(256 → 50)

**Training**: the Adam optimiser (learning rate 0.001, weight decay 1e-4) for 50 epochs, with a learning-rate scheduler that lowers the rate when validation loss stops improving, and checkpointing of the best model on validation loss.

**What I learned while improving it**

| Change | Result |
|---|---|
| Starting point: SGD (learning rate 0.001), no BatchNorm | Loss stayed flat, no better than random guessing (2%) |
| Switched to Adam and added BatchNorm | Training started working: **46%** |
| Stronger data augmentation (RandomResizedCrop, RandAugment) and weight decay | Still overfitting: **45%** |
| Replaced a 6M-parameter flatten layer with **global average pooling** | Overfitting greatly reduced: **61%** |

The biggest gain came from finding the real cause of overfitting. The fully connected layer after flattening held more parameters than the rest of the network combined, which gave the model room to memorise the training images. Global average pooling removed that capacity.

## Model 2: transfer learning

- **Backbone:** ResNet18, pretrained on ImageNet, with every layer **frozen**
- **Head:** a new fully connected layer mapping ResNet18's 512 features to the 50 landmarks
- **Training:** only the new layer is trained, using Adam (learning rate 0.001)

Because the pretrained network already recognises general visual features (edges, textures, shapes and objects), reusing it raised accuracy from 61% to **75%** while training a much smaller part of the model.

## Deployment

The final model is exported with **TorchScript** (`torch.jit.script`) into a single file, `checkpoints/transfer_exported.pt`. That file bundles:
- the image preprocessing (resize, centre crop, normalise)
- the network and its trained weights
- a softmax layer, so the output is a set of probabilities

The app (`app.ipynb`) loads this file and doesn't need any of the training code. It is served as a web page with [Voila](https://voila.readthedocs.io/) on [Binder](https://mybinder.org/).

## Repository contents

```
├── app.ipynb                    # Web app: upload a photo, get the top 5 predictions
├── cnn_from_scratch.ipynb       # Model 1: design, training and evaluation
├── transfer_learning.ipynb      # Model 2: design, training and evaluation
├── src/
│   ├── model.py                 # CNN architecture
│   ├── transfer.py              # ResNet18 transfer learning model
│   ├── data.py                  # Data loading and augmentation
│   ├── train.py                 # Training, validation and test loops
│   ├── optimization.py          # Loss function and optimisers
│   └── predictor.py             # Wrapper used for TorchScript export
├── checkpoints/
│   └── transfer_exported.pt     # Exported model used by the app
├── requirements.txt             # Packages installed by Binder
└── runtime.txt                  # Python version for Binder
```

## Tech stack

Python · PyTorch · torchvision · TorchScript · NumPy · Jupyter · ipywidgets · Voila · Binder

Models were trained on an NVIDIA A100 GPU through the Slurm job scheduler on a university high-performance computing cluster.

## Acknowledgements

This project was completed as part of the Udacity Convolutional Neural Networks course. The starter code structure was provided by Udacity.
