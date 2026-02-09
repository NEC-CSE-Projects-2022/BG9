# Team Number – Project Title

## Team Info
. 22471A05C2 — POLISETTY RISHITHA SAI SRI
  . LinkedIn: https://www.linkedin.com/in/rishitha-polisetty-914293271/
  . Work Done:
    . Dataset preprocessing
    . HSL image fusion
    . Model training and evaluation
    . Results analysis

. 22471A05B0 — NAGASARAPU MOHANA SRI KRUPA
  . LinkedIn: https://www.linkedin.com/in/mohana1414
  . Work Done:
    . Literature survey
    . Data augmentation strategies
    . Performance comparison with baseline CNN models

. 22471A05C5 — RAYALA LAKSHMI
  . LinkedIn: https://linkedin.com/in/lakshmi
  . Work Done:
    . Model architecture study
    . Result visualization (Confusion Matrix, ROC)
    . Documentation and report preparation


## Abstract
. Microplastic pollution poses a serious environmental and health threat
. Manual identification is time-consuming and inefficient
. This project proposes an automated classification system using Swin Transformer V2
. Holographic amplitude and phase images are fused using HSL color space
. The model achieves:
  . Accuracy: 91.65%
  . F1-score: 91.79%
. The approach outperforms traditional CNN-based models
. Demonstrates suitability for environmental monitoring applications


## Paper Reference (Inspiration)
. Title: Microplastic Classification in Holographic Images Using Swin Transformer V2
. Authors:
  . Teresa Cacace
  . Marco Del-Coco
  . Pierluigi Carcagnì
  . Mariacristina Cocca
  . Melania Maturzo
  . Cosimo Distante
. Source:
  . https://link.springer.com/chapter/10.1007/978-3-031-43153-1_11


## Our Improvement Over Existing Paper
. Uses Swin Transformer V2 instead of CNNs and basic ViT models
. Employs hierarchical shifted-window self-attention
. Improves learning of fine holographic textures
. Achieves better precision–recall balance
. Reduces false positives
. Scales efficiently for high-resolution scientific images
. Suitable for real-time monitoring systems


## About the Project
. What it does:
  . Automatically classifies holographic images as microplastic or non-microplastic
. Why it is useful:
  . Faster and more accurate detection
  . Reduces human effort
  . Enables scalable monitoring

. Workflow:
  . Input:
    . Amplitude holographic images
    . Phase holographic images
  . Processing:
    . Grayscale conversion
    . HSL image fusion
    . Data augmentation
  . Model:
    . Swin Transformer V2
  . Output:
    . Binary classification (Microplastic / Non-Microplastic)


## Dataset Used
. Dataset Name:
  . Holographic Microplastic Dataset (HMPD)
. Dataset Link:
  . https://github.com/beppe2hd/HMPD/tree/main

. Dataset Details:
  . Total images: 13,172
  . Image pairs: 6,586
  . Classes:
    . Microplastic: 3,293
    . Non-Microplastic: 3,293
  . Image type:
    . Paired amplitude and phase holographic images
  . Dataset is balanced to avoid bias


## Dependencies Used
. Python
. PyTorch
. OpenCV
. NumPy
. Matplotlib
. Torchvision
. Scikit-learn


## EDA & Preprocessing
. Convert amplitude and phase images to grayscale
. Perform HSL fusion:
  . Hue → Amplitude
  . Saturation → Phase
  . Lightness → 0.5
. Convert fused HSL image to RGB
. Resize images to 256 × 256
. Data augmentation techniques:
  . Horizontal flip
  . Vertical flip
  . Rotation (±15 degrees)
  . Color jittering
. Train–validation split:
  . Training: 80%
  . Validation: 20%


## Model Training Info
. Model:
  . Swin Transformer V2
. Loss function:
  . Cross-entropy loss
. Optimizer:
  . Adam
. Training epochs:
  . 40
. Attention mechanism:
  . Shifted window self-attention
  . Hierarchical feature learning


## Model Testing / Evaluation
. Evaluation metrics:
  . Accuracy
  . Precision
  . Recall
  . F1-score
  . Confusion Matrix
  . ROC-AUC
. Validation accuracy range:
  . 85% – 93%
. ROC-AUC score:
  . 0.979
. Indicates strong class separability


## Results
. Accuracy: 91.65%
. Precision: 94.6%
. Recall: 89.3%
. F1-score: 91.7%
. Performance comparison:
  . Outperformed AlexNet
  . Outperformed ResNet18
  . Outperformed VGG11


## Limitations & Future Work
. Dataset mainly consists of laboratory-captured images
. Only binary classification implemented
. Future enhancements:
  . Multi-class microplastic classification
  . Real-world water sample testing
  . Real-time deployment
  . Embedded and edge-device optimization


## Deployment Info
. Model can be integrated into:
  . Environmental monitoring systems
  . Holographic imaging devices
. Enables:
  . Real-time microplastic detection
  . Continuous water quality monitoring
. With optimization:
  . Suitable for edge and embedded systems
