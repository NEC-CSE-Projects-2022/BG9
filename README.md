
# Team Number – Project Title

## Team Info
- 22471A05C2 — **POLISETTY RISHITHA SAI SRI** ( [LinkedIn](https://www.linkedin.com/in/rishitha-polisetty-914293271/) )
_Work Done: Dataset preprocessing, HSL image fusion, model training and evaluation, results analysis

- 22471A05B0 — **NAGASARAPU MOHANA SRI KRUPA** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Literature survey, data augmentation strategies, performance comparison with baseline CNN models

- 22471A05C5 — **RAYALA LAKSHMI** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: Model architecture study, result visualization (confusion matrix, ROC), documentation and report preparation



---

## Abstract
Microplastic pollution has emerged as a serious environmental threat due to its harmful impact on ecosystems and human health. Manual identification of microplastics is inefficient and time-consuming. This project proposes an automated microplastic classification framework using Swin Transformer V2 on holographic images. Amplitude and phase images from the HMPD dataset are fused using the HSL color space, enriching both spatial and color information. The proposed approach achieves an accuracy of 91.65% and an F1-score of 91.79%, outperforming traditional CNN-based models. The results demonstrate the effectiveness of vision transformers for precise microplastic detection in environmental monitoring applications. 



---

## Paper Reference (Inspiration)
👉 **[Paper Title Microplastic Classification in Holographic Images Using Swin Transformer V2
  – Author Names Teresa Cacace, Marco Del-Coco, ierluigi Carcagnì,Mariacristina Cocca, Melania aturzo, and Cosimo Distante ]([Paper URL here](https://link.springer.com/chapter/10.1007/978-3-031-43153-1_11))**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
Unlike earlier CNN-based and ViT models, this project employs Swin Transformer V2, which uses hierarchical shifted-window self-attention. This significantly improves feature learning for minute holographic textures. Compared to previous approaches, our method achieves a better balance between precision and recall, reducing false positives while maintaining high detection accuracy. The model also scales efficiently for high-resolution scientific images, making it suitable for real-time environmental monitoring.

---

## About the Project
What it does: Automatically classifies holographic images as microplastic or non-microplastic
Why it is useful: Enables fast, accurate, and scalable microplastic detection, reducing human effort

Workflow:
Input: Amplitude & Phase holographic images
→ Processing: Grayscale conversion, HSL fusion, data augmentation
→ Model: Swin Transformer V2
→ Output: Binary classification (Microplastic / Non-Microplastic)

---

## Dataset Used
👉 **[Holographic Microplastic Dataset (HMPD)]([Dataset URL](https://github.com/beppe2hd/HMPD/tree/main ))**

**Dataset Details:**
Total images: 13,172
Pair of images: 6586
Classes: Microplastic (3,293) and Non-Microplastic (3,293)
Image type: Paired amplitude and phase holographic images
Dataset is balanced, ensuring unbiased model training

---

## Dependencies Used
Python, PyTorch, OpenCV, NumPy, Matplotlib, Torchvision, Scikit-learn

---

## EDA & Preprocessing
Conversion of amplitude and phase images to grayscale
HSL fusion (Amplitude → Hue, Phase → Saturation, Lightness = 0.5)
RGB conversion and resizing to 256×256
Data augmentation: horizontal flip, vertical flip, rotation (±15°), color jittering
Train–validation split: 80% – 20%

---

## Model Training Info
Model: Swin Transformer V2
Loss function: Cross-entropy loss
Optimizer: Adam
Training epochs: 40
Attention mechanism: Shifted window self-attention for hierarchical feature learning

---

## Model Testing / Evaluation
Metrics used: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC
Validation accuracy consistently ranged between 85% and 93%
ROC-AUC achieved: 0.979, indicating strong class separability

---

## Results
Accuracy: 91.65%
Precision: 94.6%
Recall: 89.3%
F1-Score: 91.7%

The Swin Transformer V2 with HSL-fused images significantly outperformed AlexNet, ResNet18, and VGG11 models.

---

## Limitations & Future Work
Dataset consists mainly of laboratory-captured images
Only binary classification (microplastic vs non-microplastic)
Future work includes multi-class classification, real-world water samples, and real-time deployment on embedded systems

---

## Deployment Info
The model can be integrated into compact environmental monitoring systems and holographic imaging devices for real-time microplastic detection in water bodies. With optimization, it can be deployed on edge devices for continuous monitoring.

---
