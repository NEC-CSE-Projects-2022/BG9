# Project Documents

This folder contains all official documents related to the **Microplastic Classification in Holographic Images Using Swin Transformer V2** project, including the abstract, paper, presentations, and detailed project documentation submitted for academic evaluation.

## Repository Contents

- `CONFERENCE_PAPER.pdf`
  Final camera-ready research paper describing the methodology, models, experiments, and results.

- `ABSTRACT_BG9.pdf`
  Concise abstract outlining the problem statement, approach, and key findings.

- `BG9_PPT.pptx`
  Editable presentation used for internal college/project reviews.

- `CameraReadyPaper.ppt` 
  Presentation prepared for conference or external evaluation.

- `BG-9 FINAL DOCUMENTATION REVIEW.docx`
  Complete project documentation including system design, dataset description, preprocessing, model architecture, experimental setup, and results.


## System Description

**Input:** Paired amplitude and phase images captured using digital holography



**Processing**

- Image resizing and normalization to standard dimensions
- HSL color space fusion, where amplitude is mapped to Hue and phase to Saturation
- Data augmentation techniques such as rotation, flipping, and color jittering to improve generalization
- Conversion of fused HSL images into RGB format for model compatibility

**Models Used**

- Swin Transformer V2 for hierarchical feature extraction
- Shifted window self-attention to capture both local textures and global contextual information

**Output**

- Classification of samples as Microplastic or Non-Microplastic
- Performance evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC

## Tools & Technologies

- **Programming Language:** Python 3.x
- **Frameworks:** PyTorch, Torchvision
- **Libraries:** NumPy, OpenCV, Matplotlib, scikit-learn
- **Development Environment:**
  - Local system (Windows 11)
  - Google Colab (GPU-enabled training)

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix


Detailed quantitative results are provided in the camera-ready paper and project documentation.

## Reproducibility

For full reproducibility, refer to `BG-9 FINAL DOCUMENTATION REVIEW`, which includes:

- Dataset preprocessing steps
- Data split strategy
- Model architectures and hyperparameters
- Training configuration and evaluation protocol

## Notes

- All documents are intended for academic and research purposes only.
- The dataset and pretrained models used in this project are subject to their respective licenses.
- Results may vary depending on hardware and training configuration.
