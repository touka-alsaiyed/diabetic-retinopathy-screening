# Diabetic Retinopathy Screening

An AI-based screening system for detecting diabetic retinopathy (DR) from retinal fundus
photographs, developed as part of a larger capstone project screening for three
vision-threatening eye conditions (DR, Keratoconus, and Glaucoma).

## Problem

Diabetic retinopathy often goes undiagnosed in its early stages due to limited access to
ophthalmologists and specialized equipment. Early detection is critical to prevent irreversible
vision loss, but manual screening of retinal images is time-consuming and subjective. Without
scalable, automated screening tools, many patients are diagnosed too late — once damage is
already severe or irreversible.

**Goal:** build an automated, AI-based screening pipeline that grades DR severity (0 – No DR
through 4 – Proliferative DR) from a retinal photo, fast and consistently enough to support
doctors in early diagnosis.

## Approach

- **Datasets:** APTOS 2015 and APTOS 2019 (Kaggle), combined via two-stage transfer learning.
- **Preprocessing:** crop-from-gray (removes the black border around the fundus), resize, and a
  Gaussian-blur weighted-blend sharpening filter to bring out vessels, hemorrhages, and
  microaneurysms. Offline data augmentation addresses class imbalance in the minority grades.
- **Model:** EfficientNetB6, pretrained on 2015, then fine-tuned on 2019.
- **Explainability:** Grad-CAM overlays highlight the regions of the retina driving each
  prediction.

### Layer freezing schedule

- **Stage 1 — pretrain on 2015:** the entire EfficientNetB6 backbone is frozen
  (`base_model.trainable = False`); only the new head — Global Average Pooling → Dropout →
  Dense(512, ReLU) → Dropout → Dense(5, softmax) — is trained. This uses the backbone purely as
  a fixed ImageNet feature extractor while the head learns the DR task from scratch.
- **Stage 2 — fine-tune on 2019:** the backbone is unfrozen (`model.layers[0].trainable = True`),
  **except every `BatchNormalization` layer inside it, which is explicitly re-frozen**. This is
  because fine-tuning at a small batch size lets BatchNorm's running statistics drift and
  destabilize an otherwise well-pretrained network — keeping BN frozen while everything else
  fine-tunes is a common fix for that.

## Screenshots

**Preprocessing — before vs. after**

| Before | After |
|---|---|
| ![Raw fundus photo before preprocessing](assets/before_preprocessing.png) | ![Fundus photo after crop, resize, and sharpening](assets/after_preprocessing.jpeg) |

**Grad-CAM explainability**

![Grad-CAM on a Mild DR example: original fundus image, raw Grad-CAM heatmap, and the heatmap overlaid on the original — predicted Mild with 93.5% confidence](assets/gradcam_example.png)

### How Grad-CAM is implemented

The notebook implements Grad-CAM (Selvaraju et al., 2017) directly with `tf.GradientTape` —
no external Grad-CAM library:

1. **Target layer:** the last convolutional feature map in the EfficientNetB6 backbone,
   `top_conv` (found via a helper that falls back to scanning backwards for the last `Conv2D`
   layer if EfficientNet's naming ever changes).
2. **Graph construction:** a helper model is built with two outputs from the *same* forward
   pass — the `top_conv` feature map, and the final class probabilities — by re-applying the
   head layers (GAP → Dropout → Dense(512) → Dropout → Dense(softmax)) on top of the backbone's
   output. Keeping both outputs on one connected graph (rather than two separate models) is what
   makes the gradient of the prediction with respect to the conv feature map computable.
3. **Gradient + weighting:** under `GradientTape`, the gradient of the predicted class's score
   is taken with respect to the `top_conv` output, then globally average-pooled over height and
   width to get one importance weight per feature-map channel — the core Grad-CAM step.
4. **Heatmap:** each channel of the conv feature map is scaled by its importance weight and
   summed, passed through ReLU (negative contributions discarded), and normalized to `[0, 1]`.
5. **Overlay:** the heatmap is resized to the original image resolution, colorized with
   matplotlib's `jet` colormap, and alpha-blended (`alpha=0.4`) over the original RGB fundus
   image.
6. **Demo selection:** for each of the 5 classes, the notebook picks the validation image with
   the *highest-confidence correct prediction* for that class, then runs Grad-CAM on it — so the
   examples shown are the model's most confident correct calls per class, not random samples.

## Results

![Validation classification report on APTOS 2019: 0.85 accuracy, 0.85 macro F1, 0.93 quadratic weighted kappa](assets/classification_report.png)

| Metric | Score |
|---|---|
| Quadratic Weighted Kappa | 0.93 |
| Test Accuracy | 0.85 |
| Macro F1-score | 0.85 |

## Repository structure

```
DR_screening/
├── Data_preprossing/
│   ├── 01_preprocessing_2015.ipynb   # reproducible preprocessing + offline class balancing (2015)
│   └── 02_preprocessing_2019.ipynb   # same pipeline, applied to APTOS 2019
└── Training_notebook/
    └── DR_Screening_final.ipynb      # original training + Grad-CAM notebook (being refactored
                                       # to match the current preprocessing pipeline and to use a
                                       # kappa-optimized regression approach instead of softmax
                                       # classification)
```

## Team

Part of a joint capstone project (AI Engineering) covering DR, Keratoconus, and Glaucoma
screening, deployed through a shared web interface. DR and Keratoconus modeling: Touka Alsaiyed
and Sabah Aljajeh. Glaucoma modeling: Efe Omer Guler and Emircan Cankara. Deployment: Nese Nur Bas.
