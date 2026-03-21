# Mini Project: Deep Learning–Based Classification of Imbalanced WCE Datasets

This mini-project implements the tasks defined in `mini-project/MINIPROJECT.pdf` using **PyTorch** in a **Google Colab–ready** notebook.

## Deliverables
- `mini-project/wce_imbalanced_classification_colab.ipynb` — complete working notebook (Tasks 1–7)
- `mini-project/report.tex` — LaTeX report in academic format
- `mini-project/README.md` — setup and execution guide

## Project Overview
The notebook addresses class imbalance in gastrointestinal disease classification on Wireless Capsule Endoscopy (WCE) images by combining:
1. Dataset exploration and imbalance analysis
2. Random under-sampling of majority classes
3. Data augmentation–based over-sampling for minority classes
4. Transfer learning with three pretrained CNN backbones
5. Intelligent learning-rate control (ReduceLROnPlateau)
6. Comparative evaluation across three imbalance settings

## Google Colab Execution Steps
1. Open Google Colab and upload `wce_imbalanced_classification_colab.ipynb`.
2. Enable GPU: `Runtime -> Change runtime type -> T4 GPU`.
3. Set dataset path in the configuration cell:
   - `DATASET_ROOT = Path("/content/kvasir-capsule")`
4. Ensure dataset follows class-folder format:
   - `/content/kvasir-capsule/<class_name>/<image files>`
5. Run all notebook cells in order.
6. Outputs are saved to:
   - `OUTPUT_DIR = /content/wce_outputs`

## Dataset Setup Instructions
Use Kvasir-Capsule (or equivalent WCE dataset with class-wise folder structure).

Expected structure:
```text
/content/kvasir-capsule/
  ├── class_1/
  │   ├── img001.jpg
  │   └── ...
  ├── class_2/
  │   ├── img101.jpg
  │   └── ...
  └── ...
```

## Implemented Tasks Summary
- **Task 1**: Class distribution plot + majority/minority analysis + 5–6 line explanation
- **Task 2**: Random under-sampling to fixed threshold with updated distribution
- **Task 3**: Minority-only augmentation (flip, rotation ±20°, shift 0.2, zoom 0.2), before/after samples, dataset summary
- **Task 4**: Resize to 224×224, normalize to [0,1], split 70/15/15
- **Task 5**: Transfer learning with EfficientNet-B0, MobileNetV3-Small, ResNet101; frozen vs trainable parameter summary
- **Task 6**: ReduceLROnPlateau with LR-vs-epoch and train-vs-val loss plots
- **Task 7**: Comparison across 3 settings (no handling / under-sampling / under+aug) using Accuracy, Precision, Recall, F1, confusion matrix

## Results Summary
The notebook produces:
- Class-distribution plots (original, under-sampled, under+augmented)
- Before/after augmentation visualization
- Model parameter comparison table
- Metric comparison table (`task7_comparison_table.csv`)
- Confusion matrices for best model per setting
- Learning-rate and loss curves for each model-setting pair

In typical imbalanced medical classification workflows, macro-level metrics (especially macro-recall and macro-F1) improve when under-sampling is combined with targeted augmentation.

## Notes
- Default epochs are intentionally modest for Colab iteration speed (`EPOCHS = 5`). Increase to 10–25 for final training.
- Validation and test sets are kept untouched for fair evaluation.
