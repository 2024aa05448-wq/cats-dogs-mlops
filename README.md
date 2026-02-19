# Cats vs Dogs MLOps Pipeline (Assignment 2)

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs).

## Quickstart

1. `pip install -r requirements.txt`
2. Run notebook: `jupyter notebook notebooks/cats_vs_dogs_pipeline.ipynb`
3. Data pipeline verified ✅

## Status
- ✅ Data acquisition (Kaggle download)
- ✅ Train/val/test split (800/100/100 per class)
- ✅ PyTorch dataloaders with augmentation
- ⏳ Model training (CNN + MLflow)
- ⏳ API service (FastAPI)
- ⏳ Docker + CI/CD
