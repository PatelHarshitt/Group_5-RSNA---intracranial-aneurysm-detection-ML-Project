# Group_5-RSNA---intracranial-aneurysm-detection-ML-Project

This repository contains the code for a lightweight, 2D multimodal approach to detecting intracranial aneurysms for the RSNA 2025 competition. Instead of relying on computationally expensive 3D segmentation models, this project uses a novel 3D-to-2D feature projection technique to train an efficient 2D classifier.

## 🧠 Motivation

Intracranial aneurysms affect ~3% of the global population and their rupture causes approximately 500,000 deaths annually. Up to half are only diagnosed after rupture, leading to severe morbidity and mortality. The manual review of 3D scans is time-consuming and subject to human error due to the small size and complex location of aneurysms. An automated, accurate detection system can enable early intervention, fundamentally transforming patient prognosis and saving lives.

## ✨ Key Features

This project's novelty comes from its lightweight and efficient design, which contrasts with traditional 3D segmentation approaches.

1.  **Novel 3D-to-2D Feature Projection:** We compress the 3D scan data into a single 3-channel 2D image. This allows a powerful, pre-trained 2D model to "see" 3D spatial and intensity relationships without the massive overhead of a 3D CNN. The channels are:
    * **Channel 1:** Middle Slice (Spatial Snapshot)
    * **Channel 2:** Max Intensity Projection (MIP) (Highlights bright structures)
    * **Channel 3:** Standard Deviation Projection (Highlights areas of variance)

2.  **Lightweight Multimodal Architecture:** The `ImprovedMultiFrameModel` fuses the deep-learned image features from the `MobileNetV3` backbone with tabular patient metadata (`PatientAge` and `PatientSex`) to make a more context-aware decision.

3.  **Custom Dual-Objective Loss Function:** We use an `ImprovedLoss` which is a weighted combination of **Weighted BCE Loss** (to heavily weight the "Aneurysm Present" class) and **Focal Loss** (to focus on hard-to-classify examples).

4.  **Robust Dual-Path Data Pipeline:** The `EightFrameDataset` is built to run fast by reading pre-processed PNGs, but includes a robust fallback to load and process raw DICOMs on the fly if PNGs are not found.

## 🚀 Environment & Setup

This project is designed to run in a standard Kaggle Notebook environment.

* **Platform:** Kaggle Notebooks
* **Accelerator:** GPU (P100, T4, or similar)
* **Internet:** Must be **ON** (to allow `timm` to download the pre-trained `MobileNetV3` model).

## 🏃 How to Run

1.  Create a new Kaggle Notebook.
2.  Set the Accelerator to **GPU** and turn **Internet ON** in the notebook's settings.
3.  Add the following two datasets:
    * `rsna-intracranial-aneurysm-detection` (The official competition data)
    * `another1` (The pre-processed PNG dataset and mapping CSVs)
4.  Copy the entire Python script into a single cell in the notebook.
5.  Run the cell to start training.

## ⚙️ Configuration

All training parameters are controlled by the `Config` class at the top of the script. You can easily toggle between a fast debug run and a full training run.

### Fast Debug Run (10-15 Minutes)

To test the entire pipeline on 30 samples for 10 epochs, use these settings:

```python
class Config:
    # ... (other settings) ...
    
    # --- MODEL ---
    MODEL_NAME_BACKBONE = "mobilenetv3_large_100" 
    
    # --- SETTINGS FOR FAST DEBUG RUN ---
    NUM_EPOCHS = 10         
    NUM_FOLDS = 1           # Runs a single 80/20 train/val split
    USE_GROUP_CV = False    # Uses fast StratifiedKFold (skips slow DICOM reading)
    DEBUG_SAMPLE_SIZE = 30  # Uses only 30 scans
