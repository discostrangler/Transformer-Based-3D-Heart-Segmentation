# Minimizing Quadratic Complexity: A Transformer Employing Linear Self-Attention for 3D Cardiac Segmentation

## 🧠 Overview

This repository presents a **Vision Transformer (ViT)** framework for efficient and accurate **3D cardiac MRI segmentation**, focusing on the left ventricle (LV), right ventricle (RV), and myocardium (MYO).
The proposed architecture introduces a **linear self-attention** mechanism and a **Bidirectional Multi-Head Attention (B-MHA)** module to drastically reduce computational complexity while maintaining segmentation precision.

By combining **multi-level feature fusion** with linear attention, the model achieves **state-of-the-art segmentation performance** and efficiency on the **ACDC dataset**, outperforming existing 3D medical segmentation methods.

---

## 📚 Abstract

3D segmentation of the heart enables detailed structural and functional analysis essential for diagnosing cardiovascular conditions. However, standard Vision Transformers (ViTs) face challenges in handling 3D medical scans due to high computational demands and limited annotated data.

This work introduces:

* A **linear self-attention mechanism** that reduces complexity from quadratic to linear.
* A **multi-level feature extraction and fusion strategy** that captures both global spatial context and fine-grained details.

The proposed model achieves high segmentation accuracy on the **Automated Cardiac Diagnosis Challenge (ACDC)** dataset, outperforming benchmarks such as **nnU-Net**, **UNETR**, and **SwinUNETR**.

---

## ⚙️ Methodology

### 1. Linear Self-Attention

Standard self-attention scales quadratically with the number of tokens.
We reformulate it using a **linearized variant** that significantly reduces memory and runtime while preserving global dependency modeling.

### 2. Optimized Multi-Head Attention (MHA)

We employ an **optimized MHA** that introduces low-rank projections to reduce redundancy and achieve linear complexity, lowering the overhead from 𝒪(n²d) to 𝒪(nld), where *l ≪ n*.

### 3. Bidirectional Multi-Head Attention (B-MHA)

A **Bidirectional Multi-Head Attention module** performs **non-linear dimensionality reduction**, retaining meaningful semantics while minimizing redundancy.
Shared query-key projections enable symmetrical attention between token maps, cutting computation by nearly half.

### 4. Multi-Level Feature Fusion

Multi-scale semantic maps from different encoder levels are concatenated and processed through a transformer block.
This enables **cross-scale interaction**, where high-level features refine low-level boundaries and vice versa.

### 5. Architecture Summary

* **Encoder:** Convolutional residual stem + hierarchical B-MHA transformers
* **Fusion Module:** Multi-scale semantic fusion via all-to-all attention
* **Decoder:** Upsampling layers with skip connections and deep supervision
* **Loss:** Combined Dice and Cross-Entropy with auxiliary supervision

---

## 🯩 Dataset

The model is trained and evaluated on the **Automated Cardiac Diagnosis Challenge (ACDC)** dataset:

* 100 patient cardiac MRI volumes
* Expert annotations for LV, RV, and MYO
* Split: 70% training, 10% validation, 20% testing

---

## 📈 Results

| Structure            | Dice (↑)  | Hausdorff Distance (↓) |
| -------------------- | --------- | ---------------------- |
| Left Ventricle (LV)  | **0.940** | **5.6 mm**             |
| Right Ventricle (RV) | **0.894** | **8.9 mm**             |
| Myocardium (MYO)     | **0.892** | **7.1 mm**             |

### 🏆 Highlights

* +1.2–2.5% Dice improvement over nnU-Net and UNETR
* 45% lower GPU memory usage
* 1.8× faster inference than standard transformers

---

## 🧪 Key Contributions

* Introduced **Bidirectional Multi-Head Linear Attention (B-MHA)** for volumetric data
* Proposed **multi-level semantic fusion** for cross-scale feature refinement
* Achieved **state-of-the-art segmentation accuracy** on 3D cardiac data
* Reduced **computational and memory cost** by over 50%

---

## 🏗️ Implementation

### Requirements

```bash
Python >= 3.8  
PyTorch >= 2.0  
torchvision  
SimpleITK  
numpy  
matplotlib  
scikit-image  
tqdm  
```
---

## 📊 Evaluation Metrics

* **Dice Similarity Coefficient (DSC)** – measures overlap accuracy
* **Hausdorff Distance (HD)** – measures boundary precision

---

## 🧠 Architecture Summary

```
Input Volume  
   ↓  
Convolutional Stem → Encoder (B-MHA Blocks)  
   ↓  
Multi-Scale Semantic Fusion Transformer  
   ↓  
Decoder with Skip Connections + Deep Supervision  
   ↓  
Segmentation Output (LV, RV, MYO)
```
