# Robust Pedestrian Detection with Uncertain Modality

## 📌 Overview

This repository provides the **multimodal dataset and codebase** for the paper:

> **Robust Pedestrian Detection with Uncertain Modality**

We focus on pedestrian detection under **modality uncertainty**, where one or more modalities (RGB / NIR / TIR) may be degraded, missing, or unreliable.

---

## 🖼️ Dataset Overview

### ✨ Modalities

* **RGB** (visible spectrum)
* **NIR** (near-infrared)
* **TIR** (thermal infrared)
* **Annotations** (TXT format)

All modalities are:

* Pixel-aligned
* Filename-aligned

---


## 📂 Dataset Structure

```
dataset/
├── RGB/
│   ├── train/
│   └── test/
├── NIR/
│   ├── train/
│   └── test/
├── TIR/
│   ├── train/
│   └── test/
├── label/
│   ├── train/
│   └── test/
```

---

## 📊 Dataset Statistics
<img width="1272" height="398" alt="image" src="https://github.com/user-attachments/assets/499e52b5-e90a-4815-9ea6-9c9daf821afd" />


---

## ⚙️ Method Overview
<img width="1949" height="731" alt="image" src="https://github.com/user-attachments/assets/a58dee3e-4921-434e-b906-a9c073e2320c" />

---

## 🚀 Usage

### 1. Environment

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Download dataset from:

* baidu drive: https://pan.baidu.com/s/1IleD45JYPDKLYdrtCYLHuw?pwd=drnt 提取码: drnt
---

### 3. Training

```bash
python train.py
```

---

### 4. Evaluation

```bash
python test.py
```

---

## 📜 License

This dataset is released under:

* CC BY 4.0 

---

## 🙋 Citation

```bibtex
@article{bie2026robust,
  title={Robust Pedestrian Detection with Uncertain Modality},
  author={Bie, Qian and Wang, Xiao and Yang, Bin and Yu, Zhixi and Chen, Jun and Xu, Xin},
  journal={arXiv preprint arXiv:2602.06363},
  year={2026}
}
```

---

