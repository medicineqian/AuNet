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
  
Download checkpoint from:
* baidu drive: https://pan.baidu.com/s/1GPldjHnnGaHG9zAx8qwJvg 提取码: best
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

## 🔧 Simulating Missing Modalities

To evaluate the model under **missing modality conditions**, you can manually set a modality to zero in `test.py`.

Specifically, modify **lines 161–164** in `test.py` by replacing the corresponding modality input with `torch.zeros_like()`.

Example:

```python
# simulate missing RGB
rgb = torch.zeros_like(rgb)

# simulate missing NIR
nir = torch.zeros_like(nir)

# simulate missing TIR
tir = torch.zeros_like(tir)
```

This will simulate the **absence of the selected modality** during inference.

You can test different modality combinations such as:

| Input Combination | Setting                           |
| ----------------- | --------------------------------- |
| RGB + NIR         | set `tir = torch.zeros_like(tir)` |
| RGB + TIR         | set `nir = torch.zeros_like(nir)` |
| NIR + TIR         | set `rgb = torch.zeros_like(rgb)` |
| RGB only          | set `nir` and `tir` to zero       |
| NIR only          | set `rgb` and `tir` to zero       |
| TIR only          | set `rgb` and `nir` to zero       |


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

