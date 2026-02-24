<div align="center">

<h2>
  <a href="https://manishgovind.github.io/unilact-vla/" style="color:#9C276A; text-decoration:none;">
    UniLACT: Depth-Aware RGB Latent Action Learning for Vision-Language-Action Models
  </a>
</h2>

<p>
  <a href="https://arxiv.org/abs/2510.13808">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat&logo=arxiv" />
  </a>
  <a href="https://manishgovind.github.io/unilact-vla/">
    <img src="https://img.shields.io/badge/Website-Project%20Page-2ea44f?style=flat" />
  </a>
</p>

</div>

## ⚙️ Setup

### 1) Create and activate environment
```bash
conda create -n unilact python=3.10 -y
conda activate unilact
````

### 2) Clone the repo and install dependencies

```bash
git clone https://github.com/manishgovind/uniact-vla.git
cd UniLACT
pip install -r requirements.txt
```

### 3) (Optional) Set project root

```bash
export PROJECT_UNILACT_ROOT=/path/to/UniLACT
```

---

## 🚀 Training (3 Stages)

UniLACT training consists of **three stages**:

1. **Stage 1:** Unified latent action learning (**UniLARN**)
2. **Stage 2:** Unified latent pretraining 
3. **Stage 3:** Action fine-tuning

> Training is driven by YAML configs under `unilact/configs/train/` and configs under `unilarn/configs/train`.

---

### Stage 1 — UniLARN (Unified Latent Action Learning)

```bash
# (Update this command to your UniLARN entrypoint/config if different)
cd ${PROJECT_UNILACT_ROOT}/unilarn
accelerate launch --main_process_port <master_port> train_unilarn.py --config_path "${PROJECT_UNILACT_ROOT}/unilarn/configs/train/train_unilarn_on_calvin.yaml"
```

---

### Stage 2 — Unified latent Pretraining 



```bash
# (Update this command to your pretraining config if different)
cd ${PROJECT_UNILACT_ROOT}/unilact/train
accelerate launch --main_process_port <master_port>  train_unilact.py --config_path "${PROJECT_UNILACT_ROOT}/unilact/configs/train/pretrain_unilact_on_calvin.yaml"
```

---

### Stage 3 — Fine-tuning

```bash
cd ${PROJECT_UNILACT_ROOT}/unilact/train
accelerate launch --main_process_port <master_port> train_unilact.py   --config_path "${PROJECT_UNILACT_ROOT}/unilact/configs/train/finetune_unilact_on_calvin.yaml"
```

---

### Evaluation on CALVIN (ABC→D) Benchmark

Install the CALVIN benchmark in the same conda environment (`unilact`) by following the official [CALVIN](https://github.com/mees/calvin) repository instructions.

```bash
conda activate unilact
export PROJECT_UNILACT_ROOT=/path/to/UniLACT
cd ${PROJECT_UNILACT_ROOT}/scripts
bash evaluate_unilact_on_calvin.sh
```

---

## ⏳ To-Do
- [ ] Support for OXE-pretraining 
- [ ] Release pretrained and finetuned model checkpoints




## 🙏 Acknowledgements

This project builds on top of [Moto](https://github.com/TencentARC/Moto), [CALVIN](https://github.com/mees/calvin). We thank the authors and maintainers for releasing their code and datasets to the community.
