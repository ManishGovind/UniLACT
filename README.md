<div align="center">

<h2>
  <a href="" style="color:#9C276A; text-decoration:none;">
  UniLACT: Depth-Aware RGB Latent Action Learning for Vision-Language-Action Models
  </a>
</h2>



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

> Training is driven by YAML configs under `unilact/configs/train/` (and UniLARN configs under `unilarn/`).

---

### Stage 1 — UniLARN (Unified Latent Action Learning)

```bash
# (Update this command to your UniLARN entrypoint/config if different)
cd ${PROJECT_UNILACT_ROOT}/unilarn
python train_unilarn.py --config_path "${PROJECT_UNILACT_ROOT}/unilarn/configs/train/train_unilarn_on_calvin.yaml"
```

---

### Stage 2 — UniLACT Pretraining (Cross-modal / Latent Pretrain)

```bash
# (Update this command to your pretraining config if different)
cd ${PROJECT_UNILACT_ROOT}/unilact/train
python train_unilact.py --config_path "${PROJECT_UNILACT_ROOT}/unilact/configs/train/pretrain_unilact_on_calvin.yaml"
```

---

### Stage 3 — Fine-tuning on CALVIN

```bash
cd ${PROJECT_UNILACT_ROOT}/unilact/train

python train_unilact.py   --config_path "${PROJECT_UNILACT_ROOT}/unilact/configs/train/finetune_unilact_on_calvin.yaml"
```


## 🙏 Acknowledgements

This project builds on top of [Moto](https://github.com/TencentARC/Moto), [CALVIN](https://github.com/mees/calvin). We thank the authors and maintainers for releasing their code and datasets to the community.
