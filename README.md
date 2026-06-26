<div align="center">

<h2>
  <a href="https://manishgovind.github.io/unilact-vla/" style="color:#9C276A; text-decoration:none;">
    UniLACT: Depth-Aware RGB Latent Action Learning for Vision-Language-Action Models
  </a>
</h2>

<p>
  <a href="https://arxiv.org/abs/2602.20231">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat&logo=arxiv" />
  </a>
  <a href="https://manishgovind.github.io/unilact-vla/">
    <img src="https://img.shields.io/badge/Website-Project%20Page-2ea44f?style=flat" />
  </a>
  <a href="https://huggingface.co/mgovind7/UniLACT">
    <img src="https://img.shields.io/badge/Models-Hugging%20Face-ffd21e?style=flat&logo=huggingface" />
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

## 🤗 Model Zoo

Pretrained and finetuned checkpoints are hosted on Hugging Face: [mgovind7/UniLACT](https://huggingface.co/mgovind7/UniLACT). Each checkpoint directory contains `config.yaml` and `pytorch_model.bin`.

| Model Name | HF Path | Note |
| --- | --- | --- |
| `unilarn-calvin` | [`unilarn_trained_on_calvin`](https://huggingface.co/mgovind7/UniLACT/tree/main/unilarn_trained_on_calvin) | Stage-1 UniLARN trained on CALVIN RGB-D. |
| `unilarn-oxe` | [`unilarn_trained_on_oxe`](https://huggingface.co/mgovind7/UniLACT/tree/main/unilarn_trained_on_oxe) | Stage-1 UniLARN trained on Open X-Embodiment. |
| `unilact-pretrain-calvin` | [`unilact_pretrained_on_calvin`](https://huggingface.co/mgovind7/UniLACT/tree/main/unilact_pretrained_on_calvin) | Stage-2 depth-aware latent pretraining on CALVIN. |
| `unilact-pretrain-oxe` | [`unilact_pretrained_on_oxe`](https://huggingface.co/mgovind7/UniLACT/tree/main/unilact_pretrained_on_oxe) | Stage-2 depth-aware latent pretraining on OXE. |
| `unilact-calvin` | [`unilact_finetuned_on_calvin`](https://huggingface.co/mgovind7/UniLACT/tree/main/unilact_finetuned_on_calvin) | Fine-tuned on CALVIN for ABC→D evaluation (in-domain pretraining). |
| `unilact-calvin-ood` | [`unilact_finetuned_on_calvin_from_pretrained_oxe`](https://huggingface.co/mgovind7/UniLACT/tree/main/unilact_finetuned_on_calvin_from_pretrained_oxe) | Fine-tuned on CALVIN after out-of-domain OXE pretraining. |

### Download all checkpoints

```bash
pip install -U "huggingface_hub[cli]"
export PROJECT_UNILACT_ROOT=/path/to/UniLACT

hf download mgovind7/UniLACT --local-dir ${PROJECT_UNILACT_ROOT}/checkpoints
```

### Download a single checkpoint

```bash
# Stage 3 fine-tuned model used for CALVIN evaluation
hf download mgovind7/UniLACT \
  --local-dir ${PROJECT_UNILACT_ROOT}/checkpoints \
  --include "unilact_finetuned_on_calvin/*"
```

After downloading, point the training or evaluation configs at the local checkpoint directory. For example, set `unilarn_path` in `unilact/configs/train/pretrain_unilact_on_calvin.yaml` to `${PROJECT_UNILACT_ROOT}/checkpoints/unilarn_trained_on_calvin`, or set `UniLACT_PATH` in `scripts/evaluate_unilact_on_calvin.sh` to `${PROJECT_UNILACT_ROOT}/checkpoints/unilact_finetuned_on_calvin`.

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
- [ ] Training Data preparation
- [ ] Support for OXE-pretraining
- [x] Release pretrained and finetuned model checkpoints




## 🙏 Acknowledgements

This project builds on top of [Moto](https://github.com/TencentARC/Moto), [CALVIN](https://github.com/mees/calvin). We thank the authors for their open-sourced work.

## 📝 Citation

If you find our work useful, please cite:

```bibtex
@article{govind2026unilactdepthawarergblatent,
  title= {UniLACT: Depth-Aware RGB Latent Action Learning for Vision-Language-Action Models},
  author= {Manish Kumar Govind and Dominick Reilly and Pu Wang and Srijan Das},
  journal={arXiv preprint arXiv:2602.20231},
  year={2026}
}



