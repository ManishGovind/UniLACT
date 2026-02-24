export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${PROJECT_UNILACT_ROOT}/unilact/train
accelerate launch --main_process_port 29508 train_unilact.py --config_path "${PROJECT_UNILACT_ROOT}/unilact/configs/train/pretrain_unilact_on_calvin.yaml"












