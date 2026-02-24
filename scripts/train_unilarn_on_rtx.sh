
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${PROJECT_UNILACT_ROOT}/unilarn/train
accelerate launch --main_process_port 29503 train_unilarn.py --config_path "${PROJECT_UNILACT_ROOT}/unilarn/configs/train/data_rtx_vq_multi_modal.yaml"


