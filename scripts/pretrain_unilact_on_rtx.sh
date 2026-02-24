export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ${PROJECT_DILACT_ROOT}/UniLACT/train
accelerate launch --main_process_port 29508 train_UniLACT.py --config_path "${PROJECT_DILACT_ROOT}/UniLACT/configs/train/pretrain_UniLACT_calvin_unified.yaml"












