export CUDA_VISIBLE_DEVICES=0
export CALVIN_ROOT=${PROJECT_UNILACT_ROOT}/../calvin/
export MESA_GL_VERSION_OVERRIDE=3.3
export PROJECT_UNILACT_ROOT=/path/to/UniLACT/


EvalCALVIN() {
cd ${PROJECT_UniLACT_ROOT}/UniLACT/evaluation/robot_manipulation_benchmarks/calvin
accelerate launch evaluate_calvin.py \
    --UniLACT_path ${UniLACT_PATH} \
    --test_chunk_size ${TEST_CHUNK_SIZE} \
    --mask_latent_action_probability ${MLMP} \
    --eval_dir ${EVAL_DIR}
echo "Done! EvalCALVIN ${EVAL_DIR}"
}

MLMP=0.0
TEST_CHUNK_SIZE=8
UniLACT_PATH="${PROJECT_UniLACT_ROOT}/UniLACT/outputs_refresh/UniLACT_finetuned_on_calvin"
EVAL_DIR="${PROJECT_UniLACT_ROOT}/UniLACT/evaluation/robot_manipulation_benchmarks/calvin/eval_results"
EvalCALVIN





