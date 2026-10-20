#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${1:-bc2gm1}"
case "$DATASET" in
  bc2gm1|bc4chemed|bc5cdr_disease|bc5cdr_chemical|ncbi_disease|jnlpba) ;;
  *)
    echo "Usage: bash scripts/run_main_experiment.sh {bc2gm1|bc4chemed|bc5cdr_disease|bc5cdr_chemical|ncbi_disease|jnlpba}" >&2
    exit 1
    ;;
esac
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL")}"
DATA_DIR="${DATA_DIR:-./data/${DATASET}}"
GRPO_DATA_DIR="${GRPO_DATA_DIR:-./data_swift_bioner_grpo/${DATASET}_random_code}"
EPOCHS="${EPOCHS:-5}"
GRPO_EPOCHS="${GRPO_EPOCHS:-2}"
STYLE_SEED="${STYLE_SEED:-42}"
GRPO_STYLE=mixed
CODE_VOTE_THRESHOLD=2

SFT_OUT_DIR="${SFT_OUT_DIR:-./outputs/${DATASET}_random_code_${MODEL_TAG}_lora_sft_ep${EPOCHS}}"
GRPO_OUT_DIR="${GRPO_OUT_DIR:-./outputs/${DATASET}_random_code_${GRPO_STYLE}_${MODEL_TAG}_lora_sft_ep${EPOCHS}_grpo_ep$GRPO_EPOCHS}"

RUN_PREPARE_GRPO_DATA="${RUN_PREPARE_GRPO_DATA:-true}"
RUN_SFT="${RUN_SFT:-true}"
RUN_SFT_TEST="${RUN_SFT_TEST:-true}"
RUN_GRPO="${RUN_GRPO:-true}"
RUN_GRPO_TEST="${RUN_GRPO_TEST:-true}"

for style in python java cpp; do
  for split in train dev test; do
    path="$DATA_DIR/${style}_${split}.jsonl"
    if [[ ! -f "$path" ]]; then
      echo "Missing dataset file: $path" >&2
      exit 1
    fi
  done
done

if [[ "$RUN_PREPARE_GRPO_DATA" == "true" ]]; then
  for style in python java cpp; do
    python tools/convert_sft_to_grpo.py \
      --input "$DATA_DIR/${style}_train.jsonl" \
      --output "$GRPO_DATA_DIR/${style}_train.jsonl"
  done

  mixed_train_data="$GRPO_DATA_DIR/mixed_train.jsonl"
  : > "$mixed_train_data"
  for style in python java cpp; do
    cat "$GRPO_DATA_DIR/${style}_train.jsonl" >> "$mixed_train_data"
  done
fi

GRPO_TRAIN_DATA="$GRPO_DATA_DIR/${GRPO_STYLE}_train.jsonl"
if [[ "$RUN_GRPO" == "true" && ! -f "$GRPO_TRAIN_DATA" ]]; then
  echo "Missing GRPO dataset file: $GRPO_TRAIN_DATA" >&2
  echo "Set RUN_PREPARE_GRPO_DATA=true or provide GRPO_DATA_DIR." >&2
  exit 1
fi

export BIONER_MODEL="$MODEL"
export BIONER_DEV_DATA="$DATA_DIR/python_dev.jsonl"
export BIONER_CODE_DATA_DIR="$DATA_DIR"
export BIONER_CODE_STYLE_SEED="$STYLE_SEED"
export BIONER_EVAL_GPU="$GPU"
export BIONER_EVAL_BACKEND="${BIONER_EVAL_BACKEND:-vllm}"
export BIONER_EVAL_GPU_MEM="${BIONER_EVAL_GPU_MEM:-0.25}"
export BIONER_CODE_VOTE_THRESHOLD="$CODE_VOTE_THRESHOLD"

mkdir -p "$SFT_OUT_DIR"
if [[ "$RUN_SFT" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" swift sft \
    --model "$MODEL" \
    --dataset "$DATA_DIR/python_train.jsonl" \
    --val_dataset "$DATA_DIR/python_dev.jsonl" \
    --dataset_shuffle true \
    --lazy_tokenize true \
    --output_dir "$SFT_OUT_DIR" \
    --add_version false \
    --external_plugins tools/bioner_callbacks.py \
    --callbacks random_code_style bioner_code_vote_f1 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs "$EPOCHS" \
    --save_strategy epoch \
    --eval_strategy no \
    --load_best_model_at_end false \
    --create_checkpoint_symlink true \
    --max_length 1536 \
    --max_new_tokens 256 \
    --enable_thinking false \
    --report_to swanlab \
    --swanlab_project BIO_NER \
    --swanlab_exp_name "${DATASET}_random_code_vote_sft_ep${EPOCHS}"
fi

SFT_ADAPTER="${SFT_ADAPTER:-$SFT_OUT_DIR/best_dev_f1}"
if [[ "$RUN_SFT_TEST" == "true" || "$RUN_GRPO" == "true" ]]; then
  if [[ ! -e "$SFT_ADAPTER" ]]; then
    echo "Missing SFT adapter: $SFT_ADAPTER" >&2
    echo "Set SFT_ADAPTER=/path/to/adapter if reusing an existing SFT checkpoint." >&2
    exit 1
  fi
fi

run_style_infer_eval() {
  local adapter="$1"
  local output_dir="$2"
  local gpu_mem="$3"

  mkdir -p "$output_dir"
  for style in python java cpp; do
    local prediction="$output_dir/test_${style}_predictions_vllm.jsonl"
    local metrics="$output_dir/test_${style}_metrics.json"
    local pending_prediction="$output_dir/.test_${style}_predictions.tmp.jsonl"
    rm -f -- "$pending_prediction"

    CUDA_VISIBLE_DEVICES="$GPU" swift infer \
      --model "$MODEL" \
      --adapters "$adapter" \
      --val_dataset "$DATA_DIR/${style}_test.jsonl" \
      --result_path "$pending_prediction" \
      --infer_backend vllm \
      --max_new_tokens 256 \
      --enable_thinking false \
      --vllm_gpu_memory_utilization "$gpu_mem" \
      --vllm_max_model_len 2048 \
      --vllm_tensor_parallel_size 1 \
      --vllm_max_lora_rank 16

    mv -f -- "$pending_prediction" "$prediction"
    python tools/eval_swift_bioner_f1.py --input "$prediction" --output "$metrics"
  done

  python tools/eval_swift_bioner_f1.py \
    --python "$output_dir/test_python_predictions_vllm.jsonl" \
    --java "$output_dir/test_java_predictions_vllm.jsonl" \
    --cpp "$output_dir/test_cpp_predictions_vllm.jsonl" \
    --output "$output_dir/test_code_vote_metrics.json"
}

if [[ "$RUN_SFT_TEST" == "true" ]]; then
  run_style_infer_eval "$SFT_ADAPTER" "$SFT_OUT_DIR" "${SFT_TEST_VLLM_GPU_MEM:-0.25}"
fi

if [[ "$RUN_GRPO" == "true" ]]; then
  mkdir -p "$GRPO_OUT_DIR"

  CUDA_VISIBLE_DEVICES="$GPU" swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --adapters "$SFT_ADAPTER" \
    --ref_adapters "$SFT_ADAPTER" \
    --dataset "$GRPO_TRAIN_DATA" \
    --output_dir "$GRPO_OUT_DIR" \
    --add_version false \
    --external_plugins tools/grpo_bioner_plugin.py tools/bioner_callbacks.py \
    --callbacks bioner_code_vote_f1 \
    --reward_funcs bioner_format bioner_f1 \
    --reward_weights 0.02 1.0 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --per_device_train_batch_size "${GRPO_BATCH_SIZE:-16}" \
    --gradient_accumulation_steps "${GRPO_GRAD_ACCUM:-4}" \
    --per_device_eval_batch_size 1 \
    --num_train_epochs "$GRPO_EPOCHS" \
    --save_strategy steps \
    --save_steps "${GRPO_SAVE_STEPS:-300}" \
    --eval_strategy no \
    --create_checkpoint_symlink true \
    --max_length 1536 \
    --max_completion_length 256 \
    --num_generations "${GRPO_NUM_GENERATIONS:-4}" \
    --num_generations_eval 1 \
    --temperature 0.9 \
    --top_p 0.95 \
    --enable_thinking false \
    --log_completions true \
    --logging_steps 50 \
    --report_to swanlab \
    --swanlab_project BIO_NER \
    --swanlab_exp_name "${DATASET}_random_code_${GRPO_STYLE}_${MODEL_TAG}_lora_sft_ep${EPOCHS}_grpo_ep$GRPO_EPOCHS"
fi

GRPO_ADAPTER="${GRPO_ADAPTER:-$GRPO_OUT_DIR/best_dev_f1}"
if [[ "$RUN_GRPO_TEST" == "true" ]]; then
  if [[ ! -e "$GRPO_ADAPTER" ]]; then
    echo "Missing GRPO adapter: $GRPO_ADAPTER" >&2
    echo "Set GRPO_ADAPTER=/path/to/adapter if reusing an existing GRPO checkpoint." >&2
    exit 1
  fi

  run_style_infer_eval "$GRPO_ADAPTER" "$GRPO_OUT_DIR" "${GRPO_TEST_VLLM_GPU_MEM:-0.3}"
fi

echo "Physical SFT epochs: $EPOCHS"
echo "Code-style seed: $STYLE_SEED"
echo "GRPO style: $GRPO_STYLE"
echo "Code vote threshold: $CODE_VOTE_THRESHOLD"
echo "SFT output: $SFT_OUT_DIR"
echo "SFT voting metrics: $SFT_OUT_DIR/test_code_vote_metrics.json"
echo "GRPO output: $GRPO_OUT_DIR"
echo "GRPO voting metrics: $GRPO_OUT_DIR/test_code_vote_metrics.json"
