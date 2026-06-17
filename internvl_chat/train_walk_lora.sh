# #!/bin/bash

# # ============================================
# # InternVL2.5-4B LoRA Fine-tuning for WalkVLM
# # FIXED VERSION
# # ============================================

# set -e

# GPUS=1
# BATCH_SIZE=16
# PER_DEVICE_BATCH_SIZE=2
# GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

# # --- PATHS ---
# OUTPUT_DIR='work_dirs/internvl2_5_4b_walk_lora'
# TRAIN_META_PATH="./data/walk_vlm/walk_train_meta.json"
# EVAL_META_PATH="./data/walk_vlm/walk_val_meta.json"
# MODEL_PATH="OpenGVLab/InternVL2_5-4B"
# DS_CONFIG="./zero_stage1_config.json"

# # --- ENVIRONMENT CONFIG ---
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# # WandB Configuration
# export WANDB_PROJECT="internvl-walk"
# export WANDB_ENTITY="vlm-blind-assist"
# export WANDB_NAME="InternVL2_5-4B_walkvlm_lora_fixed"
# export WANDB_WATCH="false"
# export WANDB_LOG_MODEL="false"

# # --- PRE-FLIGHT CHECKS ---
# echo "============================================"
# echo "🚀 InternVL2.5-4B LoRA Training (FIXED)"
# echo "============================================"

# if [ ! -f "$DS_CONFIG" ]; then
#     echo "❌ Error: DeepSpeed config not found at $DS_CONFIG"
#     exit 1
# fi
# echo "✅ DeepSpeed config: $DS_CONFIG"

# if [ ! -f "$TRAIN_META_PATH" ]; then
#     echo "❌ Error: Train meta not found at $TRAIN_META_PATH"
#     exit 1
# fi
# echo "✅ Train meta: $TRAIN_META_PATH"

# if [ ! -f "$EVAL_META_PATH" ]; then
#     echo "⚠️  Warning: Eval meta not found at $EVAL_META_PATH"
#     EVAL_ARGS=""
# else
#     echo "✅ Eval meta: $EVAL_META_PATH"
#     EVAL_ARGS="--eval_meta_path ${EVAL_META_PATH} --do_eval True --evaluation_strategy steps --eval_steps 25 --per_device_eval_batch_size ${PER_DEVICE_BATCH_SIZE}"
# fi

# # Clean old checkpoints
# rm -rf ${OUTPUT_DIR}/checkpoint-* 2>/dev/null || true
# mkdir -p "$OUTPUT_DIR"

# echo "============================================"
# echo "⚙️  Key Settings:"
# echo "   freeze_llm: False (LoRA will train)"
# echo "   freeze_mlp: False"  
# echo "   freeze_backbone: True"
# echo "   use_llm_lora: 16"
# echo "============================================"

# # --- RUN TRAINING ---
# torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr=127.0.0.1 \
#     --nproc_per_node=${GPUS} \
#     --master_port=63666 \
#     internvl/train/internvl_chat_finetune.py \
#     --model_name_or_path ${MODEL_PATH} \
#     --conv_style "internvl2_5" \
#     --output_dir ${OUTPUT_DIR} \
#     --meta_path ${TRAIN_META_PATH} \
#     ${EVAL_ARGS} \
#     --overwrite_output_dir True \
#     --force_image_size 448 \
#     --max_dynamic_patch 6 \
#     --down_sample_ratio 0.5 \
#     --drop_path_rate 0.0 \
#     --freeze_llm False \
#     --freeze_mlp False \
#     --freeze_backbone True \
#     --use_llm_lora 16 \
#     --vision_select_layer -1 \
#     --dataloader_num_workers 4 \
#     --bf16 True \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
#     --gradient_accumulation_steps ${GRADIENT_ACC} \
#     --save_strategy "steps" \
#     --save_steps 50 \
#     --save_total_limit 2 \
#     --learning_rate 1e-4 \
#     --weight_decay 0.01 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --max_seq_length 2048 \
#     --do_train True \
#     --grad_checkpoint True \
#     --group_by_length True \
#     --dynamic_image_size True \
#     --use_thumbnail True \
#     --ps_version "v2" \
#     --deepspeed "$DS_CONFIG" \
#     --report_to "wandb"

# echo "============================================"
# echo "✅ Training completed!"
# echo "============================================"




















# #!/bin/bash

# # ============================================
# # InternVL3-2B LoRA Fine-tuning for WalkVLM
# # With Early Stopping (3 Epochs)
# # ============================================

# set -e

# GPUS=1
# BATCH_SIZE=16
# PER_DEVICE_BATCH_SIZE=2
# GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

# # --- PATHS ---
# OUTPUT_DIR='work_dirs/internvl3_1b_walk_lora'
# TRAIN_META_PATH="./data/walk_vlm/walk_train_meta.json"
# EVAL_META_PATH="./data/walk_vlm/walk_val_meta.json"
# MODEL_PATH="OpenGVLab/InternVL3-1B"
# DS_CONFIG="./zero_stage1_config.json"

# # --- ENVIRONMENT CONFIG ---
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# # WandB Configuration
# export WANDB_PROJECT="internvl-walk"
# export WANDB_ENTITY="vlm-blind-assist"
# export WANDB_NAME="InternVL3-1B_walkvlm_lora_earlystop"
# export WANDB_WATCH="false"
# export WANDB_LOG_MODEL="false"

# # --- PRE-FLIGHT CHECKS ---
# echo "============================================"
# echo "🚀 InternVL3-1B LoRA Training (Early Stopping)"
# echo "============================================"

# if [ ! -f "$DS_CONFIG" ]; then
#     echo "❌ Error: DeepSpeed config not found at $DS_CONFIG"
#     exit 1
# fi

# if [ ! -f "$TRAIN_META_PATH" ]; then
#     echo "❌ Error: Train meta not found at $TRAIN_META_PATH"
#     exit 1
# fi

# # Early stopping REQUIRES evaluation dataset
# if [ ! -f "$EVAL_META_PATH" ]; then
#     echo "❌ Error: Early stopping requires eval meta at $EVAL_META_PATH"
#     exit 1
# fi

# # Eval args with early stopping enabled
# EVAL_ARGS="--eval_meta_path ${EVAL_META_PATH} \
#     --do_eval True \
#     --evaluation_strategy steps \
#     --eval_steps 15 \
#     --per_device_eval_batch_size ${PER_DEVICE_BATCH_SIZE} \
#     --load_best_model_at_end True \
#     --metric_for_best_model eval_loss \
#     --greater_is_better False"

# mkdir -p "$OUTPUT_DIR"

# echo "⚙️  Early Stopping Config:"
# echo "   patience: 5 evaluations"
# echo "   threshold: 0.1% improvement"
# echo "   eval_steps: 15"
# echo "   epochs: 3 (max)"
# echo "============================================"

# # --- RUN TRAINING ---
# torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr=127.0.0.1 \
#     --nproc_per_node=${GPUS} \
#     --master_port=63666 \
#     internvl/train/internvl_chat_finetune.py \
#     --model_name_or_path ${MODEL_PATH} \
#     --conv_style "internvl2_5" \
#     --use_fast_tokenizer False \
#     --output_dir ${OUTPUT_DIR} \
#     --meta_path ${TRAIN_META_PATH} \
#     ${EVAL_ARGS} \
#     --overwrite_output_dir True \
#     --force_image_size 448 \
#     --max_dynamic_patch 8 \
#     --down_sample_ratio 0.5 \
#     --drop_path_rate 0.0 \
#     --freeze_llm True \
#     --freeze_mlp False \
#     --freeze_backbone True \
#     --use_llm_lora 128 \
#     --vision_select_layer -1 \
#     --dataloader_num_workers 4 \
#     --bf16 True \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
#     --gradient_accumulation_steps ${GRADIENT_ACC} \
#     --save_strategy "steps" \
#     --save_steps 15 \
#     --save_total_limit 5 \
#     --learning_rate 4e-5 \
#     --weight_decay 0.05 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --max_seq_length 4096 \
#     --do_train True \
#     --grad_checkpoint True \
#     --group_by_length True \
#     --dynamic_image_size True \
#     --use_thumbnail True \
#     --ps_version "v2" \
#     --deepspeed "$DS_CONFIG" \
#     --report_to "wandb" \
#     2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

# echo "✅ Training completed!"































#!/bin/bash

# ============================================
# InternVL3-1B LoRA Fine-tuning for WalkVLM
# Standard Training (Full 3 Epochs)
# ============================================

set -e

GPUS=1
BATCH_SIZE=16
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

# --- PATHS ---
OUTPUT_DIR='work_dirs/internvl3_1b_walk_lora'
TRAIN_META_PATH="./data/walk_vlm/walk_train_meta.json"
EVAL_META_PATH="./data/walk_vlm/walk_val_meta.json"
MODEL_PATH="OpenGVLab/InternVL3-1B"
DS_CONFIG="./zero_stage1_config.json"

# --- ENVIRONMENT CONFIG ---
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# WandB Configuration
export WANDB_PROJECT="internvl-walk"
export WANDB_ENTITY="vlm-blind-assist"
export WANDB_NAME="InternVL3-1B_walkvlm_lora_2_epochs_without_ealry_stopping"
export WANDB_WATCH="false"
export WANDB_LOG_MODEL="false"

# --- PRE-FLIGHT CHECKS ---
echo "============================================"
echo "🚀 InternVL3-1B LoRA Training (Full Epochs) without ealry stopping"
echo "============================================"

if [ ! -f "$DS_CONFIG" ]; then
    echo "❌ Error: DeepSpeed config not found at $DS_CONFIG"
    exit 1
fi

if [ ! -f "$TRAIN_META_PATH" ]; then
    echo "❌ Error: Train meta not found at $TRAIN_META_PATH"
    exit 1
fi

if [ ! -f "$EVAL_META_PATH" ]; then
    echo "❌ Error: Eval meta at $EVAL_META_PATH"
    exit 1
fi

# Standard evaluation configurations (Early stopping removed)
EVAL_ARGS="--eval_meta_path ${EVAL_META_PATH} \
    --do_eval True \
    --evaluation_strategy steps \
    --eval_steps 15 \
    --per_device_eval_batch_size ${PER_DEVICE_BATCH_SIZE}"

mkdir -p "$OUTPUT_DIR"

echo "⚙️  Training Config:"
echo "   Early Stopping: DISABLED"
echo "   eval_steps: 15"
echo "   epochs: 3 (will complete fully)"
echo "============================================"

# --- RUN TRAINING ---
# --- RUN TRAINING ---
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=63666 \
    internvl/train/internvl_chat_finetune.py \
    --model_name_or_path ${MODEL_PATH} \
    --conv_style "internvl2_5" \
    --use_fast_tokenizer False \
    --output_dir ${OUTPUT_DIR} \
    --meta_path ${TRAIN_META_PATH} \
    ${EVAL_ARGS} \
    --overwrite_output_dir True \
    --force_image_size 448 \
    --max_dynamic_patch 8 \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.0 \
    --freeze_llm True \
    --freeze_mlp False \
    --freeze_backbone True \
    --use_llm_lora 128 \
    --vision_select_layer -1 \
    --dataloader_num_workers 4 \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --save_strategy "steps" \
    --save_steps 15 \
    --save_total_limit 5 \
    --learning_rate 4e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_seq_length 4096 \
    --do_train True \
    --grad_checkpoint True \
    --group_by_length True \
    --dynamic_image_size True \
    --use_thumbnail True \
    --ps_version "v2" \
    --deepspeed "$DS_CONFIG" \
    --report_to "wandb" \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

echo "✅ Training completed!"