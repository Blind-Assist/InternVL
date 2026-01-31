# #!/bin/bash
# GPUS=1
# BATCH_SIZE=16
# PER_DEVICE_BATCH_SIZE=2
# GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

# OUTPUT_DIR='work_dirs/internvl2_5_4b_walk_lora'
# META_PATH="./data/walk_vlm/walk_meta.json"
# MODEL_PATH="OpenGVLab/InternVL2_5-4B"
# DS_CONFIG="./zero_stage1_config.json"

# # --- ENVIRONMENT CONFIG ---
# # 1. Add current directory to Python Path so it finds 'internvl' module
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# # 2. Set Library Path so bitsandbytes finds the GPU drivers
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# # 3. WandB Configuration
# export WANDB_PROJECT="internvl-walk"
# export WANDB_ENTITY="vlm-blind-assist"
# export WANDB_NAME="walk-vlm-finetune-02"
# export WANDB_WATCH="false"
# export WANDB_LOG_MODEL="false"

# # Safety Check: Ensure the config file actually exists before running
# if [ ! -f "$DS_CONFIG" ]; then
#   echo "❌ Error: DeepSpeed config not found at $DS_CONFIG"
#   exit 1
# fi

# if [ ! -d "$OUTPUT_DIR" ]; then
#   mkdir -p "$OUTPUT_DIR"
# fi

# # Run Training
# # NOTE: Removed --launcher flag because the code is already patched
# torchrun \
#   --nnodes=1 \
#   --node_rank=0 \
#   --master_addr=127.0.0.1 \
#   --nproc_per_node=${GPUS} \
#   --master_port=6366 \
#   internvl/train/internvl_chat_finetune.py \
#   --model_name_or_path ${MODEL_PATH} \
#   --conv_style "internvl2_5" \
#   --output_dir ${OUTPUT_DIR} \
#   --meta_path ${META_PATH} \
#   --overwrite_output_dir True \
#   --force_image_size 448 \
#   --max_dynamic_patch 6 \
#   --down_sample_ratio 0.5 \
#   --drop_path_rate 0.0 \
#   --freeze_llm True \
#   --freeze_mlp True \
#   --freeze_backbone True \
#   --use_llm_lora 16 \
#   --vision_select_layer -1 \
#   --dataloader_num_workers 4 \
#   --bf16 True \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
#   --gradient_accumulation_steps ${GRADIENT_ACC} \
#   --evaluation_strategy "no" \
#   --save_strategy "steps" \
#   --save_steps 200 \
#   --learning_rate 4e-5 \
#   --weight_decay 0.01 \
#   --warmup_ratio 0.03 \
#   --lr_scheduler_type "cosine" \
#   --logging_steps 1 \
#   --max_seq_length 2048 \
#   --do_train True \
#   --grad_checkpoint True \
#   --group_by_length True \
#   --dynamic_image_size True \
#   --use_thumbnail True \
#   --ps_version 'v2' \
#   --deepspeed "$DS_CONFIG" \
#   --report_to "wandb"













#!/bin/bash
GPUS=1
BATCH_SIZE=16
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

OUTPUT_DIR='work_dirs/internvl2_5_4b_walk_lora'
META_PATH="./data/walk_vlm/walk_meta.json"
VAL_META_PATH="./data/walk_vlm/walk_meta.json"
MODEL_PATH="OpenGVLab/InternVL2_5-4B"
DS_CONFIG="./zero_stage1_config.json"

# --- ENVIRONMENT CONFIG ---
# 1. Add current directory to Python Path so it finds 'internvl' module
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 2. Set Library Path so bitsandbytes finds the GPU drivers
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 3. WandB Configuration
export WANDB_PROJECT="internvl-walk"
export WANDB_ENTITY="vlm-blind-assist"
export WANDB_NAME="walk-vlm-finetune-02"
export WANDB_WATCH="false"
export WANDB_LOG_MODEL="false"

# Safety Check: Ensure the config file actually exists before running
if [ ! -f "$DS_CONFIG" ]; then
  echo "❌ Error: DeepSpeed config not found at $DS_CONFIG"
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Run Training
# NOTE: Removed --launcher flag because the code is already patched
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=6366 \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path ${MODEL_PATH} \
  --conv_style "internvl2_5" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path ${META_PATH} \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --per_device_eval_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "steps" \
  --eval_steps 100 \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 3 \
  --load_best_model_at_end True \
  --metric_for_best_model "eval_loss" \
  --greater_is_better False \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --do_train True \
  --do_eval True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "$DS_CONFIG" \
  --report_to "wandb"