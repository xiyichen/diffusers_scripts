# export MODEL_NAME="SG161222/Realistic_Vision_V5.1_noVAE"
# export INSTANCE_DIR="./monica_dreambooth_all/"
# export OUTPUT_DIR="./monica_dreambooth_all_checkpoints"
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of <Monica Geller> wearing <Monica's dress>" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --train_text_encoder \
#   --gradient_accumulation_steps=1 \
#   --checkpointing_steps=500 \
#   --learning_rate=1e-5 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=8000 \
#   --validation_prompt="a photo of <Monica Geller> wearing <Monica's dress>" \
#   --validation_epochs=10 \
#   --seed="0"

# # export MODEL_NAME="dreamlike-art/dreamlike-photoreal-2.0"
# export MODEL_NAME="stablediffusionapi/cyberrealistic"
# export INSTANCE_DIR="./monica_dreambooth_all_masked/"
# export OUTPUT_DIR="./monica_dreambooth_all_masked_cr_checkpoints"
# accelerate launch train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of <Monica Geller>" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --train_text_encoder \
#   --gradient_accumulation_steps=1 \
#   --checkpointing_steps=500 \
#   --learning_rate=1e-5 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=10000 \
#   --validation_prompt="a photo of <Monica Geller>" \
#   --validation_epochs=10 \
#   --seed="0" \
#   # --enable_xformers_memory_efficient_attention \
#   # --use_8bit_adam \
#   # --gradient_checkpointing \
#   # --gradient_accumulation_steps=1 



export MODEL_NAME="stablediffusionapi/cyberrealistic"
export INSTANCE_DIR="./monica_dreambooth_all/"
export CLASS_DIR="./monica_class_dir"
export OUTPUT_DIR="./monica_dreambooth_all_wbg_pp_cr_lora_checkpoints_10000"
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of [P] woman" \
  --class_prompt="a photo of woman" \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=500 \
  --learning_rate=5e-6 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --validation_prompt="a photo of [P] woman" \
  --validation_epochs=10 \
  --seed="0" \
  # --enable_xformers_memory_efficient_attention \
  # --use_8bit_adam \
  # --gradient_checkpointing \
  # --gradient_accumulation_steps=1 
