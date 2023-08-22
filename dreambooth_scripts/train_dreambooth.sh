export MODEL_NAME="stablediffusionapi/cyberrealistic"
export INSTANCE_DIR="./monica_dreambooth_all_masked_2x/"
export CLASS_DIR="./monica_class_dir"
export OUTPUT_DIR="./monica_masked_dreambooth_2x_16_pp_sd2.1_checkpoints_800"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks woman" \
  --class_prompt="a photo of woman" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=300 \
  --max_train_steps=1000 \
  --report_to="wandb" \
  --validation_prompt="a photo of sks woman" \
  --validation_steps=100 \
  --seed="0"
