CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 --main_process_port 29500 train_lsd.py \
--enable_xformers_memory_efficient_attention --dataloader_num_workers 4 --learning_rate 2e-5 \
--mixed_precision fp16 --num_validation_images 32 --val_batch_size 32 --max_train_steps 500000 \
--checkpointing_steps 25000 --checkpoints_total_limit 2 --gradient_accumulation_steps 1 \
--seed 42 --encoder_lr_scale 1.0 --train_split_portion 1.0 \
--output_dir logs/coco \
--backbone_config pretrain_dino \
--slot_attn_config configs/coco/slot_attn/config.json \
--scheduler_config none \
--unet_config pretrain_sd \
--dataset_root /shared/s2/lab01/dataset/lsd/coco/train2017 \
--dataset_glob '**/*.jpg' --flip_images --train_batch_size 32 --resolution 256 --validation_steps 5000 \
--tracker_project_name stable_lsd \
--report_to wandb
# --unet_config pretrain_sd \