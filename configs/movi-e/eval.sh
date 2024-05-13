CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 src/eval/eval.py \
--ckpt_path logs/lsd/movi-e/output_norm_linear/checkpoint-xxx/ \
--dataset_root /shared/s2/lab01/dataset/lsd/movi/movi-e/movi-e-val-with-label/images/ \
--dataset_glob '**/*.png' --resolution 256 --linear_prob_train_portion 0.83 \
--enable_xformers_memory_efficient_attention --mixed_precision fp16