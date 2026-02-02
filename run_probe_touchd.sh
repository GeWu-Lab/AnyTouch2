CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --master_port=13934 --nproc_per_node=1 main_touchdbench.py --distributed --accum_iter 1 \
    --batch_size 64 \
    --epochs 50 \
    --weight_decay 0.05 \
    --lr 2e-4 \
    --warmup_epochs 0 \
    --output_dir log/touchd \
    --log_dir log/touchd \
    --pooling none \
    --dataset ours_force \
    --num_workers 20 \
    --model_size base \
    --load_path /home/yaoguocai/code-frx/tactile/output_videomae_9.6_align_match_diff/checkpoint-4frames.pth \
    --num_frames 4 \
    --stride 2 \
    --model anytouch \
    --data_sensor digit
    