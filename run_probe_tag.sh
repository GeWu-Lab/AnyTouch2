CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --master_port=21634 --nproc_per_node=1 main_objbench.py --distributed --accum_iter 1 \
    --batch_size 64 \
    --epochs 50 \
    --weight_decay 0.05 \
    --lr 2e-4 \
    --warmup_epochs 0 \
    --output_dir log/tag \
    --log_dir log/tag \
    --pooling cls \
    --dataset material \
    --num_workers 24 \
    --model_size base \
    --load_path checkpoints/checkpoint-4frames.pth \
    --num_frames 4 \
    --stride 2 \
    --model anytouch

    