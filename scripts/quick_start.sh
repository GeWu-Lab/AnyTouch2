CUDA_VISIBLE_DEVICES=0 python -u scripts/quick_start.py \
    --output_dir log/quick_start \
    --model_size base \
    --load_path checkpoints/checkpoint-4frames.pth \
    --num_frames 4 \
    --stride 2 \
    --model anytouch \
    --data_sensor gelsight_mini