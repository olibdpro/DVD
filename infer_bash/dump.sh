CKPT='ckpt'
DATASET='Sintel'          # Sintel|TartanAir|Spring|SceneNet|UnrealEXR|DepthCollapse,
                          # or a folder of frames / a video file
DATA_ROOT=''              # empty -> configs/dataset/<dataset>.yaml root_dir
OUT='./depth_dump'          # -> $OUT/{train,512,1024,original}/<clip>/frame_*.png

MAX_CLIPS=10
CLIP_LEN=16               # dataset clips; must be <= the shortest scene

WINDOW_SIZE=81
OVERLAP=21

python test_script/eval_depth_dump.py --ckpt $CKPT --dataset $DATASET \
    ${DATA_ROOT:+--data_root $DATA_ROOT} -o $OUT \
    --max_clips $MAX_CLIPS --clip_len $CLIP_LEN \
    --window_size $WINDOW_SIZE --overlap $OVERLAP
