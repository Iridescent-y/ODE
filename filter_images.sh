#!/bin/bash
suffixes=("Random" "Fictional" "Longtailed" "Standard")
text="./data/truth_objects.txt"

for suffix in "${suffixes[@]}"; do
    image="./Imgs/$suffix"
    output_file="output_$suffix.json"

    echo "Processing image: $image with text: $text"
    
    python ./YOLO-World/demo/data_filter.py \
        --config ./YOLO-World/configs/pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_800ft_lvis_minival.py \
        --checkpoint ./YOLO-World/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth \
        --image "$image" \
        --text "$text" \
        --topk 100 \
        --threshold 0.005 \
        --device cuda:4 \
        --output_dir ./Imgs \
        --output_file "$output_file"
done

echo "All images processed."

