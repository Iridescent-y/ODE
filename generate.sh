#!/bin/bash

BASE_DIR="/ODE"
DATA_DIR="$BASE_DIR/data"
IMG_DIR="$BASE_DIR/Imgs"
GRAPH_PATH="$DATA_DIR/objects_cooccurrence_graph_T.graphml"
ANNOTATIONS_FILE="$DATA_DIR/annotations.json"
TRUTH_OBJECTS_FILE="$DATA_DIR/truth_objects.txt"
MERGED_OUTPUT_FILE="$IMG_DIR/merged.json"
QUERIES_OUTPUT_DIR="$DATA_DIR"
API_URL="http://127.0.0.1:5000/generate"

echo "Step 1: Running DataExtractor..."
python3 "$BASE_DIR/data_extractor.py" \
    --dataset_path "$DATA_DIR" \
    --annotations_path "$DATA_DIR/BaseInfo/amber_annotations.json" \
    --properties_path "$DATA_DIR/BaseInfo/object-properties.json" \
    --attribute_path "$DATA_DIR/BaseInfo/attribute.json"

if [ $? -ne 0 ]; then
    echo "Error: DataExtractor failed."
    exit 1
fi


echo "Step 2: Running ImageGenerator..."
python3 "$BASE_DIR/image_generator.py" \
    --data_id 1 \
    --data_info_path "$ANNOTATIONS_FILE" \
    --api_url "$API_URL" \
    --output_dir "$IMG_DIR"

if [ $? -ne 0 ]; then
    echo "Error: ImageGenerator failed."
    exit 1
fi


echo "Step 3: Running Image Filter Bash Script..."
bash "$BASE_DIR/filter_images.sh"

if [ $? -ne 0 ]; then
    echo "Error: Image Filter Script failed."
    exit 1
fi

echo "Step 4: Running Queries Generator..."
python3 "$BASE_DIR/queries_generator.py" \
    --types "Random,Fictional,Longtailed,Standard" \
    --image_dir "$IMG_DIR" \
    --annotations_file "$ANNOTATIONS_FILE" \
    --merged_output_file "$MERGED_OUTPUT_FILE" \
    --queries_output_dir "$QUERIES_OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Queries Generator failed."
    exit 1
fi

echo "All steps completed successfully!"
