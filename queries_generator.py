import json
import os
import re
import random
import networkx as nx

def check_image_count(image_folder_path):
    existing_images = set(os.listdir(image_folder_path))
    print(f"Image count in {image_folder_path}: {len(existing_images)}")
    return len(existing_images)

def extract_objects(image_name):
    match = re.search(r'a(?:_[^_]+)*_([^_]+)_a(?:_[^_]+)*_([^_]+)_(?:ee|ev)\d+\.png', image_name)
    return match.groups() if match else (None, None)

# Filter images based on grouped confidence and keep top 2 per prefix
def img_filter(input_file, image_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    grouped_images = {}

    for item in data:
        img_name = item['img']
        labels = item['label']
        object_a, object_b = extract_objects(img_name)
        
        prefix_match = re.match(r"(.*)_(?:ee|ev)\d+", img_name)
        prefix = prefix_match.group(1) if prefix_match else None
        if not prefix:
            continue

        confidences = {label.split()[0]: float(label.split()[1]) for label in labels}
        confidence_a = confidences.get(object_a, 0.0)
        confidence_b = confidences.get(object_b, 0.0)

        confidence_tuple = (confidence_a, confidence_b)
        if prefix not in grouped_images:
            grouped_images[prefix] = []
        grouped_images[prefix].append((img_name, confidence_tuple))

    selected_images = {}
    for prefix, images in grouped_images.items():
        images.sort(key=lambda x: sum(x[1]), reverse=True)
        selected_images[prefix] = [img[0] for img in images[:2]]
        for img_name, _ in images[2:]:
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Deleted image: {img_path}")

def filter_images_by_confidence(input_file, image_dir, min_confidence=0.2):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        img_name = item['img']
        labels = item['label']
        obj_a, obj_b = extract_objects(img_name)
        img_suffix = os.path.splitext(img_name)[0].split('_')[-1]

        obj_a_conf = any(obj == obj_a and float(conf) >= min_confidence for label in labels for obj, conf in [label.split()])
        obj_b_conf = 'ee' not in img_suffix or any(obj == obj_b and float(conf) >= min_confidence for label in labels for obj, conf in [label.split()])

        if not (obj_a_conf and obj_b_conf):
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Deleted low-confidence image: {img_path}")


def filter_annotations(input_json_path, base_image_dir):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = []
    new_id = 0
    for item in data:
        img_path = os.path.join(base_image_dir, item['type'], item['image'])
        if os.path.exists(img_path):
            new_id += 1
            item['id'] = new_id
            filtered_data.append(item)
        else:
            print(f"Missing image: {item['image']} of type {item['type']}, skipping.")
    return filtered_data, new_id


def merge_json_files(file_paths, output_file_path, min_confidence=0.6):
    merged_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            merged_data.extend(json.load(f))

    filtered_data = []
    for item in merged_data:
        filtered_item = {
            "img": item["img"],
            "label": [label for label in item["label"] if float(label.split()[1]) > min_confidence]
        }
        filtered_data.append(filtered_item)

    with open(output_file_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)


def generate_queries(filtered_data, detections_dict, graph, new_id):
    sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]["frequency"], reverse=True)
    gen_queries = []
    dis_existence_queries = []
    dis_attribute_queries = []
    new_id += 1

    for item in filtered_data:
        obj1, obj2 = extract_objects(item['image'])
        if not obj1 or not obj2:
            continue
        
        detection_labels = detections_dict.get(item["image"], [])
        detection_objects = [label.split()[0] for label in detection_labels]
        item["hallu"] = []
        for edge in sorted_edges:
            if edge[0] in item["truth"] and (edge[1] not in item["truth"] and edge[1] not in detection_objects):
                item["hallu"].append(edge[1])
            if len(item["hallu"]) >= 5:
                break
            
        gen_queries.append({
            "id": new_id,
            "type": item["type"],
            "image": item["image"],
            "generative": "Please describe this picture."
        })
        new_id += 1

        dis_existence_queries.append({
            "id": new_id,
            "type": item["type"],
            "image": item["image"],
            "discriminative": f"Is there a {obj1} in the picture?",
            "dis_truth": "yes"
        })
        new_id += 1
        
        if len(item["hallu"]) > 1:
            hallu_obj1 = item["hallu"][0]
            hallu_obj2 = item["hallu"][1]
            dis_existence_queries.append({
                "id": new_id, 
                "type": item["type"],  
                "image": item["image"], 
                "discriminative": f"Is there a {hallu_obj1} in the picture?", 
                "dis_truth": "no"
            })
            new_id += 1
            dis_existence_queries.append({
                "id": new_id, 
                "type": item["type"],  
                "image": item["image"], 
                "discriminative": f"Is there a {hallu_obj2} in the picture?", 
                "dis_truth": "no"
            })
            new_id += 1
        elif len(item["hallu"]) == 1:
            hallu_obj1 = item["hallu"][0]
            dis_existence_queries.append({
                "id": new_id, 
                "type": item["type"],  
                "image": item["image"],  
                "discriminative": f"Is there a {hallu_obj1} in the picture?", 
                "dis_truth": "no"
            })
            new_id += 1
            
        truth_attr = item.get("truth_attr", [])
        hallu_attr = item.get("hallu_attr", [])
        for i, attr in enumerate(truth_attr):
            if not re.match(r"^\d+$", attr):  # Ensure attribute is not numeric
                dis_attribute_queries.append({
                    "id": new_id,
                    "type": item["type"],
                    "image": item["image"],
                    "discriminative": f"Is the {obj1 if i % 2 == 0 else obj2} {attr} in the picture?",
                    "dis_truth": "yes"
                })
                new_id += 1
        for i, attr in enumerate(hallu_attr):
            if not re.match(r"^\d+$", attr):
                dis_attribute_queries.append({
                    "id": new_id,
                    "type": item["type"],
                    "image": item["image"],
                    "discriminative": f"Is the {obj1 if i % 2 == 0 else obj2} {attr} in the picture?",
                    "dis_truth": "no"
                })
                new_id += 1
        
        dis_attribute_queries.append({
                    "id": new_id,
                    "type": item["type"], 
                    "image": item["image"],
                    "dis-type": 'dis-attribute-number',
                    "discriminative": f"Is there one {obj1} in the picture?",
                    "dis_truth": "yes"
                })
        new_id += 1
        
        dis_attribute_queries.append({
                    "id": new_id,
                    "type": item["type"], 
                    "image": item["image"],
                    "dis-type": 'dis-attribute-number',
                    "discriminative": f"Are there two {obj1} in the picture?",
                    "dis_truth": "no"
                })
        new_id += 1
        
        dis_attribute_queries.append({
                    "id": new_id,
                    "type": item["type"], 
                    "image": item["image"],
                    "dis-type": 'dis-attribute-number',
                    "discriminative": f"Are there three {obj1} in the picture?",
                    "dis_truth": "no"
                })
        new_id += 1

    sorted_annotations = sorted(filtered_data, key=lambda x: x['id'])
    return sorted_annotations, gen_queries, dis_existence_queries, dis_attribute_queries



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Queries Generator and process data.")
    parser.add_argument("--types", type=str, required=True, help="Comma-separated list of image types")
    parser.add_argument("--image_dir", type=str, required=True, help="Base directory for image files")
    parser.add_argument("--annotations_file", type=str, required=True, help="Path to the annotations JSON file")
    parser.add_argument("--merged_output_file", type=str, required=True, help="Path to save the merged output JSON file")
    parser.add_argument("--queries_output_dir", type=str, required=True, help="Directory to save query JSON files")
    args = parser.parse_args()

    types = args.types.split(",")
    image_counts = {}

    for type_ in types:
        image_dir = os.path.join(args.image_dir, type_)
        check_image_count(image_dir)
        input_file = os.path.join(args.image_dir, f"output_{type_}.json")
        filter_images_by_confidence(input_file, image_dir)
        image_counts[type_] = check_image_count(image_dir)

    min_count = min(image_counts.values())
    for type_ in types:
        image_dir = f'Imgs/{type_}'
        images = os.listdir(image_dir)
        if len(images) > min_count:
            random.seed(42)
            images_to_delete = random.sample(images, len(images) - min_count)
            for img in images_to_delete:
                img_path = os.path.join(image_dir, img)
                os.remove(img_path)
                print(f"Deleted excess image: {img_path}")


    filtered_data, length = filter_annotations(args.annotations_file, args.image_dir)
    merge_json_files([os.path.join(args.image_dir, f"output_{type_}.json") for type_ in types], args.merged_output_file)

    with open(args.merged_output_file, 'r') as f:
        detections = json.load(f)
    detections_dict = {d['img']: d['label'] for d in detections}


    graph = nx.read_graphml(os.path.join(args.image_dir, "objects_cooccurrence_graph_T.graphml"))
    sorted_annotations, gen_queries, dis_existence_queries, dis_attribute_queries = generate_queries(filtered_data, detections_dict, graph, length)

   
    with open(os.path.join(args.queries_output_dir, "queries-generative.json"), 'w') as f:
        json.dump(gen_queries, f, indent=4)
    with open(os.path.join(args.queries_output_dir, "queries-discriminative_existence.json"), 'w') as f:
        json.dump(dis_existence_queries, f, indent=4)
    with open(os.path.join(args.queries_output_dir, "queries-discriminative_attribute.json"), 'w') as f:
        json.dump(dis_attribute_queries, f, indent=4)

    with open(os.path.join(args.queries_output_dir, "annotations_updated.json"), 'w') as f:
        json.dump(sorted_annotations, f, indent=4)

    print("Queries generation and sorted annotations saving completed successfully.")