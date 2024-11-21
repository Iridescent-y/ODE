import json
import argparse
import base64
import requests
import os
import random
import time

import re
from collections import defaultdict
import networkx as nx

from concept_composer import (
    object_combination_Longtailed,
    object_combination_Fictional,
    object_combination_Random,
    object_combination_Standard
)

class ImageGenerator:
    """
    generate images based on specific prompts and configurations, 
    using an external API for image generation
    """
    def __init__(self, data_id, data_info_path, api_url, output_dir):
        self.data_id = data_id
        self.api_url = api_url
        self.output_dir = output_dir
        self.data_info_path = data_info_path
        if os.path.exists(self.data_info_path):
            with open(self.data_info_path, 'r') as f:
                self.data_info = json.load(f)
        else:
            self.data_info = []

        self.data_template = {
            'prompt': '',
            'negative_prompt': '',
            'sampler_index': 'DPM++ SDE',
            'seed': 31339,
            'steps': 20,
            'width': 768,
            'height': 768,
            'cfg_scale': 10
        }

        self.seeds = [31339, 452943117] + [random.randint(0, 1000000) for _ in range(6)]
        self.steps = [32, 30]
        self.cfg_scales = [7, 10]
        self.extra_prompts = {
            "ee":[
                '(clear and obvious entity)(entity A and entity B is equally important)(Photography)(best quality)(masterpiece)(concise)(complete body)', 
                '(clear and obvious entity)(entity A and entity B is equally important)(ANIME)(best quality)(masterpiece)(concise)(complete body)'
            ],
            "ev":[
                '(highlight the subject and only this environment)(Entity as the main body and clear and obvious)(Photography)(best quality)(masterpiece)',
                '(highlight the subject and only this environment)(Entity as the main body and clear and obvious)(ANIME)(best quality)(masterpiece)(physically-based rendering)(clear subject)'
            ]
        }
        self.negative_prompts = {
            "ee":[
                '(missing entity)(bad anatomy)(deformed)(disconnected limbs)(missing limb)(out of focus)(mutated)(incomplete body)(poorly face)', 
                '(missing entity)(bad anatomy)(deformed)(disconnected limbs)(missing limb)(out of focus)(mutated)(incomplete body)(poorly face)'
            ],
            "ev":[
            '(extra environmental elements)(extra entity)(bad anatomy)(deformed)(disconnected limbs)(missing limb)(out of focus)(mutated)',
            '(extra environmental elements)(extra entity)(bad anatomy)(deformed)(disconnected limbs)(missing limb)(out of focus)(mutated)'
            ]
        }
        print("Image Generator initialized.")

    def submit_post(self, url: str, data: dict):
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return response

    def save_encoded_image(self, b64_image: str, output_path: str):
        with open(output_path, 'wb') as image_file:
            print("save:",output_path)
            image_file.write(base64.b64decode(b64_image))

    def generate_image(self, data: dict, save_image_path: str):

        start_time = time.time()
        response = self.submit_post(self.api_url, data)
        self.save_encoded_image(response.json()['image'], save_image_path)
        print("Finished: ", save_image_path)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

    def update_data_info(self, save_image_path,  data_info):
        
        Data_Item = {
            "id": self.data_id,
            "type": save_image_path.split("/")[-2],
            "image": save_image_path.split("/")[-1],
            "info_type": data_info["info_type"],
            "truth": data_info["truth"],
            "truth_attr": data_info["attr_truth"],
            "hallu_attr": data_info["attr_hallu"]
        }
        self.data_id += 1
        
        self.data_info.append(Data_Item)
        
        with open(self.data_info_path, "w", encoding="utf-8") as f:
            json.dump(self.data_info, f, indent=4)

    def check_isinstance(self, word):
        if isinstance(word, list):
            word = word[0]
        return word
    
    def select_random_word(self, combination, group_type):
        if combination:
            index = random.randint(0, len(combination) - 1)
            item = combination[index]
            item = tuple(item)
            if group_type in ["Standard", "Longtailed"]:
                selected_word = item[0]
                not_selected_word = item[1]
            else:    
                selected_word = random.choice(item)
                not_selected_word = [word for word in item if word != selected_word]
            
            selected_word = self.check_isinstance(selected_word)
            not_selected_word = self.check_isinstance(not_selected_word)

            return selected_word, not_selected_word
        return "", ""

    def generate_parts(self, state1, state2, action1, action2, obj1, obj2, combination_type):
        state_word1, state_hallu1 = self.select_random_word(state1, combination_type)
        state_word2, state_hallu2 = self.select_random_word(state2, combination_type)
        action_word1, action_hallu1 = self.select_random_word(action1, combination_type)
        action_word2, action_hallu2 = self.select_random_word(action2, combination_type)
        
        part1 = f"a {state_word1} {action_word1} {obj1}".strip()
        part2 = f"a {state_word2} {action_word2} {obj2}".strip()
        
        part1 = ' '.join(filter(bool, part1.split()))
        part2 = ' '.join(filter(bool, part2.split()))
        
        return part1, part2, (state_word1, state_word2, action_word1, action_word2), (state_hallu1, state_hallu2, action_hallu1, action_hallu2)

    def generate_prompt(self, img_dir, entities, prompt_type, combination_type):
        generated_combinations = set()  

        for item in entities:
            obj1, obj2 = item[0], item[1]
            state1, state2 = item[2], item[3]
            action1, action2 = item[4], item[5]

            for _ in range(2): 
                part1, part2, truth_attrs, hallucinated_attrs = self.generate_parts(state1, state2, action1, action2, obj1, obj2, combination_type)

                combination_key = (part1, part2)
                if combination_key in generated_combinations:
                    print(combination_type, combination_key)
                    continue  
         
                    
                generated_combinations.add(combination_key)

                if prompt_type == 'ev':
                    self.data_template['prompt'] = f'{part1} in {part2}'
                else:
                    self.data_template['prompt'] = f'{part1} and {part2}'

                data_info = {
                    "prompt": self.data_template['prompt'],
                    "info_type": prompt_type,
                    "truth": [obj1, obj2],
                    "attr_truth": list(truth_attrs),
                    "attr_hallu": list(hallucinated_attrs)  
                }

                for idx in range(6):
                    part1_clean = part1.replace(" ", "_")
                    part2_clean = part2.replace(" ", "_")
                    img_name = f'{part1_clean}_{part2_clean}_{prompt_type}{idx + 1}.png'
                    save_image_path = os.path.join(img_dir, img_name)

               
                    self.data_template.update({
                        'seed': self.seeds[idx],
                        'steps': self.steps[idx % 2],
                        'negative_prompt': self.negative_prompts[prompt_type][idx % 2],
                        'cfg_scale': self.cfg_scales[idx % 2]
                    })
                    self.update_data_info(save_image_path, data_info)
                    self.data_template['prompt'] += self.extra_prompts[prompt_type][idx % 2]
                    self.generate_image(self.data_template, save_image_path)

                

    def run(self):
        img_dir = self.output_dir
        os.makedirs(img_dir, exist_ok=True)

        combinations = {
            'Fictional': object_combination_Fictional,
            'Standard': object_combination_Standard,
            'Longtailed': object_combination_Longtailed,
            'Random': object_combination_Random
        }

        path = "data/objects_cooccurrence_graph_T.graphml"
       
        for group_type, func in combinations.items():
            img_dir = os.path.join(self.output_dir, group_type)
            os.makedirs(img_dir, exist_ok=True)

            entities, entities_env = func( path, limit=40)
            self.generate_prompt(img_dir, entities_env, 'ev', group_type)
            self.generate_prompt(img_dir, entities, 'ee', group_type)

        print("Data creation complete.")

def extract_objects(info, info_type):
    if info_type == "ev":
        match = re.match(r'a (.+) in (.+)', info)
        if match:
            return match.groups()
    elif info_type == "ee":
        match = re.match(r'a (.+) and a (.+)', info)
        if match:
            return match.groups()
    return ()

def process_json_file(input_file, output_dir):
    with open(input_file, 'r') as file:
        data = json.load(file)

    type_to_objects = defaultdict(set)

    for entry in data:
        objects = extract_objects(entry['info'], entry['info_type'])
        type_to_objects[entry['type']].update(objects)

    for type_, objects in type_to_objects.items():
        with open(f"{output_dir}/{type_}.txt", 'w') as file:
            for obj in sorted(objects):
                file.write(f"{obj}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ImageGenerator and process data.")
    parser.add_argument("--data_id", type=int, required=True, help="Starting data ID")
    parser.add_argument("--data_info_path", type=str, required=True, help="Path to the annotations JSON file")
    parser.add_argument("--api_url", type=str, required=True, help="API URL for image generation")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    
    args = parser.parse_args()
    generator = ImageGenerator(
        data_id=args.data_id,
        data_info_path=args.data_info_path,
        api_url=args.api_url,
        output_dir=args.output_dir
    )
    # 运行生成器
    generator.run()
    
    output_obj_dir = f'{args.output_dir}/objects.txt' 
    process_json_file(args.data_info_path, output_obj_dir) # All meta concepts contained in a certain type of image
