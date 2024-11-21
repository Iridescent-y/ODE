import json
from collections import defaultdict
import networkx as nx
import random
import re
from data_extractor import DataExtractor 
total_limit = 20

def select_attribute(attribute_list):
    if isinstance(attribute_list, list) and attribute_list:
        index = random.randint(0, len(attribute_list) - 1)
        item = attribute_list[index]
        return tuple(item)
    
    return ""


def object_combination_Standard(graph_path, limit):
    G = nx.read_graphml(graph_path)
    extractor = DataExtractor(dataset_path="./data")
    extractor.convert_attributes_from_string(G)

    return _get_sorted_combinations(G, limit, reverse=True) 


def _get_sorted_combinations(G, limit=20, reverse=True):
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]["frequency"], reverse=reverse)
    combinations_entities = set()
    combinations_ent_env = set()
    for edge in sorted_edges:
        node1, node2 = edge[0], edge[1]
        
        proper0 = G.nodes[node1].get("type", "")
        proper1 = G.nodes[node2].get("type", "")
        
        state1 = G.nodes[node1].get("state", [])
        state2 = G.nodes[node2].get("state", [])
        action1 = G.nodes[node1].get("action", [])
        action2 = G.nodes[node2].get("action", [])
        
        if proper0 == "entity" and proper1 == "entity" and len(combinations_entities) < limit:
            combinations_entities.add((
                node1,
                node2,
                tuple(tuple(x) for x in state1),
                tuple(tuple(x) for x in state2),
                tuple(tuple(x) for x in action1),
                tuple(tuple(x) for x in action2)
            ))
        
        if ((proper0 == "entity" and proper1 == "environment") or (proper0 == "environment" and proper1 == "entity")) and len(combinations_ent_env) < limit:
            if proper0 == "environment":
                node1, node2 = node2, node1
                state1, state2 = state2, state1
                action1, action2 = action2, action1
            combinations_ent_env.add((
                node1,
                node2,
                tuple(tuple(x) for x in state1),
                tuple(tuple(x) for x in state2),
                tuple(tuple(x) for x in action1),
                tuple(tuple(x) for x in action2)
            ))

        if len(combinations_entities) == limit and len(combinations_ent_env) == limit:
            break
        
    return list(combinations_entities), list(combinations_ent_env)

def _get_random_combinations(G, limit=20):
    random.seed(42)
    nodes = list(G.nodes())
    selected_pairs_entities = set()
    selected_pairs_ent_env = set()
    while len(selected_pairs_entities) < limit or len(selected_pairs_ent_env) < limit:
        node1, node2 = random.sample(nodes, 2)
        
        proper0 = G.nodes[node1].get("type", "")
        proper1 = G.nodes[node2].get("type", "")
        
        state1 = G.nodes[node1].get("state", [])
        state2 = G.nodes[node2].get("state", [])
        action1 = G.nodes[node1].get("action", [])
        action2 = G.nodes[node2].get("action", [])
        
        if proper0 == "entity" and proper1 == "entity" and len(selected_pairs_entities) < limit:
            selected_pairs_entities.add((
                node1,
                node2,
                tuple(tuple(x) for x in state1),
                tuple(tuple(x) for x in state2),
                tuple(tuple(x) for x in action1),
                tuple(tuple(x) for x in action2)
            ))
        elif not (proper0 == "environment" and proper1 == "environment") and len(selected_pairs_ent_env) < limit:
            if proper0 == "environment":
                node1, node2 = node2, node1
                state1, state2 = state2, state1
                action1, action2 = action2, action1
            selected_pairs_ent_env.add((
                node1,
                node2,
                tuple(tuple(x) for x in state1),
                tuple(tuple(x) for x in state2),
                tuple(tuple(x) for x in action1),
                tuple(tuple(x) for x in action2)
            ))

    return list(selected_pairs_entities), list(selected_pairs_ent_env)


def object_combination_Random(graph_path, limit):
    G = nx.read_graphml(graph_path)
    extractor = DataExtractor(dataset_path="./data")
    extractor.convert_attributes_from_string(G)
    
    return _get_random_combinations(G, limit)


def object_combination_Fictional(graph_path, limit):
    random.seed(42)
    G = nx.read_graphml(graph_path)
    extractor = DataExtractor(dataset_path="./data")
    extractor.convert_attributes_from_string(G)
    nodes = list(G.nodes())

    selected_pairs_entities = set()
    selected_pairs_ent_env = set()
    
    all_node_pairs = [(node1, node2) for i, node1 in enumerate(nodes) for node2 in nodes[i+1:]]
    non_edge_pairs = [pair for pair in all_node_pairs if not G.has_edge(*pair)]

    while len(selected_pairs_entities) < limit or len(selected_pairs_ent_env) < limit and non_edge_pairs:
        node1, node2 = random.choice(non_edge_pairs)
        non_edge_pairs.remove((node1, node2))
        
        proper0 = G.nodes[node1].get("type", "")
        proper1 = G.nodes[node2].get("type", "")
        
        if not G.has_edge(node1, node2):
            state1 = G.nodes[node1].get("state", [])
            state2 = G.nodes[node2].get("state", [])
            action1 = G.nodes[node1].get("action", [])
            action2 = G.nodes[node2].get("action", [])
            
            if proper0 == "entity" and proper1 == "entity" and len(selected_pairs_entities) < limit:
                selected_pairs_entities.add((
                    node1,
                    node2,
                    tuple(tuple(x) for x in state1),
                    tuple(tuple(x) for x in state2),
                    tuple(tuple(x) for x in action1),
                    tuple(tuple(x) for x in action2)
                ))
            elif not (proper0 == "environment" and proper1 == "environment") and len(selected_pairs_ent_env) < 20:
                if proper0 == "environment":
                    node1, node2 = node2, node1
                    state1, state2 = state2, state1
                    action1, action2 = action2, action1
                selected_pairs_ent_env.add((
                    node1,
                    node2,
                    tuple(tuple(x) for x in state1),
                    tuple(tuple(x) for x in state2),
                    tuple(tuple(x) for x in action1),
                    tuple(tuple(x) for x in action2)
                ))

    return list(selected_pairs_entities), list(selected_pairs_ent_env)

def object_combination_Longtailed(graph_path, limit):
    G = nx.read_graphml(graph_path)
    extractor = DataExtractor(dataset_path="./data")
    extractor.convert_attributes_from_string(G)

    return _get_sorted_combinations(G, limit, reverse=False)

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

def add_hallu():
    graph = nx.read_gml('data/objects_cooccurrence_graph_H.gml')

    nodes = set(graph.nodes())

    existing_objects = set()
    
    types = ["Random", "Fictional", "Confusible", "Longtailed", "Standard"]
    for type in types:
        file_name = f'imgs/{type}.txt'
        with open(file_name, 'r') as file:
            for line in file:
                obj = line.strip()
                if obj: 
                    existing_objects.add(obj)
                
        with open(file_name, 'a') as file:
            for node in nodes:
                if node not in existing_objects:
                    file.write(node + '\n')
def object():
    input_file = 'annotations.json'  
    output_dir = 'imgs'  

    process_json_file(input_file, output_dir)

if __name__ == "__main__":
    graph_path = "data/objects_cooccurrence_graph_T.graphml"  # or H for hallucinated graph
    limit = 50
    Standard_entities, Standard_ent_env = object_combination_Standard(graph_path, limit)
    # print(Standard_entities)
    # print("Standard Entities Combinations:", len(Standard_entities))
    # print("Standard Entity-Environment Combinations:", len(Standard_ent_env))

    random_entities, random_ent_env = object_combination_Random(graph_path, limit)
    # print("Random Entities Combinations:", len(random_entities))
    # print("Random Entity-Environment Combinations:", len(random_ent_env))

    nonexistent_entities, nonexistent_ent_env = object_combination_Fictional(graph_path, limit)
    # print("Fictional Entities Combinations:", len(nonexistent_entities))
    # print("Fictional Entity-Environment Combinations:", len(nonexistent_ent_env))

    longtailed_entities, longtailed_ent_env = object_combination_Longtailed(graph_path, limit)
    # print("longtailed Entities Combinations:", len(longtailed_entities))
    # print("longtailed Entity-Environment Combinations:", len(longtailed_ent_env))
    
    # print(longtailed_ent_env)

    
    # # # object()
    # # # add_hallu()
