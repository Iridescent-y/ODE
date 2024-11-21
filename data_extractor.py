import json
import argparse
from collections import defaultdict
import networkx as nx
import re
import base64

class DataExtractor:
    def __init__(self, dataset_path, annotations_path=None, object_properties_path=None, attribute_path=None):
        self.dataset_path = dataset_path
        self.annotations_path = annotations_path
        self.properties_path = object_properties_path
        self.query_path = attribute_path

    def create_graph(self):
        with open(self.annotations_path, "r", encoding='utf-8') as f:
            annotations = json.load(f)

        G_truth = nx.Graph()
        G_hallu = nx.Graph()

        for entry in annotations:
            if entry['type'] == 'generative':
                truth_objects = entry.get("truth", [])
                hallu_objects = entry.get("hallu", [])

                self._add_edges(G_truth, truth_objects)
                self._add_correspondence_edges(G_hallu, truth_objects, hallu_objects)
                
                G_truth = self._add_property(G_truth)
                G_hallu = self._add_property(G_hallu)
            else:
                continue
        
        G_truth = self._create_attribute_graph(G_truth)
        G_hallu = self._create_attribute_graph(G_hallu)
        
        self._convert_attributes_to_string(G_truth)
        self._convert_attributes_to_string(G_hallu)
        
        nx.write_graphml(G_truth, f"{self.dataset_path}/objects_cooccurrence_graph_T.graphml")
        nx.write_graphml(G_hallu, f"{self.dataset_path}/objects_cooccurrence_graph_H.graphml")

        return G_truth, G_hallu

    def _add_edges(self, graph, objects, exclude=None):
        """
        Add edges between co-occurring objects in the graph.
        """
        if exclude is None:
            exclude = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1, obj2 = sorted([objects[i], objects[j]])
            
                if obj1 in exclude or obj2 in exclude:
                    continue
                if graph.has_edge(obj1, obj2):
                    graph[obj1][obj2]["frequency"] += 1
                    graph[obj2][obj1]["frequency"] += 1
                else:
                    graph.add_edge(obj1, obj2, frequency=1)

    def _add_correspondence_edges(self, graph, truth_objects, hallu_objects):
        """
        Add edges between truth and hallucination objects.
        """
        for truth_obj in truth_objects:
            for hallu_obj in hallu_objects:
                if truth_obj and hallu_obj: 
                    if graph.has_edge(truth_obj, hallu_obj):
                        graph[truth_obj][hallu_obj]["frequency"] += 1
                    else:
                        graph.add_edge(truth_obj, hallu_obj, frequency=1)
                    
    def _add_property(self, graph):
        """
        Add properties to graph nodes based on external data.  (entity or environment)
        """
        with open(self.properties_path, "r", encoding='utf-8') as f:
            node_properties = json.load(f)
        
        nodes = list(graph.nodes())
        for node in nodes:
            property_name = node_properties.get(node, {}).get("property")
            if property_name:
                graph.nodes[node]["type"] = property_name
            graph.nodes[node]["state"] = []
            graph.nodes[node]["action"] = []
            
        return graph
    
    def _create_attribute_graph(self, graph):
        """
        Add state and action attributes to graph nodes based on queries.
        """
        with open(self.query_path, "r", encoding='utf-8') as f:
            queries_attribute = json.load(f)

        nodes = list(graph.nodes())

        for i in range(0, len(queries_attribute) - 1, 2):
            attribute_type = "state"
            attribute_query = r"Is the (\w+) (\w+) in this image"
            match1 = re.findall(attribute_query, queries_attribute[i]['query'])
            match2 = re.findall(attribute_query, queries_attribute[i + 1]['query'])

            if len(match1) == 0:
                attribute_type = "action"
                attribute_query = r"Does the (\w+) (\w+) in this image"
                match1 = re.findall(attribute_query, queries_attribute[i]['query'])
                match2 = re.findall(attribute_query, queries_attribute[i + 1]['query'])
            
            if match1 and match2:
                noun1, attribute1 = match1[0]
                noun2, attribute2 = match2[0]

                if noun1 == noun2 and noun1 in nodes:
                    combined_attributes = tuple([attribute1, attribute2])

                    if combined_attributes not in graph.nodes[noun1][attribute_type]:
                        graph.nodes[noun1][attribute_type].append(combined_attributes)
  
        return graph
            
    def _convert_attributes_to_string(self, graph):
        for node, data in graph.nodes(data=True):
            if isinstance(data.get('state'), list):
                encoded_state = json.dumps(data['state'], ensure_ascii=False).encode('utf-8')
                graph.nodes[node]['state'] = base64.b64encode(encoded_state).decode('utf-8')
            if isinstance(data.get('action'), list):
                encoded_action = json.dumps(data['action'], ensure_ascii=False).encode('utf-8')
                graph.nodes[node]['action'] = base64.b64encode(encoded_action).decode('utf-8')

    def convert_attributes_from_string(self, graph):
        for node, data in graph.nodes(data=True):
            if isinstance(data.get('state'), str):
                try:
                    decoded_state = base64.b64decode(data['state']).decode('utf-8')
                    graph.nodes[node]['state'] = json.loads(decoded_state)
                except Exception:
                    graph.nodes[node]['state'] = data['state']
            if isinstance(data.get('action'), str):
                try:
                    decoded_action = base64.b64decode(data['action']).decode('utf-8')
                    graph.nodes[node]['action'] = json.loads(decoded_action)
                except Exception:
                    graph.nodes[node]['action'] = data['action']
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate co-occurrence graphs from a dataset.")
    parser.add_argument("--dataset_path", type=str, default="./data", help="Path to the dataset directory.")
    parser.add_argument("--annotations_path", type=str, default="./data/BaseInfo/amber_annotations.json", help="Path to the annotations JSON file.")
    parser.add_argument("--properties_path", type=str, default="./data/BaseInfo/object-properties.json", help="Path to the object properties JSON file.")
    parser.add_argument("--attribute_path", type=str, default="./data/BaseInfo/attribute.json", help="Path to the attribute query JSON file.")
    
    args = parser.parse_args()
    
    # Initialize DataExtractor
    extractor = DataExtractor(
        dataset_path=args.dataset_path,
        annotations_path=args.annotations_path,
        object_properties_path=args.properties_path,
        attribute_path=args.attribute_path,
    )
    # Create graphs
    G_truth, G_hallu = extractor.create_graph()
   
    