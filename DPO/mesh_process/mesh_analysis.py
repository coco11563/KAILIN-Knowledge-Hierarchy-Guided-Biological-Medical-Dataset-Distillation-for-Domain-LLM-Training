import argparse
from tqdm.auto import tqdm
import json
from treelib import Tree, Node

# 1. build a mesh_dict for refer
def build_mesh_dict(mesh_desc_path):
    mesh_dict = dict()
    with open(mesh_desc_path, 'r') as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            ui = data['DescriptorUI']
            tree_list = data['TreeNumberList']
            mesh_dict[ui] = tree_list
    return mesh_dict

# 2. define a function to find certain node in a tree
def find_node_in_tree(tree, target_id):
    for node in tree.all_nodes_itr():
        if node.identifier == target_id:
            node.data['count'] += 1
            return node
    return None

# 3. build a tree via treelib through the mesh_dict
def build_tree_and_save(mesh_dict, output_path):
    Mesh_Tree = Tree() # Tree

    Mesh_Tree.create_node('Root', 'root', data={'count': 0})

    for key in tqdm(mesh_dict.keys()):
        tree_list = mesh_dict[key]
        for tree_ in tree_list:
            node_list = tree_.split('.')
            this_node = ""
            father_node = "root"
            for node in node_list:
                this_node += ('.' + node)
                target_node = find_node_in_tree(Mesh_Tree, this_node)
                if not target_node:
                    Mesh_Tree.create_node(this_node, this_node, parent=father_node, data={'count': 0})
                    father_node = this_node
                else:
                    father_node = this_node

    with open(output_path, 'w') as f:
        tree_json = Mesh_Tree.to_json(with_data=True)
        f.write(tree_json)

    return Mesh_Tree

def load_tree(json_path):
    tree = Tree()
    with open(json_path, 'r') as f:
        tree_json = f.read()
        tree.load(tree_json, with_data=True)

    return tree

# 4. traversal all the processed pubmed data for mesh
def traversal_and_count(Mesh_Tree, pubmed_path, mesh_dict):
    with open(pubmed_path, 'r') as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            Mesh_list = data['metadata']['MeshHeading']
            mesh_list = []
            for item in Mesh_list:
                mesh_list.append(item['UI'])

            for mesh in mesh_list:
                node_list = mesh.split('.')
                this_node = ""
                for node in node_list:
                    this_node += ('.' + node)
                    target_node = find_node_in_tree(Mesh_Tree, this_node)
                    if target_node is not None:
                        target_node.data['count'] += 1

    return Mesh_Tree

# 5. main
def main(mesh_desc_path, pubmed_path, tree_output_path, tree_sampled_output_path):
    mesh_dict = build_mesh_dict(mesh_desc_path)

    Mesh_tree = build_tree_and_save(mesh_dict, tree_output_path)

    result_tree = traversal_and_count(Mesh_Tree=Mesh_tree, 
                                      pubmed_path=pubmed_path,
                                      mesh_dict=mesh_dict)
    
    with open(tree_sampled_output_path, 'w') as f:
        tree_json = result_tree.to_json(with_data=True)
        f.write(tree_json)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--mesh_desc_path', type=str, required=True, help='Path to the mesh description file')
    parser.add_argument('--pubmed_path', type=str, required=True, help='Path to the pubmed data file')
    parser.add_argument('--tree_output_path', type=str, required=True, help='Path to save the output tree')
    parser.add_argument('--tree_sampled_output_path', type=str, required=True, help='Path to save the sampled output tree')

    args = parser.parse_args()
    
    main(args.mesh_desc_path, args.pubmed_path, args.tree_output_path, args.tree_sampled_output_path)
