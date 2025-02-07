import json
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
import argparse

def get_DescriptorUI(mesh):
    """
    Extracts the DescriptorUI from a mesh element.

    Args:
        mesh (xml.etree.ElementTree.Element): The mesh element.

    Returns:
        str: The DescriptorUI string.
    """
    ui = ""
    for elem in mesh.iter('DescriptorUI'):
        ui = elem.text
        break
    return ui.strip()

def get_DescriptorName(mesh):
    """
    Extracts the DescriptorName from a mesh element.

    Args:
        mesh (xml.etree.ElementTree.Element): The mesh element.

    Returns:
        str: The DescriptorName string.
    """
    name = ""
    for elem in mesh.iter('DescriptorName'):
        for elem_ in elem.iter('String'):
            name += (elem_.text)
        break
    return name.strip()

def get_tree(mesh):
    """
    Extracts the TreeNumberList from a mesh element.

    Args:
        mesh (xml.etree.ElementTree.Element): The mesh element.

    Returns:
        list: A list of TreeNumber strings.
    """
    tree = []
    for elem in mesh.iter('TreeNumber'):
        tree.append(elem.text)
    return tree

def process(xml_file_name, out_file_name):
    """
    Processes an XML file to extract mesh descriptor information and writes it to a JSONL file.

    Args:
        xml_file_name (str): Path to the input XML file.
        out_file_name (str): Path to the output JSONL file.
    """
    writer = open(out_file_name, 'w')
    root = ET.parse(xml_file_name)
    mesh_list = root.findall('DescriptorRecord')

    for mesh in tqdm(mesh_list):
        ui = get_DescriptorUI(mesh)
        name = get_DescriptorName(mesh)
        tree = get_tree(mesh)

        out_data = {"DescriptorUI": ui, "DescriptorName": name, "TreeNumberList": tree}

        if ui == "" or name == "" or tree == []:
            continue 
        else:
            writer.write(json.dumps(out_data, ensure_ascii=False) + '\n')

    writer.close()

def main():
    """
    Main function to parse command-line arguments and process the XML file.
    """
    parser = argparse.ArgumentParser(description='Process XML to JSONL.')
    parser.add_argument('--xml_file', type=str, required=True, help='Path to the input XML file')
    parser.add_argument('--out_file', type=str, required=True, help='Path to the output JSONL file')

    args = parser.parse_args()

    process(args.xml_file, args.out_file)

if __name__ == '__main__':
    main()
