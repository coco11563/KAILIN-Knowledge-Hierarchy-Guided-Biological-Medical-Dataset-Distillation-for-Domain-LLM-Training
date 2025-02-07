import json
import requests
from collections import defaultdict
from xml.etree import ElementTree as ET

def fetch_mesh_terms(pubmed_ids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    fetch_url = base_url + "efetch.fcgi"
    params = {
        'db': 'pubmed',
        'retmode': 'xml',
        'id': ','.join(pubmed_ids)
    }
    response = requests.get(fetch_url, params=params)
    response.raise_for_status()
    return response.text

def parse_mesh_terms(xml_data):
    root = ET.fromstring(xml_data)
    mesh_terms = defaultdict(list)
    for article in root.findall(".//PubmedArticle"):
        pubmed_id = article.find(".//PMID").text
        for mesh_heading in article.findall(".//MeshHeading"):
            descriptor = mesh_heading.find(".//DescriptorName").text
            mesh_terms[pubmed_id].append(descriptor)
    return mesh_terms


with open('./data/ori_pqal.json', 'r') as f:
    data = json.load(f)


pubmed_ids = list(data.keys())


batch_size = 200  
all_mesh_terms = defaultdict(list)

for i in range(0, len(pubmed_ids), batch_size):
    batch_ids = pubmed_ids[i:i+batch_size]
    xml_data = fetch_mesh_terms(batch_ids)
    batch_mesh_terms = parse_mesh_terms(xml_data)
    all_mesh_terms.update(batch_mesh_terms)


for pubmed_id in data.keys():
    data[pubmed_id]['mesh_terms'] = all_mesh_terms.get(pubmed_id, [])


with open('pqal_with_mesh.json', 'w') as f:
    json.dump(data, f, indent=2)

print("MeSH terms added and saved to pqal_with_mesh.json")
