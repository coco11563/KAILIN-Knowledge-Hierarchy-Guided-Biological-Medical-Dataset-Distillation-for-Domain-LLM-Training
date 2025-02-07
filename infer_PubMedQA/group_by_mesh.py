import json
from collections import Counter
import os


with open('./data/ori_pqal.json', 'r') as f:
    data = json.load(f)


mesh_counter = Counter()
for key, value in data.items():
    meshes = value.get('MESHES', [])
    mesh_counter.update(meshes)


top_10_meshes = mesh_counter.most_common(10)


output_dir = 'by_mesh'
os.makedirs(output_dir, exist_ok=True)


mesh_records = {mesh: [] for mesh, _ in top_10_meshes}


for key, value in data.items():
    meshes = value.get('MESHES', [])
    for mesh in meshes:
        if mesh in mesh_records:
            mesh_records[mesh].append({key: value})
            break  


for mesh, records in mesh_records.items():
    with open(os.path.join(output_dir, f'{mesh}.json'), 'w') as f:
        json.dump(records, f, indent=2)



for mesh, count in top_10_meshes:
    print(f"{mesh}: {count} terms")


