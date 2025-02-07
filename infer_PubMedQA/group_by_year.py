import json
from collections import defaultdict

def group_by_year(data):
    grouped_data = defaultdict(list)
    
    for pubmedid, entry in data.items():
        year = entry.get('YEAR')
        if year:
            grouped_data[year].append({pubmedid: entry})
    
    return grouped_data


with open('./data/ori_pqal.json', 'r') as file:
    data = json.load(file)

grouped_data = group_by_year(data)


for year, entries in grouped_data.items():
    filename = f'./PubMedQA_group/by_year/grouped_by_year_pubmedqa_data_{year}.json'
    with open(filename, 'w') as outfile:
        json.dump(entries, outfile, indent=4)


print(json.dumps(grouped_data, indent=4))
