import pandas as pd
import json
from collections import defaultdict
import copy

clean_path='/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/output_all.csv'
df_all=pd.read_csv(clean_path)

result = {}
current_children = []


def remove_duplicates_from_json(json_data):
    for key in json_data:
        if isinstance(json_data[key], list):
            seen = set()
            unique_values = []
            for value in json_data[key]:
                if value not in seen:
                    unique_values.append(value)
                    seen.add(value)
            json_data[key] = unique_values
    return json_data

# iter DataFrame
for index, row in df_all.iterrows():
    if row['source'] == 'user':
        current_children.append(row['label'])

    elif row['source'] == 'wikimedia':
        if row['label'] not in result:
            result[row['label']] = current_children
        else:
            result[row['label']].extend(current_children)
        current_children = []

result_copy = copy.deepcopy(result)
result_out = remove_duplicates_from_json(result_copy)

# save as json
with open('/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/wiki_user.json', 'w') as json_file:
    json.dump(result_out, json_file, indent=4)

print("JSON文件已生成并保存为 grouped_data.json")

swapped_data = defaultdict(list)

# switch key and values
for key, values in result.items():
    for value in values:
        swapped_data[value].append(key)


# switch defaultdict to dict
swapped_data = dict(swapped_data)
swapped_data_out=remove_duplicates_from_json(swapped_data)

# save as json
with open('/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/user_wiki.json', 'w') as json_file:
    json.dump(swapped_data_out, json_file, indent=4)

