import itertools
import json
import os.path
from collections import defaultdict
import requests


def get_wikidata_qcode(entity_name, language='en'):

    search_url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'search': entity_name,
        'language': language,
        'format': 'json'
    }

    response = requests.get(search_url, params=params)

    if response.status_code == 200:
        search_results = response.json().get('search', [])
        if search_results:
            # return first match id
            return search_results[0].get('id')
        else:
            print(f"Can not find '{entity_name}' match。")
            return None
    else:
        print(f"request fail：{response.status_code}")
        return None

def extract_subclass_relationships(data,outpath=None):
    # Flatten all values into a single list
    class_all=[]
    data_all=[]
    all_values = list(itertools.chain(*data.values()))

    # Remove duplicates
    for key in data.keys():
        print(key)
        key_qcode = get_wikidata_qcode(key)
        if key_qcode is not None:
            class_all.append(key_qcode)
    if outpath is not None:
        with open(os.path.join(outpath,'classes.txt'), "w") as file:
            for item in class_all:
                file.write(f"{item}\n")

    unique_values = list(set(all_values))
    for value in unique_values:
        value_qcode=get_wikidata_qcode(value)
        if value_qcode is not None:
            data_all.append(value_qcode + ' rdfs:label ' +value)

    # Generate subclass relationships
    for key, values in data.items():

        key_qcode = get_wikidata_qcode(key)
        if key_qcode is not None:
            for value in values:
                value_qcode=get_wikidata_qcode(value)
                if value_qcode is not None:
                    data_all.append((value_qcode + " subclassof "+ key_qcode))
    if outpath is not None:
        with open(os.path.join(outpath,'entitles.txt'), "w") as file:
            for item in data_all:
                file.write(f"{item}\n")



def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def compare_json_keys_values(json1, json2):
    overlapping_keys_values = defaultdict(list)

    # get all key
    keys1 = set(json1.keys())
    keys2 = set(json2.keys())

    # find common keys
    common_keys = keys1.intersection(keys2)

    # get overlap values
    for key in common_keys:
        set1 = set(json1[key])
        set2 = set(json2[key])
        overlap = set1.intersection(set2)
        if overlap:
            overlapping_keys_values[key] = list(overlap)

    return overlapping_keys_values

def swape_json(json_data):
    swapped_data = defaultdict(list)

    for key, values in json_data.items():
        for value in values:
            swapped_data[value].append(key)
    return swapped_data

# def output_final(json_data=None):
#     mother_class=

json1_path='/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/user_wiki_sim_06.json'
data1 = read_json(json1_path)
swapped_data = swape_json(data1)
json2_path='/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/wiki_user_sim_06.json'
data2 = read_json(json2_path)

out_result=compare_json_keys_values(swapped_data,data2)

json_out3 = '/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/wiki_user_same_09_09.json'
with open(json_out3, 'w') as json_file:
    json.dump(out_result, json_file, indent=4)

out_path='/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output'
extract_subclass_relationships(out_result,out_path)

