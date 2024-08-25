from transformers import CLIPTokenizer, CLIPModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)

# Load CLIP and tokenizer

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
text_temp=["A photo of a"]


# merge sim label
def merge_similar_values(values, similarity_threshold=0.7):
    # tokenizer, model, device = load_clip_model()
    merged_values = []
    value_to_group = {}

    # 对每个值进行编码
    value_input = [f"{text_temp[0]} {str(lbl)}" for lbl in values]
    value_inputs = tokenizer(value_input, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        value_embeddings = model.get_text_features(**value_inputs).cpu().numpy()

    merged_flags = [False] * len(values)

    for i in range(len(values)):
        if merged_flags[i]:
            continue
        current_group = [values[i]]
        for j in range(i + 1, len(values)):
            if merged_flags[j]:
                continue
            similarity = cosine_similarity([value_embeddings[i]], [value_embeddings[j]])[0][0]
            # print('s',similarity)
            if similarity >= similarity_threshold:
                current_group.append(values[j])
                merged_flags[j] = True
        merged_values.append(tuple(current_group))
        for v in current_group:
            value_to_group[v] = current_group

    return merged_values, value_to_group

# save the best match values
def select_best_value_for_key(key, merged_values):

    # encode key
    key_input = [f"{text_temp[0]} {str(key)}"]
    key_inputs = tokenizer(key_input, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        key_embedding = model.get_text_features(**key_inputs).cpu().numpy()

    best_values = []

    #
    for value_group in merged_values:
        value_inputs = tokenizer(list(value_group), return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            value_embeddings = model.get_text_features(**value_inputs).cpu().numpy()

        similarities = cosine_similarity(key_embedding, value_embeddings)[0]
        max_similarity_index = similarities.argmax()
        best_value = value_group[max_similarity_index]
        best_values.append(best_value)

    return best_values

def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def compare_json_keys_values(json1, json2):
    overlapping_keys_values = defaultdict(list)

    keys1 = set(json1.keys())
    keys2 = set(json2.keys())

    common_keys = keys1.intersection(keys2)

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

json1_path='/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/user_wiki.json'
data1 = read_json(json1_path)
result = {}

for key in tqdm(data1.keys(), desc="Merge Values"):
    merged_values, value_to_group = merge_similar_values(data1[key], similarity_threshold=0.1)
    best_values = select_best_value_for_key(key, merged_values)
    result[key] = best_values

json_out = '/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/user_wiki_sim_01.json'
with open(json_out, 'w') as json_file:
    json.dump(result, json_file, indent=4)

json2_path='/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/wiki_user.json'
data2 = read_json(json2_path)
result2 = {}

for key in tqdm(data2.keys(), desc="Merge Values"):
    # if id>500:
    #     print(key)
    merged_values, value_to_group = merge_similar_values(data2[key], similarity_threshold=0.6)
    best_values = select_best_value_for_key(key, merged_values)
    result2[key] = best_values

json_out2 = '/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/wiki_user_sim_06.json'
with open(json_out2, 'w') as json_file:
    json.dump(result2, json_file, indent=4)

# swape_res=swape_json(result)
# out_result=compare_json_keys_values(swape_res,result2)
#
# json_out3 = '/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/wiki_user_same_05_09.json'
# with open(json_out3, 'w') as json_file:
#     json.dump(out_result, json_file, indent=4)







