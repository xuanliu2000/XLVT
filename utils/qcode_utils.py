import requests
import concurrent.futures
from tqdm import tqdm

def read_qcodes_from_file(file_path):
    with open(file_path, 'r') as file:
        qcodes = [line.strip() for line in file if line.strip()]
    return qcodes

def fetch_label(qcode, language='en'):
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbgetentities',
        'ids': qcode,
        'format': 'json',
        'props': 'labels',
        'languages': language
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        entity = data.get('entities', {}).get(qcode, {})
        label = entity.get('labels', {}).get(language, {}).get('value', None)
        return label
    except requests.exceptions.RequestException as e:
        print(f"Network error while fetching {qcode}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {qcode}: {e}")
        return None

def fetch_labels(qcodes, language='en', max_workers=10):

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_qcode = {executor.submit(fetch_label, qcode, language): qcode for qcode in qcodes}
        for future in tqdm(concurrent.futures.as_completed(future_to_qcode), total=len(qcodes), desc="Fetching labels"):
            qcode = future_to_qcode[future]
            label = future.result()
            results[qcode] = label
    return results

def write_labels_to_file(labels_dict, output_file):

    with open(output_file, 'w') as file:
        for qcode, label in labels_dict.items():
            file.write(f"{qcode}\t{label if label else 'Label not found'}\n")

if __name__ == "__main__":
    input_file = '/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/submission/classes.txt'
    output_file = '/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/submission/classes_out.txt'
    language = 'en'
    max_workers = 10

    # 读取Qcodes
    qcodes = read_qcodes_from_file(input_file)
    print(f"Total Qcodes to process: {len(qcodes)}")

    # 获取标签
    labels_dict = fetch_labels(qcodes, language=language, max_workers=max_workers)

    # 写入结果
    write_labels_to_file(labels_dict, output_file)
    print(f"Labels have been written to {output_file}")
