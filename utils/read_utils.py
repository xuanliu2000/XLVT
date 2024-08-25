import os
import pandas as pd
import torch

data_path = '/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/data'  # Path for data
img_extensions = ['.JPG', '.PNG', '.gif', '.jpeg', '.jpg', '.png', '.svg', 'webp']


# get a csv name and get matched image with different extensions
def match_img(csv_name, path=data_path):
    out_image_path = None
    out_csv_path = None
    for f in os.listdir(path):
        if csv_name in f and not f.endswith('.csv'):
            out_image_path = os.path.join(path, f)
            out_csv_path = os.path.join(path, csv_name + '.csv')
    return out_image_path, out_csv_path


def get_sorted_name(path=data_path):
    # get file id
    file_names = [os.path.splitext(f)[0] for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    unique_sorted_names = sorted(set(file_names))
    # print(unique_sorted_names)
    return unique_sorted_names


def load_csv(csv_name, path=data_path):
    csv_path = os.path.join(path, csv_name + '.csv')
    csv_pd = pd.read_csv(csv_path)
    return csv_pd


class CSVFilter:
    def __init__(self, folder_path):
        """
        :param folder_path: csv path
        """
        self.folder_path = folder_path
        self.unique_values = set()
        self.df_column = ['label', 'X coordinate', 'Y coordinate', 'height', 'width', 'source']

    def filter_csv_files(self, source_value, label_column_name='label'):
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.folder_path, filename)
                df = pd.read_csv(file_path)
                df.columns = self.df_column

                if 'source' in df.columns and label_column_name in df.columns:
                    filtered_values = df[df['source'] == source_value][label_column_name]
                    self.unique_values.update(filtered_values)

        return list(self.unique_values)

    def clear_results(self):
        self.unique_values.clear()

    def load_single_csv(self, csv_name, path=data_path):
        csv_path = os.path.join(path, csv_name + '.csv')
        csv_pd = pd.read_csv(csv_path)
        csv_pd.columns = self.df_column
        return csv_pd


def crop_transform(box=None,
                   image=None):
    if box is None:
        box = [0, 0, 0, 0]
    if image is None:
        raise ValueError("Image cannot be None.")

    x_coordinate, y_coordinate, height, width = box[0], box[1], box[2], box[3]
    image_width, image_height = image.size

    # cropped_image limitation
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be greater than 0.")
    if x_coordinate < 0 or y_coordinate < 0:
        raise ValueError("X and Y coordinates must be non-negative.")
    if x_coordinate + width > image_width:
        raise ValueError("The crop width exceeds the image width.")
    if y_coordinate + height > image_height:
        raise ValueError("The crop height exceeds the image height.")

    # Cal Crop field
    left = x_coordinate
    upper = y_coordinate
    right = x_coordinate + width
    lower = y_coordinate + height

    cropped_image = image.crop((left, upper, right, lower))

    return cropped_image


def compute_image_text_similarity(model, processor, image, text):
    # Processing image and text
    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = processor(text=text, return_tensors="pt")

    # Get features
    image_features = model.get_image_features(**image_inputs)
    text_features = model.get_text_features(**text_inputs)

    # Normalize
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # cal similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features)

    return cosine_similarity.item()


if __name__ == '__main__':
    data_names = get_sorted_name(data_path)
    x, y = match_img(data_names[0])
    print(x, y)
    Fac_csv = CSVFilter(data_path)
    names_wiki = Fac_csv.filter_csv_files(source_value='user')
    Fac_csv.clear_results()
    wiki = Fac_csv.filter_csv_files(source_value='wikimedia')
    print('wiki', names_wiki)
    intersection = list(set(names_wiki) & set(wiki))

    # 输出重合内容的列表
    print(intersection)
