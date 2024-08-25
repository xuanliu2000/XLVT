import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from utils.read_utils import match_img, get_sorted_name
from tqdm import tqdm
from collections import Counter
from PIL import Image, ImageFile,ImageOps



ImageFile.LOAD_TRUNCATED_IMAGES = True
class CLIPCleaner:
    def __init__(self, data_path=None, out_path=None, model_name="openai/clip-vit-large-patch14", device=None,
                 iou_threshold=0.8):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.df_col = ['label', 'X coordinate', 'Y coordinate', 'height', 'width', 'source']
        self.df_out_col = ['source', 'label', 'box', 'img_dir']
        self.iou_threshold = iou_threshold
        self.path = data_path
        self.out_path = out_path
        self.output = None
        self.clean_data = []
        self.img_csv_path = []
        self.text_temp = ['A photo of a']

    #
    # def calculate_iou(self, box1, box2):
    #     """Cal iou of two bounding boxes"""
    #     x1, y1, h1, w1 = box1  # 正确对应 height 和 width
    #     x2, y2, h2, w2 = box2
    #
    #     xi1 = max(x1, x2)
    #     yi1 = max(y1, y2)
    #     xi2 = min(x1 + w1, x2 + w2)
    #     yi2 = min(y1 + h1, y2 + h2)
    #
    #     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    #
    #     box1_area = w1 * h1
    #     box2_area = w2 * h2
    #
    #     union_area = box1_area + box2_area - inter_area
    #
    #     iou = inter_area / union_area
    #     return iou

    @staticmethod
    def find_duplicates(input_list):
        element_count = Counter(input_list)
        duplicates = [item for item, count in element_count.items() if count > 1]
        return duplicates

    def compare_labels(self, labels, images):

        texts = [f"{self.text_temp[0]} {lbl}" for lbl in labels]
        # print(texts)
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        best_label_index = probs.sum(0).argmax().item()
        # del inputs, outputs, logits_per_image, probs
        #
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #
        # gc.collect()

        return labels[best_label_index]

    @staticmethod
    def crop_transform(box=None,
                       image=None,
                       image_dir=None
                       ):
        if box is None:
            box = [0, 0, 0, 0]
        if image is None:
            raise ValueError("Image cannot be None.")

        x_coordinate, y_coordinate, height, width = box[0], box[1], box[2], box[3]
        image_width, image_height = image.size

        # cropped_image limitation
        if width <= 0 or height <= 0:
            raise ValueError("File{} : Width and height must be greater than 0.".format(image_dir))
        if x_coordinate < 0 or y_coordinate < 0:
            raise ValueError("File{} :X and Y coordinates must be non-negative.".format(image_dir))
        if x_coordinate + width > image_width:
            raise ValueError("File{} :The crop width exceeds the image width.".format(image_dir))
        if y_coordinate + height > image_height:
            raise ValueError("File{} :The crop height exceeds the image height.".format(image_dir))

        # Cal Crop field
        left = x_coordinate
        upper = y_coordinate
        right = x_coordinate + width
        lower = y_coordinate + height

        cropped_image = image.crop((left, upper, right, lower))

        return cropped_image

    def clean_labels(self, df, image_dir):
        # print(df.shape)
        df_user = df[df['source'] == 'user']
        rows = []
        seen_box = {}
        dup_box = {}
        processed_boxes = set()
        # 记录所有的 box，找出重复的和不重复的
        for index, row in df_user.iterrows():
            index_box = tuple([row['X coordinate'], row['Y coordinate'], row['height'], row['width']])
            if index_box in seen_box:
                if index_box not in dup_box:
                    dup_box[index_box] = [seen_box[index_box]]
                dup_box[index_box].append(index)
            else:
                seen_box[index_box] = index
        # preload img
        original_img = Image.open(image_dir)
        original_img = ImageOps.exif_transpose(original_img)

        # deal with all box
        for index, row in df_user.iterrows():
            index_box = tuple([row['X coordinate'], row['Y coordinate'], row['height'], row['width']])
            if index_box in processed_boxes:
                continue

            if index_box in dup_box:
                # 处理重复 box，只执行一次
                cropped_img = self.crop_transform(list(index_box), image=original_img,image_dir=image_dir)
                label_item = df.loc[dup_box[index_box], 'label'].tolist()
                label_out = self.compare_labels(label_item, cropped_img)

                rows.append({
                    'source': df.at[dup_box[index_box][0], 'source'],
                    'label': label_out,
                    'box': list(index_box),
                    'img_dir': image_dir
                })

                # processed boxed
                processed_boxes.add(index_box)
            elif index_box not in dup_box:
                # unread box
                rows.append({
                    'source': row['source'],
                    'label': row['label'],
                    'box': list(index_box),
                    'img_dir': image_dir
                })

        # create DataFrame
        df_wiki = df[df['source'] == 'wikimedia']
        for index, row in df_wiki.iterrows():
            rows.append({
                'source': row.at['source'],
                'label': row['label'],
                'box': [row['X coordinate'], row['Y coordinate'], row['height'], row['width']],
                'img_dir': image_dir
            })
        result_df = pd.DataFrame(rows, columns=self.df_out_col)
        return result_df

    def iter_read(self):
        data_id = get_sorted_name(path=self.path)  # ['1711622935246',....., '1711622941507', ]
        print('Start load img and csv file')
        for index, ids in tqdm(enumerate(data_id), total=len(data_id)):
            # get image and csv path
            # print(index)
            img, csv = match_img(ids, self.path)
            self.img_csv_path.append([img, csv])
            # print([img, csv])
            self.clean_data.append(self.process_csv(csv_path=csv, image_dir=img))
        self.clean_data = pd.concat(self.clean_data, ignore_index=True)
        pd.DataFrame(self.clean_data).to_csv(self.out_path, index=False)
        print(f"Cleaned data saved to {self.out_path}")

    def process_csv(self, csv_path, image_dir):
        # read CSV
        df = pd.read_csv(csv_path,header=None)
        df.columns = self.df_col
        # clean data
        cleaned_df = self.clean_labels(df, image_dir)
        return cleaned_df


if __name__ == '__main__':
    data_path = '/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/data'
    out_path = '/home/lucian/Documents/Hackthon/Factory_Net/factorynet/hackathon/output/output_all.csv'
    clip_cleaner = CLIPCleaner(data_path=data_path, out_path=out_path)
    clip_cleaner.iter_read()
