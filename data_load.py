from tensorflow.keras.utils import Sequence
import cv2
from PIL import Image
import numpy as np

import yaml

import os, glob
import math
import json

def get_config(config_path):
    with open(config_path) as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    return train_config

IMG_FOLDER_PATH = './data/img'
TRAIN_CONFIG_PATH = 'config.yaml'
train_config = get_config(TRAIN_CONFIG_PATH)

# get config params
IMAGE_HEIGHT = train_config['IMAGE_HEIGHT']
IMAGE_WIDTH = train_config['IMAGE_WIDTH']

GRID_SIZE = train_config['GRID_SIZE']
BATCH_SIZE = train_config['BATCH_SIZE']

class DataGenerator(Sequence):

    def __init__(self, meta_info_folder_path):

        self.meta_info_paths = glob.glob('{0}/*.json'.format(meta_info_folder_path))

        pass

    def __len__(self):
        return math.ceil(len(self.meta_info_paths) / BATCH_SIZE)

    def __getitem__(self, idx):

        batch_meta_info_paths = self.meta_info_paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        batch_images = np.zeros((len(self.meta_info_paths), IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
        batch_boxes = np.zeros((len(self.meta_info_paths), GRID_SIZE, GRID_SIZE, 5), dtype=np.float32)

        for i, meta_info_path in enumerate(batch_meta_info_paths):

            # json file parsing
            with open(meta_info_path) as json_file:
                json_data = json.load(json_file)

            img_file_name = json_data['url'].split('/')[-1]
            img_path = os.path.join(IMG_FOLDER_PATH, img_file_name)

            with Image.open(img_path) as img:
                origin_height, origin_width = img.size[:2]

                img = img.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
                img = img.convert('RGB')
                img = np.array(img, dtype=np.float32)

                # re-scailing
                batch_images[i] = img / 255.0

            # bbox (넥라인과 object bbox가 혼재해 있음) // 넥라인 bbox 제외
            bboxes = json_data['result']['boxes']

            for bbox in bboxes:

                if '넥' not in bbox['label']:

                    minX, minY, maxX, maxY = bbox['minX'], bbox['minY'], bbox['maxX'], bbox['maxY']
                    coords = (minX, minY, maxY, maxY)
                    coords = tuple(map(math.floor, coords))

                    x0, y0, x1, y1 = coords

                    y_c = (GRID_SIZE / origin_height) * (y0 + (y1 - y0) / 2)
                    x_c = (GRID_SIZE / origin_width) * (x0 + (x1 - x0) / 2)

                    floor_y = math.floor(y_c)
                    floor_x = math.floor(x_c)

                    grid_y = (y1 - y0) / origin_height
                    grid_x = (x1 - x0) / origin_width
                    grid_yc = y_c - floor_y
                    grid_xc = x_c - floor_x

                    batch_boxes[i, floor_y, floor_x, 0] = round(grid_y, 3)
                    batch_boxes[i, floor_y, floor_x, 1] = round(grid_x, 3)
                    batch_boxes[i, floor_y, floor_x, 2] = round(grid_yc, 3)
                    batch_boxes[i, floor_y, floor_x, 3] = round(grid_xc, 3)
                    batch_boxes[i, floor_y, floor_x, 4] = 1

        return batch_images, batch_boxes


if __name__ == '__main__':

    META_INFO_FOLDER_PATH = './data/meta/train'
    data_generator = DataGenerator(META_INFO_FOLDER_PATH)

    # for batch_images, batch_boxes in data_generator:
    #     pass

