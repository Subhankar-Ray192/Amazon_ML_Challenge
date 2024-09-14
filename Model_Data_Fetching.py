from google.colab import drive
drive.mount('/content/drive')

import shutil
import os
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
from functools import partial
import urllib.request
from PIL import Image
import re
import multiprocessing
import uuid  # Use uuid module for generating unique IDs

class GLOBAL_VAR:
    def __init__(self):
        self.DRIVE = "/content/drive/MyDrive/Amazon_ML_Challenge"
        self.DATA_SET = "66e31d6ee96cd_student_resource_3/student_resource 3/dataset"

        self.DATA_PATH = os.path.join(self.DRIVE, self.DATA_SET)
        self.SAMPLE_PATH = os.path.join(self.DATA_PATH, "sample_test.csv")
        self.TRAIN_PATH = os.path.join(self.DATA_PATH, "train.csv")
        self.TEST_PATH = os.path.join(self.DATA_PATH, "test.csv")

        self.OUT_ROOT = "app"
        self.OUT_DATA = "data"
        self.OUT_TRAIN = "train_sample"
        self.OUT_TEST = "test_sample"
        self.OUT = "output"
        self.RES = "resource"
        self.IMG = "image"

        self.OUT_ROOT_PATH = os.path.join(self.DRIVE, self.OUT_ROOT)
        self.OUT_DATA_PATH = os.path.join(self.OUT_ROOT_PATH, self.OUT_DATA)
        self.OUT_TRAIN_PATH = os.path.join(self.OUT_DATA_PATH, self.OUT_TRAIN)
        self.OUT_TEST_PATH = os.path.join(self.OUT_DATA_PATH, self.OUT_TEST)
        self.OUT_PATH = os.path.join(self.OUT_ROOT_PATH, self.OUT)
        self.RES_PATH = os.path.join(self.OUT_ROOT_PATH, self.RES)
        self.IMG_PATH = os.path.join(self.RES_PATH, self.IMG)

        self.ENTITY_UNIT_MAP = {
            'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
            'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
            'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
            'voltage': {'kilovolt', 'millivolt', 'volt'},
            'wattage': {'kilowatt', 'watt'},
            'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                            'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
        }

        self.ALLOWED_UNITS = {unit for entity in self.ENTITY_UNIT_MAP for unit in self.ENTITY_UNIT_MAP[entity]}
        self.CATEGORY = {entity for entity in self.ENTITY_UNIT_MAP}
        self.CATEGORY_PATH = {os.path.join(self.IMG_PATH, entity) for entity in self.CATEGORY}


class PACKAGE(GLOBAL_VAR):
    def __init__(self):
        super().__init__()

    def create_folder(self):
        paths = [
            self.OUT_ROOT_PATH, self.OUT_DATA_PATH, self.OUT_TRAIN_PATH, self.OUT_TEST_PATH, self.OUT_PATH,
            self.RES_PATH, self.IMG_PATH
        ]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)


class RESOURCE(GLOBAL_VAR):
    def __init__(self, downloader):
        super().__init__()
        self.DOWNLOADER = downloader
        self.processed_groups = set()  # Keep track of processed group_ids

    def create_sample(self):
        if os.path.exists(self.DATA_PATH) and os.path.exists(self.SAMPLE_PATH):
            shutil.copy(self.SAMPLE_PATH, self.RES_PATH)
            shutil.copy(self.TRAIN_PATH, self.RES_PATH)
            shutil.copy(self.TEST_PATH, self.RES_PATH)

    def read_sample_batch(self, START, END):
        SRC_TRAIN_FILE = os.path.join(self.RES_PATH, "train.csv")
        if os.path.exists(SRC_TRAIN_FILE):
            self.DATA_FRAME = pd.read_csv(SRC_TRAIN_FILE)
            self.NUM_ROWS = self.DATA_FRAME.shape[0]
            self.NUM_COLS = self.DATA_FRAME.shape[1]
            self.BATCH_SIZE = 100
            print(f"Number of rows: {self.NUM_ROWS}")
            print(f"Number of columns: {self.NUM_COLS}")

            # Fix the slicing to go from START to END
            self.LINKS = self.DATA_FRAME.iloc[START:END, 0].tolist()
            self.GROUP_ID = self.DATA_FRAME.iloc[START:END, 1].tolist()
            self.createGROUPS(self.GROUP_ID)

            self.CATEGORY_D = self.DATA_FRAME.iloc[START:END, 2].tolist()
            self.DOWNLOADER.download_images(self.LINKS, self.GROUP_ID, self.CATEGORY_D)

    def createGROUPS(self, group_ids):
        for group_id in group_ids:
            # Check if the group_id has already been processed
            if group_id not in self.processed_groups:
                group_path = os.path.join(self.IMG_PATH, str(group_id))
                # Ensure thread-safe directory creation
                Path(group_path).mkdir(parents=True, exist_ok=True)

                # Create category folders inside the group folder
                for category in self.CATEGORY:
                    category_path = os.path.join(self.IMG_PATH, str(group_id), category)
                    # Ensure thread-safe directory creation
                    Path(category_path).mkdir(parents=True, exist_ok=True)

                # Add group_id to the processed set to prevent re-processing
                self.processed_groups.add(group_id)


class DOWNLOAD_IMGS(GLOBAL_VAR):
    def __init__(self):
        super().__init__()

    def common_mistake(self, unit):
        if unit in self.ALLOWED_UNITS:
            return unit
        if unit.replace('ter', 'tre') in self.ALLOWED_UNITS:
            return unit.replace('ter', 'tre')
        if unit.replace('feet', 'foot') in self.ALLOWED_UNITS:
            return unit.replace('feet', 'foot')
        return unit

    def parse_string(self, s):
        s_stripped = "" if s is None or str(s) == 'nan' else s.strip()
        if s_stripped == "":
            return None, None
        pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
        if not pattern.match(s_stripped):
            raise ValueError(f"Invalid format in {s}")
        parts = s_stripped.split(maxsplit=1)
        number = float(parts[0])
        unit = self.common_mistake(parts[1])
        if unit not in self.ALLOWED_UNITS:
            raise ValueError(f"Invalid unit [{unit}] found in {s}. Allowed units: {self.ALLOWED_UNITS}")
        return number, unit

    def create_placeholder_image(self, image_name, category):
        try:
            placeholder_image = Image.new('RGB', (100, 100), color='black')
            image_save_path = os.path.join(self.IMG_PATH, category, image_name)
            placeholder_image.save(image_save_path)
        except Exception as e:
            print(f"Error creating placeholder image: {e}")

    def download_image(self, image_link, group_id, category, retries=3, delay=3):
        if not isinstance(image_link, str):
            return

        # Create a path using IMG_PATH and category
        category_path = os.path.join(self.IMG_PATH, str(group_id), category)

        # Ensure thread-safe directory creation
        Path(category_path).mkdir(parents=True, exist_ok=True)

        # Modify the image name to IMG-{UUID}.jpg
        image_name = f"IMG-{uuid.uuid4().hex}.jpg"  # Use uuid4() to generate a unique ID
        image_save_path = os.path.join(category_path, image_name)

        for _ in range(retries):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
                return
            except Exception as e:
                print(f"Error downloading image: {e}")
                time.sleep(delay)

        # Create a placeholder image if download fails
        self.create_placeholder_image(image_name, category)

    def download_images(self, image_links, group_ids, categories, allow_multiprocessing=True):
        if not os.path.exists(self.IMG_PATH):
            os.makedirs(self.IMG_PATH)

        if allow_multiprocessing:
            # Use partial to pass both image_links, group_ids, and categories
            with multiprocessing.Pool(8) as pool:
                pool.starmap(self.download_image, tqdm(zip(image_links, group_ids, categories), total=len(image_links)))
        else:
            for image_link, group_id, category in tqdm(zip(image_links, group_ids, categories), total=len(image_links)):
                self.download_image(image_link, group_id, category)


def main():
    package = PACKAGE()
    package.create_folder()

    resource = RESOURCE(DOWNLOAD_IMGS())
    resource.create_sample()

    # Adjust the START and END parameters as needed
    START = 0
    END = 10
    resource.read_sample_batch(START, END)


if __name__ == "__main__":
    main()
