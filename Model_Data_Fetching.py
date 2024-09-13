import shutil
import os
import pandas as pd
import numpy as np
import multiprocessing
import time
from tqdm import tqdm
from pathlib import Path
from functools import partial
import urllib.request
from PIL import Image
import re
import concurrent.futures
import uuid


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
                os.mkdir(path)


class RESOURCE(GLOBAL_VAR):
    def __init__(self, DONWLOAD_IMGS):
        super().__init__()
        self.DOWNLOADER = DONWLOAD_IMGS

    def create_sample(self):
        if os.path.exists(self.DATA_PATH) and os.path.exists(self.SAMPLE_PATH):
            shutil.copy(self.SAMPLE_PATH, self.RES_PATH)
            shutil.copy(self.TRAIN_PATH, self.RES_PATH)
            shutil.copy(self.TEST_PATH, self.RES_PATH)

    def read_sample_batch(self):
        SRC_TRAIN_FILE = os.path.join(self.RES_PATH, "train.csv")
        if os.path.exists(SRC_TRAIN_FILE):
            self.DATA_FRAME = pd.read_csv(SRC_TRAIN_FILE)
            self.NUM_ROWS = self.DATA_FRAME.shape[0]
            self.NUM_COLS = self.DATA_FRAME.shape[1]

            # Get the first 10 image links
            LINKS = self.DATA_FRAME.iloc[:10, 0].tolist()

            # Use concurrent.futures for multi-threading
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for index, link in enumerate(LINKS):
                    unique_filename = f"IMG_SAMPLE_{uuid.uuid4().hex}.jpg"  # Generate unique filename
                    save_path = os.path.join(self.IMG_PATH, unique_filename)
                    futures.append(executor.submit(self.DOWNLOADER.download_image, link, save_path))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # Retrieve result to check for exceptions
                    except Exception as e:
                        print(f"Error in downloading image: {e}")


class DONWLOAD_IMGS(GLOBAL_VAR):
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
            raise ValueError("Invalid format in {}".format(s))
        parts = s_stripped.split(maxsplit=1)
        number = float(parts[0])
        unit = self.common_mistake(parts[1])
        if unit not in self.ALLOWED_UNITS:
            raise ValueError("Invalid unit [{}] found in {}. Allowed units: {}".format(
                unit, s, self.ALLOWED_UNITS))
        return number, unit

    def create_placeholder_image(self, image_save_path):
        try:
            placeholder_image = Image.new('RGB', (100, 100), color='black')
            placeholder_image.save(image_save_path)
        except Exception as e:
            print(f"Error creating placeholder image: {e}")

    def download_image(self, image_link, save_path, retries=3, delay=3):
        if not isinstance(image_link, str):
            return

        if os.path.exists(save_path):
            return

        for _ in range(retries):
            try:
                urllib.request.urlretrieve(image_link, save_path)
                return
            except Exception as e:
                print(f"Error downloading image: {e}")
                time.sleep(delay)

        # Create a placeholder image if download fails
        self.create_placeholder_image(save_path)

    def download_images(self, image_links, allow_multiprocessing=True):
        if not os.path.exists(self.IMG_PATH):
            os.makedirs(self.IMG_PATH)

        if allow_multiprocessing:
            download_image_partial = partial(self.download_image, retries=3, delay=3)

            with multiprocessing.Pool(64) as pool:
                list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
                pool.close()
                pool.join()
        else:
            for image_link in tqdm(image_links, total=len(image_links)):
                self.download_image(image_link, retries=3, delay=3)


if __name__ == "__main__":
    package_obj = PACKAGE()
    package_obj.create_folder()

    download_imgs_obj = DONWLOAD_IMGS()  # Create an instance of DONWLOAD_IMGS

    resource_obj = RESOURCE(download_imgs_obj)  # Pass the instance to RESOURCE
    resource_obj.create_sample()
    resource_obj.read_sample_batch()  # Call the batch method to download images in parallel
