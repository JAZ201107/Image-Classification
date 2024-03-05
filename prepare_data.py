import argparse
import random
import os
import gdown
from subprocess import check_call
import shutil
import zipfile

from PIL import Image
from tqdm import tqdm

import config


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", default="data/SIGNS", help="Directory with the SIGNS dataset"
    )
    parser.add_argument(
        "--output_dir", default="data/64x64_SIGNS", help="Where to write the new data"
    )

    return parser


def resize_and_save(filename, output_dir, size=config.IMAGE_SIZE):
    image = Image.open(filename)
    image = image.resize((size, size), Image.BILINEAR)

    image.save(os.path.join(output_dir, filename.split("/")[-1]))


def download_and_unzip(data_dir):

    if os.path.exists(data_dir):
        print(f"{data_dir} has already exists")
        # os.rmdir(data_dir)
        shutil.rmtree(data_dir, ignore_errors=True)

    zip_file = "data.zip"

    # Download trained_models.zip file to trained_models folder
    os.mkdir(data_dir)
    # cmd = f"wget --no-check-certificate 'https://drive.usercontent.google.com/download?id=1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC&export=download&authuser=0' -O {os.path.join(data_dir,zip_file)}"

    url = "https://drive.usercontent.google.com/download?id=1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC&export=download&authuser=0"
    output = os.path.join(data_dir, zip_file)
    gdown.download(url, output)

    zip_file_path = os.path.join(data_dir, zip_file)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(zip_file_path)

    # check_call(cmd)

    # cmd = f"unzip {os.path.join(data_dir,zip_file)} -d {data_dir}"

    # check_call(cmd)

    print("Down")


if __name__ == "__main__":
    args = get_parser().parse_args()
    # download_and_unzip(args.data_dir)

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(
        args.data_dir
    )

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, "train_signs")
    test_data_dir = os.path.join(args.data_dir, "test_signs")

    filenames = os.listdir(train_data_dir)
    filenames = [
        os.path.join(train_data_dir, f) for f in filenames if f.endswith(".jpg")
    ]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [
        os.path.join(test_data_dir, f) for f in test_filenames if f.endswith(".jpg")
    ]

    random.seed(42)
    filenames.sort()
    random.shuffle(filenames)

    split = int(config.TRAIN_SPLIT * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]

    filenames = {"train": train_filenames, "val": val_filenames, "test": test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    else:
        assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(
            args.data_dir
        )

    for split in ["train", "val", "test"]:
        output_dir_split = os.path.join(args.output_dir, f"{split}_signs")
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))
        print(
            "Processing {} data, saving preprocessed data to {}".format(
                split, output_dir_split
            )
        )
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=config.IMAGE_SIZE)

    print("Done building dataset")
