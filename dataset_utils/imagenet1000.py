from pathlib import Path

import xmltodict

from tqdm import tqdm
import argparse

import shutil
from joblib import Parallel, delayed


def process_file(xml_file_path):
    with open(str(xml_file_path)) as f:
        xml = f.read()
    parsed = xmltodict.parse(xml)
    annotation = parsed['annotation']
    file_ind = annotation['filename']

    file_name = file_ind + '.JPEG'

    objects = annotation['object']
    if isinstance(objects, list):
        class_id = objects[0]['name']
    else:
        class_id = objects['name']

    class_folder = val_path / class_id
    class_folder.mkdir(exist_ok=True)

    old_image_path = image_path / file_name

    new_image_path = class_folder / file_name

    shutil.copy(str(old_image_path), str(new_image_path))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--val_path', required=True, type=Path,
                        help='Path where validation images will be stored.')
    parser.add_argument('-i', '--image_path', required=True, type=Path,
                        help='Path where validation images are be stored.')
    parser.add_argument('-x', '--xml_path', required=True, type=Path,
                        help='Path where validation xml labels will be stored.')
    parser.add_argument('-nw', '--num_workers', help='Number of CPU threads to use', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    val_path = args.val_path
    val_path.mkdir(exist_ok=True, parents=True)

    image_path = args.image_path
    xml_path = args.xml_path

    Parallel(n_jobs=args.num_workers)(
        delayed(process_file)(xml_file_path) for xml_file_path in tqdm(sorted(xml_path.glob('*.xml'))))
