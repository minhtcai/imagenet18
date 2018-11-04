from pathlib import Path

from tqdm import tqdm
import argparse

from joblib import Parallel, delayed
import albumentations as albu
import cv2


def process_file(file_path, transform):
    image = cv2.imread(str(file_path))

    image = transform(image=image)['image']

    class_id = file_path.parent.name

    class_folder = output_path / class_id
    class_folder.mkdir(exist_ok=True, parents=True)
    file_name = file_path.name

    new_image_path = class_folder / file_name

    cv2.imwrite(str(new_image_path), image)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', required=True, type=Path,
                        help='Output path where images will be stored.')
    parser.add_argument('-i', '--input_path', required=True, type=Path,
                        help='Input path to images.')
    parser.add_argument('-nw', '--num_workers', help='Number of CPU threads to use', type=int, default=12)
    parser.add_argument('-t', '--target_size', help='Minimum side target size.', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    input_path = args.input_path
    input_path.mkdir(exist_ok=True, parents=True)

    output_path = args.output_path

    transform = albu.SmallestMaxSize(max_size=args.target_size, p=1)

    Parallel(n_jobs=args.num_workers)(
        delayed(process_file)(file_name, transform) for file_name in tqdm(sorted(input_path.glob('**/*.JPEG'))))
