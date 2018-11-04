"""Script that defines dataloader."""

from torch.utils.data import Dataset
from pathlib import Path
import cv2
import random
import math
from albumentations.torch.functional import img_to_tensor


class DatasetGenerator(Dataset):
    def __init__(self, data_path: Path, transform=None, **kwargs):
        self.img_file_names = sorted(data_path.glob('**/*.JPEG'))

        self.mode = kwargs['mode']
        self.kwargs = kwargs

        self.transform = transform

        class_folders = sorted(data_path.glob('*'))

        assert len(class_folders) == 1000

        self.class_mapping = dict(zip([x.stem for x in class_folders], range(len(class_folders))))

        assert len(self.class_mapping) == 1000

    def __len__(self):
        """Get number of image files identified by this DatasetGenerator."""
        return len(self.img_file_names)

    def __getitem__(self, idx):
        image_file_name = self.img_file_names[idx]

        image = cv2.imread(str(image_file_name))

        if self.mode == 'train':
            y_min, x_min, height, width = get_crop_coords(image, scale=(self.kwargs['min_scale'], 1))
            image = image[y_min:y_min + height, x_min:x_min + width]

        augmented_image = self.transform(image=image)['image']

        class_id = self.class_mapping[image_file_name.parent.name]

        return img_to_tensor(augmented_image), class_id


def get_crop_coords(image, scale=(0.08, 1), ratio=(3 / 4.0, 4 / 3.0)):
    """Get parameters for ``crop`` for a random sized crop.

    Args:
        img (np.array): Image to be cropped.
        scale (tuple, list): range of size of the origin size cropped
        ratio (tuple, list): range of aspect ratio of the origin aspect ratio cropped
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.

    Returns:

    """

    image_height, image_width = image.shape[:2]
    area = image_height * image_width

    target_area = random.uniform(scale[0], scale[1]) * area
    aspect_ratio = random.uniform(ratio[0], ratio[1])

    for attempt in range(10):
        width = int(round(math.sqrt(target_area * aspect_ratio)))
        height = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            width, height = height, width

        if width <= image_width and height <= image_height:
            y_min = random.randint(0, image_height - height)
            x_min = random.randint(0, image_width - width)
            return y_min, x_min, height, width

    # Fallback
    x = min(image_width, image_height)

    y_min = (image_height - x) // 2
    x_min = (image_width - x) // 2
    return y_min, x_min, x, x
