"""
Utility functions for loading INRIA Person Dataset.
"""

import numpy as np
import re
from pathlib import Path
from PIL import Image


class INRIAPersonDatasetHelper:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_samples(self, split, type, max_samples=None):
        samples_dir = Path(self.data_dir) / f"{split}" / type
        if not samples_dir.exists():
            raise ValueError(f"Directory not found: {samples_dir}")
        
        image_files = sorted(
            list(samples_dir.glob('*.png')) + 
            list(samples_dir.glob('*.jpg')) +
            list(samples_dir.glob('*.jpeg')))
        if max_samples:
            image_files = image_files[:max_samples]

        images = []
        for filepath in image_files:
            img = Image.open(filepath).convert('RGB')
            img = np.array(img)
            images.append(img)
        
        return images
    
    def center_crop(self, images, size=(128, 64)):
        images_cropped = []
        for image in images:
            # Center crop from 96x160 to 64x128
            h, w = image.shape[:2]
            crop_h, crop_w = size

            start_y = (h - crop_h) // 2
            start_x = (w - crop_w) // 2

            image_cropped = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
            images_cropped.append(image_cropped)

        return images_cropped
    
    def random_crop(self, images, size=(128, 64), crops_per_image=2):
        images_cropped = []
        for image in images:
            # Center crop from 96x160 to 64x128
            h, w = image.shape[:2]
            crop_h, crop_w = size

            for _ in range(crops_per_image):
                # Random top-left corner
                y = np.random.randint(0, h - crop_h + 1)
                x = np.random.randint(0, w - crop_w + 1)

                # Crop
                image_cropped = image[y:y+crop_h, x:x+crop_w]
                images_cropped.append(image_cropped)
            
        return images_cropped
    
    def get_data(self, split, max_samples=None, shuffle=False, crops_per_image=2):
        print("Loading INRIA Person Dataset...")

        split += "_64x128_H96"

        pos_samples = self.load_samples(split, "pos", max_samples=max_samples)
        pos = self.center_crop(pos_samples)

        neg_samples = self.load_samples(split, "neg", max_samples=max_samples)
        neg = self.random_crop(neg_samples, crops_per_image=crops_per_image)

        if shuffle:
            np.random.shuffle(pos)
            np.random.shuffle(neg)

        print(f"\nData prepared:")
        print(f"- Positive samples: {len(pos)}")
        print(f"- Negative samples: {len(neg)}")

        return pos, neg
    
    def parse_annotation(self, annotation_text):
        bboxes = []

        # Split into lines and find bounding box lines
        lines = annotation_text.split('\n')

        for line in lines:
            # Look for lines like: "Bounding box for object X ... : (Xmin, Ymin) - (Xmax, Ymax)"
            if 'Bounding box' in line and ':' in line:
                # Extract the coordinates part after ":"
                coords_part = line.split(':')[-1].strip()
                # Use regex to extract the four numbers
                match = re.search(r'\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)', coords_part)

                if match:
                    xmin, ymin, xmax, ymax = map(int, match.groups())
                    bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                    bboxes.append(bbox)

        return bboxes
    
    def get_data_for_detection(self, max_samples=None, shuffle=False):
        test_dir = Path(self.data_dir) / "Test"

        # Read pos.lst and annotations.lst
        pos_lst_path = test_dir / "pos.lst"
        annotations_lst_path = test_dir / "annotations.lst"

        if not pos_lst_path.exists():
            raise ValueError(f"pos.lst not found: {pos_lst_path}")
        if not annotations_lst_path.exists():
            raise ValueError(f"annotations.lst not found: {annotations_lst_path}")

        # Read file lists
        with open(pos_lst_path, 'r') as f:
            pos_files = {line.strip() for line in f if line.strip()}
        with open(annotations_lst_path, 'r') as f:
            annotation_files = {line.strip() for line in f if line.strip()}

        # Extract base filenames (e.g., "crop_000001") to find matches
        pos_basenames = {}
        for pos_file in pos_files:
            basename = Path(pos_file).stem  # e.g., "crop_000001"
            pos_basenames[basename] = pos_file

        annotation_basenames = {}
        for ann_file in annotation_files:
            basename = Path(ann_file).stem  # e.g., "crop_000001"
            annotation_basenames[basename] = ann_file

        # Find common basenames
        common_basenames = set(pos_basenames.keys()) & set(annotation_basenames.keys())
        common_basenames = sorted(common_basenames)

        # Load images and annotations for common files
        data_pairs = []
        for basename in common_basenames:
            # Load image
            img_path = Path(self.data_dir) / pos_basenames[basename]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)

            # Load annotation
            ann_path = Path(self.data_dir) / annotation_basenames[basename]
            with open(ann_path, 'r', encoding='latin-1') as f:
                annotation_text = f.read()

            data_pairs.append({
                'image': img,
                'annotation': self.parse_annotation(annotation_text),
                'image_path': str(img_path),
                'annotation_path': str(ann_path)
            })
        
        if shuffle:
            np.random.shuffle(data_pairs)

        if max_samples:
            data_pairs = data_pairs[:max_samples]

        print(f"Loaded {len(data_pairs)} image-annotation pairs for detection")

        return data_pairs
