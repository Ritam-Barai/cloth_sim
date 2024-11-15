import os
import json
import random
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import logging

class DeepFashion2Dataset(Dataset):
    """Dataset for DeepFashion2 with landmark detection support."""

    def __init__(self, 
                 img_dir: str,
                 anno_dir: str,
                 dataset_file:str,
                 samples: int,
                 image_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the dataset for DeepFashion2.
        
        Args:
            img_dir: img directory containing images
            annotation_dir: Path to annotation file
            phase: 'train', 'val', or 'test'
            image_size: Target image size
        """
        self.img_dir = Path(img_dir)
        self.anno_dir = Path(anno_dir)
        self.image_size = image_size
        self.samples = samples
        self.is_split = False
        self.dataset_file = dataset_file
        
        if self.dataset_file:
            self.data = pd.read_csv(dataset_file)
        
        else:
            # Load and process annotations
            self.data = self._load_annotations()
        self.data.reset_index(drop=True, inplace=True)
        
        
        
        
    def _load_annotations(self) -> pd.DataFrame:
        """Load and process annotation files for DeepFashion2."""
        processed_data = []
        
        with os.scandir(self.img_dir) as entries:
            # Filter entries to include only images and pick samples on-the-fly
            image_paths = [
            entry.path for entry in entries
                if entry.is_file() and entry.name.lower().endswith('.jpg')
            ]
            
            # If there are not enough images
            if len(image_paths) < self.samples:
                raise ValueError("Requested sample size is larger than available images.")
        
            
            # Randomly sample the images
            random_img_files = random.sample(image_paths, self.samples)
            #print(random_img_files)
        
        for img_file in random_img_files:
            #if img_file.endswith('.jpg'):
                # Assuming the corresponding JSON annotation has the same name
                json_file = img_file.replace('.jpg', '.json')
                _,json_file = os.path.split(json_file)
                anno_path = self.anno_dir / json_file
                
                
                if not anno_path.exists():
                    continue  # Skip if JSON file does not exist

                with open(anno_path) as f:
                    annos = json.load(f)
                    
                # Load the image to get its dimensions
                #image_path = os.path.join(self.img_dir,img_file)
                image = cv2.imread(img_file)
                #print(anno_path,img_file)
                height, width, _ = image.shape  # Get image dimensions
                
                # Extract all items with keys like item1, item2, ...
                item_index = 1
                while True:
                    item_key = f'item{item_index}'
                    if item_key not in annos:
                        break
                       
                    anno = annos[item_key]
                    bbox = [
                    anno['bounding_box'][0] / width,
                    anno['bounding_box'][1] / height,
                    anno['bounding_box'][2] / width,
                    anno['bounding_box'][3] / height
                    ]
                
                    # Retrieve landmarks and normalize
                    keypoints = []
                    if 'landmarks' in anno:
                        for i in range(0, len(anno['landmarks']), 3):
                            x, y, v = anno['landmarks'][i:i + 3]
                            normalized_keypoint = {
                                'x': x / width,
                                'y': y / height,
                                'visibility': v  # Use the visibility as is
                            }
                            keypoints.append(normalized_keypoint)
                
                    # Retrieve additional annotation fields
                    category = anno.get('category_name', '')
                    occlusion = anno.get('occlusion', 0)  # Default to 0 if not present
                
                    processed_data.append({
                        'image_path': img_file,
                        'height':height,
                        'width': width,
                        'item_index': item_index,
                        'keypoints': keypoints,
                        'category': category,
                        'occlusion': occlusion,
                        'bbox': bbox
                    })
        
                    item_index += 1
        
        return pd.DataFrame(processed_data)
    
    def _split_dataset(self,phase = 'train'):
        """Split data into train, validation, and test sets."""
        self.phase = phase
        if not self.is_split:
            self.train_df, temp_df = train_test_split(
            self.data, test_size=0.3, random_state=42, 
            #stratify=self.data['category']
            )
        
            self.val_df, self.test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42,
            #stratify=temp_df['category']
            )
            self.is_split = True
        
        # Setup augmentations
        self.transform = self._get_transforms()
        
        if self.phase == 'train':
            return self.train_df, self.transform
        elif self.phase == 'val':
            return self.val_df, self.transform
        else:
            return self.test_df, self.transform
            
        
        #return self.df
            
    def _get_transforms(self) -> A.Compose:
        """Define augmentation transformations."""
        if self.phase == 'train':
            return A.Compose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.HorizontalFlip(p=0.5),
                A.Resize(self.image_size[0], self.image_size[1]),
                #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
               bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))  # Enable scaling)
        else:
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))  # Enable scaling)
    




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform, category_to_keypoints: Dict[str, int]):
        """
        Initialize CustomDataset.
        
        Args:
            dataframe: Pandas DataFrame containing dataset information
            transform: Albumentations transform pipeline
        """
        self.df = dataframe
        self.transform = transform
        self.category_to_keypoints = category_to_keypoints
        logger.info(f"Initialized CustomDataset with {len(dataframe)} samples")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def normalize_visibility(self, v):
        """
        Normalize visibility values from [0,1,2] to [0,0.5,1].
        0 = not visible
        1 = partially visible
        2 = fully visible
        """
        if v == 0:
            return 0.0
        elif v == 1:
            return 0.5
        elif v == 2:
            return 1.0
        else:
            logger.warning(f"Unexpected visibility value: {v}, defaulting to 0")
            return 0.0
    
    def parse_keypoints(self, keypoints_data,num_keypoints):
        """
        Parse keypoints from DataFrame storage format.
        Returns a list of [x, y] coordinates and a list of visibility flags.
        """
        try:
            logger.debug(f"Parsing keypoints data: {keypoints_data[:100]}...")
            
            if isinstance(keypoints_data, str):
                keypoints_data = keypoints_data.strip()
                import ast
                keypoints = ast.literal_eval(keypoints_data)
                logger.debug("Parsed keypoints from string format")
            elif isinstance(keypoints_data, list):
                keypoints = keypoints_data
                logger.debug("Using list format keypoints")
            else:
                logger.warning(f"Invalid keypoints data type: {type(keypoints_data)}")
                return [[0, 0] for _ in range(40)], [0] * 40
            
            keypoints_list = []
            visibility_list = []
            
            for i, kp in enumerate(keypoints):
                if isinstance(kp, dict):
                    x = float(kp.get('x', 0))
                    y = float(kp.get('y', 0))
                    v = self.normalize_visibility(float(kp.get('visibility', 0)))
                    keypoints_list.append([x, y])
                    visibility_list.append(v)
                    logger.debug(f"Keypoint {i}: x={x}, y={y}, visibility={v}")
                elif isinstance(kp, (list, tuple)):
                    if len(kp) >= 2:
                        x, y = float(kp[0]), float(kp[1])
                        v = self.normalize_visibility(float(kp[2])) if len(kp) > 2 else 0
                        keypoints_list.append([x, y])
                        visibility_list.append(v)
                        logger.debug(f"Keypoint {i}: x={x}, y={y}, visibility={v}")
            
            logger.debug(f"Keypoints:  {len(keypoints_list)} ")

            # Log if padding was needed
            if len(keypoints_list) < 40:
                logger.debug(f"Padding keypoints from {len(keypoints_list)} to {num_keypoints} and extending to 40")
                while len(keypoints_list) < 40:
                    keypoints_list.append([0, 0])
                    visibility_list.append(0)
            
            # Validate keypoint values
            for i, (kp, v) in enumerate(zip(keypoints_list[:num_keypoints], visibility_list[:num_keypoints])):
                if not (isinstance(kp[0], (int, float)) and isinstance(kp[1], (int, float))):
                    logger.error(f"Invalid keypoint values at index {i}: {kp}")
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    logger.warning(f"Keypoint coordinates out of range at index {i}: x={kp[0]}, y={kp[1]}")
                if not (0 <= v <= 1):
                    logger.error(f"Normalized visibility value out of range at index {i}: {v}")
            
            return keypoints_list[:40], visibility_list[:40]
            
        except Exception as e:
            logger.error(f"Error parsing keypoints: {str(e)}")
            return [[0, 0] for _ in range(40)], [0] * 40
    
    def parse_bbox(self, bbox_data):
        """Parse bounding box from DataFrame storage format."""
        try:
            logger.debug(f"Parsing bbox data: {bbox_data}")
            
            if isinstance(bbox_data, str):
                import ast
                bbox = ast.literal_eval(bbox_data)
            elif isinstance(bbox_data, (list, tuple)):
                bbox = bbox_data
            else:
                logger.warning(f"Invalid bbox data type: {type(bbox_data)}")
                return [0, 0, 1, 1]
            
            bbox = [float(x) for x in bbox[:4]]
            logger.debug(f"Parsed bbox: {bbox}")
            
            # Validate bbox values
            if not all(0 <= x <= 1 for x in bbox):
                logger.warning(f"Bbox coordinates out of range: {bbox}")
            
            return bbox
            
        except Exception as e:
            logger.error(f"Error parsing bbox: {str(e)}")
            return [0, 0, 1, 1]
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample with image and landmarks."""
        try:
            item = self.df.iloc[idx]
            logger.info(f"Processing item {idx} with path: {item['image_path']}")
            
            # Load and process image
            image = cv2.imread(str(item['image_path']))
            if image is None:
                logger.error(f"Failed to load image: {item['image_path']}")
                raise ValueError(f"Failed to load image: {item['image_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.debug(f"Loaded image shape: {image.shape}")
            
            category = item['category']
            num_keypoints = self.category_to_keypoints[category]
            # Parse keypoints and visibility
            keypoints, visibility = self.parse_keypoints(item['keypoints'],num_keypoints)
            logger.debug(f" keypoints_x: {keypoints}, {visibility}")
            logger.debug(f"Parsed {len(keypoints)} keypoints")
            scaled_keypoints = list(map(lambda coord: [coord[0] * item['width'], coord[1] * item['width']], keypoints))
            logger.debug(f"Scaled keypoints: {scaled_keypoints}")
            
            
            # Parse bbox
            bbox = self.parse_bbox(item['bbox'])
            logger.debug(f"Parsed bbox: {bbox}")
            scaled_bbox = [coord * (item['width'] if i % 2 == 0 else item['height']) for i, coord in enumerate(bbox)]
            logger.debug(f"Parsed bbox: {scaled_bbox}")
            
            # Apply transforms
            if self.transform:
                logger.debug("Applying transforms")
                transformed = self.transform(
                    image=image,
                    keypoints=scaled_keypoints,
                    bboxes = [scaled_bbox],
                    labels = [category]
                )
                image = transformed['image']
                #diff = list(map(lambda pair: pair[0]*image.shape[0], keypoints))
                #logger.info(f" Transformed keypoints_x: {diff}")
                keypoints = transformed['keypoints']
                bbox = transformed['bboxes']
                logger.debug(f"Transformed image shape: {image.shape}")
                logger.debug(f"Transformed keypoints: {keypoints}")
                logger.debug(f"Transformed bbox: {bbox}")
                
                
            
            
            category_index = list(self.category_to_keypoints.keys()).index(category)
            #keypoints = keypoints[:num_keypoints]
            #visibility = visibility[:num_keypoints]
            # Convert to tensors
            image = torch.from_numpy(image).float().permute(2, 0, 1)
            keypoints = torch.tensor(keypoints).float()
            visibility = torch.tensor(visibility).float()
            num_keypoints = torch.tensor(num_keypoints).int()
            category_index = torch.tensor(category_index).int()
            bbox = torch.tensor(bbox).float()
            
            return {
                'image': image,
                'keypoints': keypoints,
                'visibility': visibility,
                'bbox': bbox,
                'num_keypoints':num_keypoints,
                'category_index' : category_index,
                'category': str(item.get('category', '')),
                'occlusion': float(item.get('occlusion', 0.0)),
                'item_index': int(item.get('item_index', 0)),
                
            }
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            # Return a default item in case of error
            return {
                'image': torch.zeros((3, 256, 256)),
                'keypoints': torch.zeros((40, 2)),
                'visibility': torch.zeros(40),
                'num_keypoints':40,
                'category_index' : 0,
                'bbox': torch.tensor([0, 0, 1, 1]),
                'category': '',
                'occlusion': 0.0,
                'item_index': 0
            }

def create_dataloaders(
    img_dir: str,
    anno_dir: str,
    dataset_dir: str,
    category_to_keypoints:Dict[str, int],
    sample: int = 5000, 
    batch_size: int = 32,
    num_workers: int = 4
    
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training, validation, and testing.
    """
    # Load or create dataset
    if os.path.exists(os.path.join(dataset_dir, "main_df.csv")):
        dataset_file = os.path.join(dataset_dir, "main_df.csv")
        dataset = DeepFashion2Dataset(
            img_dir=img_dir, 
            anno_dir=anno_dir,
            dataset_file=dataset_file,
            samples=sample
        )
    else:
        dataset = DeepFashion2Dataset(
            img_dir=img_dir, 
            anno_dir=anno_dir,
            dataset_file=None,
            samples=sample
        )
        os.makedirs(dataset_dir, exist_ok=True)
        dataset.data.to_csv(os.path.join(dataset_dir, "main_df.csv"), index=False)
    
    # Split dataset and create transforms
    train_df, train_transform = dataset._split_dataset(phase='train')
    val_df, val_transform = dataset._split_dataset(phase='val')
    test_df, test_transform = dataset._split_dataset(phase='test')
    
    # Create datasets
    train_dataset = CustomDataset(train_df, train_transform,category_to_keypoints)
    val_dataset = CustomDataset(val_df, val_transform,category_to_keypoints)
    test_dataset = CustomDataset(test_df, test_transform,category_to_keypoints)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class DeepFashionVisualizer:
    """Visualize dataset samples and augmentations."""

    def __init__(self, dataset: DeepFashion2Dataset):
        self.dataset = dataset

    def visualize_sample(self, idx: int):
        """Display image with keypoints and bounding box."""
        sample = self.dataset[idx]
        
        image = sample['image'].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        
        keypoints = sample['keypoints'].numpy()
        visibility = sample['visibility'].numpy()
        
        for (x, y), v in zip(keypoints, visibility):
            color = 'g' if v > 0 else 'r'
            plt.plot(x * image.shape[1], y * image.shape[0], 'o', color=color, markersize=8)
        
        if 'bbox' in sample:
            bbox = sample['bbox'].numpy()
            x1, y1, x2, y2 = bbox * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'b-')
        
        plt.title(f"Category: {sample['category']}")
        plt.axis('off')
        plt.show()




'''
def create_dataloaders(
    img_dir: str,
    anno_dir: str,
    dataset_dir: str,
    sample: int = 5000, 
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training, validation, and testing.
    """
    if len(os.listdir(dataset_dir)) is not 0:
        #print(len(os.listdir(dataset_dir)))
        dataset_file= f"{dataset_dir}/main_df.csv"
        dataset = DeepFashion2Dataset(img_dir = img_dir, anno_dir = anno_dir,dataset_file = dataset_file,samples = sample)
        #print()
    
    else:
        dataset = DeepFashion2Dataset(img_dir = img_dir, anno_dir = anno_dir,dataset_file = None,samples = sample)
        dataset.data.to_csv(f"{dataset_dir}/main_df.csv", index=True)
        
        
        
    train_df,train_transform = dataset._split_dataset( phase='train')
    train_df.reset_index(drop=True, inplace=True)
    train_dataset = CustomDataset(train_df,train_transform).df
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    print(train_dataset.shape,train_dataset.index)
    
    
    val_df,val_transform = dataset._split_dataset( phase='val')
    val_df.reset_index(drop=True, inplace=True)
    val_dataset = CustomDataset(val_df,val_transform).df
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    test_df,test_transform = dataset._split_dataset( phase='test')
    test_df.reset_index(drop=True, inplace=True)
    test_dataset = CustomDataset(test_df,test_transform).df
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)    
    
  
    
    
    
    print(next(iter(train_loader)))
    
    return train_loader, val_loader, test_loader'''
