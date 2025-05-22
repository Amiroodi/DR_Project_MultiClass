import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
import cv2
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split


IDRID_image_folder = "../IDRID/Imagenes/Imagenes" 
IDRID_csv_file = "../IDRID/idrid_labels.csv"  

MESSIDOR_image_folder = "../MESSIDOR/messidor-2/messidor-2/preprocess"
MESSIDOR_csv_file = "../MESSIDOR/messidor_data.csv"

APTOS_19_train_image_folder = "../APTOS/resized train 19"
APTOS_19_train_csv_file = "../APTOS/labels/trainLabels19.csv"  

APTOS_15_train_image_folder = "../APTOS/resized train 15"
APTOS_15_train_csv_file = "../APTOS/labels/trainLabels15.csv" 

APTOS_15_test_image_folder = "../APTOS/resized test 15"
APTOS_15_test_csv_file = "../APTOS/labels/testLabels15.csv"  

NUM_WORKERS = 8

class LoadLabels(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file) 
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx, 1]  # Assuming second column is label (0-4)
        return label
    
class LoadDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.image_folder = image_folder
        self.df = pd.read_csv(csv_file) 
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image filename and label from the DataFrame
        img_name = self.df.iloc[idx, 0]  # Assuming first column is filename
        label = self.df.iloc[idx, 1]  # Assuming second column is label (0-4)

        if label >= 1: label = 1.0

        # Load image
        if self.image_folder == MESSIDOR_image_folder: # messidor has the .jpg name in its files
            img_path = os.path.join(self.image_folder, img_name)
        else:
            img_path = os.path.join(self.image_folder, img_name) + '.jpg'

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label
    
def LoadDataset_train_val_test_split(transform, shrink_size, train_size=0.7, val_size=0.15, test_size=0.15):
    train_dataset_1 = LoadDataset(APTOS_19_train_image_folder, APTOS_19_train_csv_file, transform=transform)
    # train_dataset_2 = LoadDataset(MESSIDOR_image_folder, MESSIDOR_csv_file, transform=transform)
    # train_dataset_3 = LoadDataset(IDRID_image_folder, IDRID_csv_file, transform=transform)

    # combined_dataset = ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])
    combined_dataset = ConcatDataset([train_dataset_1])


    # StratifiedShuffleSplit is slow for combined_dataset because augmentations are applied, so we load labels seperately
    labels_dataset_1 = LoadLabels(APTOS_19_train_csv_file)
    # labels_dataset_2 = LoadLabels(MESSIDOR_csv_file)
    # labels_dataset_3 = LoadLabels(IDRID_csv_file)

    # combined_labels_dataset = ConcatDataset([labels_dataset_1, labels_dataset_2, labels_dataset_3])
    combined_labels_dataset = ConcatDataset([labels_dataset_1])

    # combined_labels_dataset = labels_dataset_1

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        combined_dataset = Subset(combined_dataset, range(shrink_size))
        combined_labels_dataset = Subset(combined_labels_dataset, range(shrink_size))

    labels = [combined_labels_dataset[i] for i in range(len(combined_labels_dataset))]  # assuming (image, label)

    # Step 3: Create stratified train, validation, and test splits
    # First, split into train and temp (val + test)
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        range(len(combined_dataset)),
        labels,
        train_size=train_size,
        stratify=labels,
        random_state=33
    )

    # Adjust val_size for the second split (val_size / (val_size + test_size))
    relative_val_size = val_size / (val_size + test_size)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_labels,
        train_size=relative_val_size,
        stratify=temp_labels,
        random_state=33
    )

    # Step 4: Create Subset datasets for train, validation, and test
    train_dataset = Subset(combined_dataset, train_idx)
    val_dataset = Subset(combined_dataset, val_idx)
    test_dataset = Subset(combined_dataset, test_idx)

    return train_dataset, val_dataset, test_dataset

def create_train_val_dataloader(
    train_transform,
    val_transform,
    batch_size: int, 
    shrink_size,
    num_workers: int=NUM_WORKERS
    ):
  
    # minimum augmentations should be applied to val_dataset, not the case for train_dataset
    train_dataset, _, _ = LoadDataset_train_val_test_split(transform=train_transform, shrink_size=shrink_size)
    _, val_dataset, _ = LoadDataset_train_val_test_split(transform=val_transform, shrink_size=shrink_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    # Get class names
    class_names = ['No DR', 'DR']

    return train_dataloader, val_dataloader, class_names

def create_test_dataloader(
    test_transform,
    batch_size: int,
    shrink_size, 
    num_workers: int=NUM_WORKERS,
    ):
  
    _, _, test_dataset = LoadDataset_train_val_test_split(transform=test_transform, shrink_size=shrink_size)

    # Get class names
    class_names = ['No DR', 'DR']

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    return test_dataloader, class_names


def create_exotic_test_dataloader(
    test_transform,
    batch_size: int,
    shrink_size, 
    dataset_name,
    num_workers: int=NUM_WORKERS,
    ):
  
    if dataset_name == 'MESSIDOR':
        test_dataset = LoadDataset(MESSIDOR_image_folder, MESSIDOR_csv_file, transform=test_transform)
    if dataset_name == "IDRID":
        test_dataset = LoadDataset(IDRID_image_folder, IDRID_csv_file, transform=test_transform)
    if dataset_name == 'APTOS_15_test':
        test_dataset = LoadDataset(APTOS_15_test_image_folder, APTOS_15_test_csv_file, transform=test_transform)
    # Get class names
    class_names = ['No DR', 'DR']

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    return test_dataloader, class_names

def create_train_dataloader(
    train_transform,
    batch_size: int,
    shrink_size, 
    num_workers: int=NUM_WORKERS,
    ):
    
    train_dataset, _, _ = LoadDataset_train_val_test_split(transform=train_transform, shrink_size=shrink_size)

    # Get class names
    class_names = ['No DR', 'DR']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    return train_dataloader, class_names