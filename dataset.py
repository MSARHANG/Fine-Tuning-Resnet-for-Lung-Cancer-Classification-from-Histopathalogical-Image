import torch
from torch.utils.data import Dataset 

import os
from PIL import Image 



class LungCancerDataset(Dataset):
    
    def __init__(self, root_dir, transforms=None):
        
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        
        for label, cls_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)
            
                
    def __len__(self):
        
        return len(self.image_paths)
    
    
    def __get_item__(self, idx):
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label
    