from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, "rb") as f:
            _, img_num, img_row, img_column = struct.unpack(">4i", f.read(16))
            X = np.frombuffer(f.read(img_num * img_row * img_column), dtype=np.uint8).astype(np.float32).reshape(img_num, img_row * img_column)
            X -= np.min(X)
            X /= np.max(X)
            self.X = X

        with gzip.open(label_filename, "rb") as f:
            _, label_num = struct.unpack(">2i", f.read(8))
            y = np.frombuffer(f.read(label_num), dtype=np.uint8)
            self.y = y

        self.transforms = transforms       
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.X[index]
        labels = self.y[index]
        
        if len(imgs.shape) == 1:
            imgs = np.expand_dims(imgs, 0)
            
        imgs = np.array([self.apply_transforms(img.reshape(28, 28, 1)).reshape(-1) for img in imgs])
        
        if imgs.shape[0] == 1:
            imgs = imgs[0]
        
        return imgs, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION