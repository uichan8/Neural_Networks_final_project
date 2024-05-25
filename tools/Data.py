import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image

class muffin_chihuahua(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.muffin_paths = [os.path.join(root,'muffin', img) for img in os.listdir(os.path.join(root,'muffin'))]
        self.chihuahua_paths = [os.path.join(root,'chihuahua', img) for img in os.listdir(os.path.join(root,'chihuahua'))]
        self.target = torch.zeros(len(self.muffin_paths) + len(self.chihuahua_paths))
        self.target[:len(self.muffin_paths)] = 1. #치와와:1 머핀:0
        self.target = self.target.long()
        self.img_path = self.muffin_paths + self.chihuahua_paths
        self.transform = transform
        self.basic_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"muffin: {len(self.muffin_paths)}")
        print(f"chihuahua: {len(self.chihuahua_paths)}")

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        x = Image.open(self.img_path[index])
        if x.mode != 'RGB':
            x = x.convert('RGB')
        x = self.basic_transform(x)
        if self.transform:
            x = self.transform(x)

        y = self.target[index]

        return x, y
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = muffin_chihuahua(root="data/test")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for x, y in loader:
        print(x.shape)
        print(y)
        break