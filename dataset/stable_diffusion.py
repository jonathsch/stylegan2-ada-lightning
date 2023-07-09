import os
from typing import List

import numpy as np
import torch
import torchvision
from pathlib import Path
from PIL import Image


class StableDiffusionShapeNetDataset(torch.utils.data.Dataset):
    def __init__(
        self, root_dir, image_size, size=None, shapenet_split_file: os.PathLike = None
    ):
        super(StableDiffusionShapeNetDataset, self).__init__()
        
        if shapenet_split_file is not None:
            with open(shapenet_split_file, "r") as f:
                shape_dirs = [Path(root_dir, p.strip()) for p in f.readlines()]
        else:
            shape_dirs = [p for p in Path(root_dir).iterdir() if p.is_dir()]
        
        self.img_paths: List[Path] = []
        for shape_dir in shape_dirs:
            img_folder = shape_dir / "images"
            self.img_paths += [
                p
                for p in img_folder.iterdir()
                if p.is_file() and p.name.endswith(".png")
            ]
        self.size = min(
            size if size is not None else len(self.img_paths), len(self.img_paths)
        )
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.ToTensor(),
            ]
        )


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = img_path.parent.parent / "masks" / img_path.name
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        background = Image.new("RGB", img.size, (255, 255, 255))
        img = Image.composite(img, background, mask)

        img.save("test.png")
        return {"image": self.transform(img) * 2 - 1}


if __name__ == "__main__":
    ds = StableDiffusionShapeNetDataset("/storage/user/schmidjo/shapenet-sd-photo-v2/02958343", 512)
    batch = ds[0]
    print(type(batch))