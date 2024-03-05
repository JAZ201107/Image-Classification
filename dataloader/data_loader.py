import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader


from .transforms import train_transformer, eval_transformer


class SIGNDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.filenames = os.listdir(data_dir)
        self.filenames = [
            os.path.join(data_dir, f) for f in self.filenames if f.endswith(".jpg")
        ]

        self.labels = [
            int(os.path.split(filename)[-1][0]) for filename in self.filenames
        ]
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        image = self.transform(image)

        return image, self.labels[idx]

    def __len__(self):
        return len(self.filenames)


def fetch_dataloader(types, data_dir, params):
    dataloaders = {}

    for split in ["train", "val", "test"]:
        if split in types:
            path = os.path.join(data_dir, f"{split}_signs")

            if split == "train":
                dl = DataLoader(
                    SIGNDataset(path, train_transformer),
                    batch_size=params.batch_size,
                    shuffle=True,
                    num_workers=params.num_workers,
                    pin_memory=params.cuda,
                )
            else:
                dl = DataLoader(
                    SIGNDataset(path, eval_transformer),
                    batch_size=params.batch_size,
                    shuffle=False,
                    num_workers=params.num_workers,
                    pin_memory=params.cuda,
                )

            dataloaders[split] = dl

    return dataloaders
