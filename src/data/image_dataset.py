import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataloaders = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if split == "train":
            transform = train_transform
            shuffle = True
        else:
            transform = eval_transform
            shuffle = False

        dataset = datasets.ImageFolder(split_dir, transform=transform)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return dataloaders

# Usage:
# dataloaders = get_dataloaders("data/raw/cats_vs_dogs")
