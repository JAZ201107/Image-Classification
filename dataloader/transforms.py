import torchvision.transforms as transforms


train_transformer = transforms.Compose(
    [
        transforms.Resize(64),  # Resize Image to (64, 64)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


eval_transformer = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
