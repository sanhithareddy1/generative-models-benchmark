from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(dataset_name="MNIST", batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
