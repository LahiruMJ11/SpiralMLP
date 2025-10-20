# prepare_cifar.py
from torchvision import datasets, transforms
datasets.CIFAR10(root='./.data', train=True, download=True)
datasets.CIFAR10(root='./.data', train=False, download=True)
print("CIFAR-10 downloaded to ./.data")
