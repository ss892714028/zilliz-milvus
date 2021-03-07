import torch
import torchvision
from Net import Net
import torch.optim as optim
import train
from Transformer import FeatureVector


root = 'data\\'
network = Net()
network.load_state_dict(torch.load('results/model.pth'))

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(root, train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), shuffle=True)