import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from metalearning.model import SimpleMLP
from metalearning.regularizer import StructuralPlasticityOptimizer
import time
import matplotlib.pyplot as plt
import numpy as np

def train(model, device, train_loader, optimizer, epoch, regularizer=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()
        optimizer.step()
        
        # Apply Zero Mask
        if regularizer:
            regularizer.apply_mask()
            
            # Perform Plasticity every 100 batches
            if batch_idx % 100 == 0:
                regularizer.step()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return acc

import torch.nn.functional as F

def main():
    # Settings
    batch_size = 64
    epochs = 5
    device = torch.device("cpu")
    
    # Data (Use Fake Data if MNIST download fails or is slow? No, let's try download)
    # Actually, for speed/stability in this env, let's use generated random data 
    # if we can't access internet easily. But usually we can.
    # Let's try standard MNIST.
    
    try:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=1000, shuffle=True)
    except Exception as e:
        print(f"Failed to download MNIST: {e}. Using Fake Data.")
        # Create fake dataset
        x_train = torch.randn(1000, 1, 28, 28)
        y_train = torch.randint(0, 10, (1000,))
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size)
        test_loader = train_loader
        
    print("--- Baseline Training ---")
    model_base = SimpleMLP().to(device)
    optimizer_base = optim.Adam(model_base.parameters(), lr=0.001)
    
    acc_base = []
    for epoch in range(1, epochs + 1):
        train(model_base, device, train_loader, optimizer_base, epoch)
        acc = test(model_base, device, test_loader)
        acc_base.append(acc)
        print(f"Epoch {epoch}: {acc:.2f}%")
        
    print("\n--- Bio-Regularized Training ---")
    model_bio = SimpleMLP().to(device)
    optimizer_bio = optim.Adam(model_bio.parameters(), lr=0.001)
    regularizer = StructuralPlasticityOptimizer(model_bio, pruning_rate=0.05, regrowth_rate=0.05)
    
    acc_bio = []
    for epoch in range(1, epochs + 1):
        train(model_bio, device, train_loader, optimizer_bio, epoch, regularizer)
        acc = test(model_bio, device, test_loader)
        acc_bio.append(acc)
        sparsity = 1.0 - (regularizer.masks['fc1.weight'].sum().item() / regularizer.masks['fc1.weight'].numel())
        print(f"Epoch {epoch}: {acc:.2f}% (Sparsity FC1: {sparsity:.2f})")

    # Plot Comparison
    plt.figure()
    plt.plot(acc_base, label='Baseline')
    plt.plot(acc_bio, label='Bio-Regularized')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Meta-Learning Benchmark')
    plt.savefig('viz/benchmark_result.png')
    print("Saved viz/benchmark_result.png")

if __name__ == "__main__":
    main()
