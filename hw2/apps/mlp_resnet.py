import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

from needle.data import DataLoader
from needle.data.datasets import MNISTDataset

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    block =  nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
        )
    return nn.Sequential(nn.Residual(block), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.train() if opt else model.eval()
    tot_loss, tot_error = 0.0, 0.0        
    softmax = nn.SoftmaxLoss()                

    for X, y in dataloader:
        logits = model(X)                 
        loss = softmax(logits, y)            
        tot_loss += loss.numpy() * X.shape[0]  
        tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())  

        if opt:  
            opt.reset_grad()  
            loss.backward()  
            opt.step()        

    sample_nums = len(dataloader.dataset)  
    avg_loss = tot_loss / sample_nums      
    error_rate = tot_error / sample_nums   
    
    return error_rate, avg_loss          
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model = MLPResNet(dim=28*28, hidden_dim=hidden_dim, num_blocks=3, num_classes=10)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = MNISTDataset(
        image_filename=f"{data_dir}/train-images-idx3-ubyte.gz",
        label_filename=f"{data_dir}/train-labels-idx1-ubyte.gz"
    )
    test_set = MNISTDataset(
        image_filename=f"{data_dir}/t10k-images-idx3-ubyte.gz",
        label_filename=f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    for epoch_num in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)
        print(f"Epoch {epoch_num + 1}/{epochs} - Train Error: {train_err:.4f}, Loss: {train_loss:.4f}")
    
    test_err, test_loss = epoch(test_loader, model)
    print(f"Test Error: {test_err:.4f}, Loss: {test_loss:.4f}")
    
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")