import os
import time
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

import datetime  # Import datetime module

# Step 1: Initialize distributed process group
def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # Ensure only one worker downloads the model first
    if rank == 0:
        print("Worker 0 is downloading model weights...")
        torchvision.models.mobilenet_v2(weights=None)  
        torchvision.models.resnet18(weights=None)
        print("Model weights downloaded.")

    # Fix: Use datetime.timedelta instead of torch.distributed.timedelta
    dist.init_process_group(
        backend="gloo", 
        rank=rank, 
        world_size=world_size, 
        timeout=datetime.timedelta(seconds=300)  # Correct way to set timeout
    )

    print(f"Process {rank} initialized in a world of {world_size} workers, connecting to {master_addr}:{master_port}")

    dist.barrier()  # Ensure synchronization before training starts


# Step 2: Set up dataset (CIFAR-10)
def get_dataloader(batch_size=2):  # Reduce batch size to 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    # Reduce workers and enable pin_memory for lower memory usage
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)  
    return dataloader


# Step 3: Define the model
def get_model(model_type="resnet"):
    if model_type == "mobilenet":
        model = torchvision.models.mobilenet_v2(weights=None)  # Fix deprecated pretrained argument
    else:
        model = torchvision.models.resnet18(weights=None)

    model = model.to(torch.device("cpu"))
    model = DDP(model)
    return model


# Step 4: Training function with logging
def train(model, dataloader, epochs=5):
    rank = int(os.environ["RANK"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Worker {rank} is starting training...")  # Debugging log

    log_data = []  # Store logs for later analysis

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to("cpu"), labels.to("cpu")  

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Accuracy Calculation
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        epoch_time = time.time() - start_time  # Time taken for epoch

        print(f"Worker {rank}, Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_time:.2f}s")

        # Save logs
        log_data.append([rank, epoch+1, epoch_loss, epoch_accuracy, epoch_time])

        # Synchronize workers
        dist.barrier()

    # Convert log data to Pandas DataFrame and save as CSV
    df = pd.DataFrame(log_data, columns=["Worker", "Epoch", "Loss", "Accuracy", "Time"])
    df.to_csv(f"training_logs_worker_{rank}.csv", index=False)

# Step 5: Run distributed training
if __name__ == "__main__":
    setup_distributed()
    dataloader = get_dataloader()

    model_type = os.getenv("MODEL_TYPE", "resnet")  
    model = get_model(model_type)

    train(model, dataloader)
