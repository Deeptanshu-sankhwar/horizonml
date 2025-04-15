import os
import time
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import datetime
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm
from collections import OrderedDict
import socket
import random
import psutil
import sys

# Find a free port for distributed communication
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# Custom implementation of tensor-parallel linear layer
class TensorParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        super(TensorParallelLinear, self).__init__()
        
        # Split the output features across workers
        self.out_features_per_worker = out_features // world_size
        self.rank = rank
        self.world_size = world_size
        
        # Each worker gets a slice of the weight matrix
        self.linear = nn.Linear(in_features, self.out_features_per_worker, bias=bias)
        
    def forward(self, x):
        # Each worker computes its portion of the output
        local_output = self.linear(x)
        
        # We'll use a different approach that's more autograd-friendly
        # First, create a tensor of the right size on each worker
        output_size = list(local_output.size())
        output_size[1] = self.out_features_per_worker * self.world_size
        
        # Create a zero tensor with the right size
        full_output = torch.zeros(output_size, device=local_output.device)
        
        # Each worker will broadcast their part to all others
        for i in range(self.world_size):
            # Calculate the slice for this worker's output
            start_idx = i * self.out_features_per_worker
            end_idx = start_idx + self.out_features_per_worker
            
            # If this is our rank, we already have the data
            if i == self.rank:
                full_output[:, start_idx:end_idx] = local_output
            
            # Synchronize this slice across all workers
            dist.broadcast(full_output[:, start_idx:end_idx], src=i)
        
        return full_output

# Custom ResNet implementation with tensor parallelism
class TensorParallelResNet(nn.Module):
    def __init__(self, world_size, rank):
        super(TensorParallelResNet, self).__init__()
        self.world_size = world_size
        self.rank = rank
        
        # Load a standard ResNet18 model
        resnet = torchvision.models.resnet18(weights=None)
        
        # Keep most of the model as is
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Replace the final fully connected layer with a tensor-parallel version
        self.fc = TensorParallelLinear(resnet.fc.in_features, 10, world_size, rank)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Set up the distributed process group
def setup_distributed(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=300)
    )
    
    print(f"Process {rank} initialized in a world of {world_size} workers on port {port}")
    dist.barrier()

# Get data loader for tensor parallel training
def get_dataloader(rank, world_size, batch_size=64, sample_size=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_path = "./data"

    if rank == 0:
        torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        print("Worker 0 downloaded the dataset.")

    dist.barrier()

    dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=transform)
    
    if sample_size and sample_size < len(dataset):
        indices = torch.randperm(len(dataset))[:sample_size]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # All workers use the same data
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )

    return dataloader

# Train the model using tensor parallelism
def train_tensor_parallel(rank, world_size, epochs=5, sample_size=1000, log_interval=5, logs_dir="tensor_parallel_logs"):
    os.makedirs(logs_dir, exist_ok=True)
    
    batch_size = 64
    dataloader = get_dataloader(rank, world_size, batch_size, sample_size)
    
    # Create tensor-parallel model
    model = TensorParallelResNet(world_size, rank).to(torch.device("cpu"))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Worker {rank} is starting training with tensor-parallel model...")
    
    log_data = []
    
    total_comm_time = 0
    total_compute_time = 0
    total_idle_time = 0
    step_times = []
    memory_usage = []
    cpu_usage = []
    bandwidth_usage = []
    
    prev_grads = None
    grad_divergences = []
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        idle_start = time.time()
        dist.barrier()
        total_idle_time += time.time() - idle_start
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Worker {rank}, Epoch {epoch+1}")):
            step_start = time.time()
            
            cpu_usage.append(psutil.cpu_percent(interval=None))
            memory_usage.append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
            
            images = images.to("cpu")
            labels = labels.to("cpu")
            
            # Forward pass
            compute_start = time.time()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            
            comm_start = time.time()
            loss.backward()
            compute_time = comm_start - compute_start
            total_compute_time += compute_time
            
            # Synchronize gradients across workers
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= world_size
            
            optimizer.step()
            
            comm_time = time.time() - comm_start
            total_comm_time += comm_time
            
            # Calculate bandwidth usage (approximate)
            data_sent = sum(p.grad.element_size() * p.grad.nelement() for p in model.parameters() if p.grad is not None)
            bandwidth_usage.append(data_sent)
            
            # Track gradient divergence
            current_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    current_grads.append(param.grad.data.clone().flatten())
            
            if prev_grads is not None and current_grads:
                prev_concat = torch.cat(prev_grads)
                current_concat = torch.cat(current_grads)
                
                if prev_concat.shape == current_concat.shape:
                    divergence = torch.norm(prev_concat - current_concat).item()
                    grad_divergences.append(divergence)
            
            prev_grads = current_grads
            
            # Calculate metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # Synchronize workers
            idle_start = time.time()
            dist.barrier()
            total_idle_time += time.time() - idle_start
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_time:.2f}s")
        
        log_data.append({
            "epoch": epoch+1,
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "epoch_time": epoch_time,
            "avg_step_time": sum(step_times) / len(step_times) if step_times else 0,
            "compute_time": total_compute_time,
            "comm_time": total_comm_time,
            "idle_time": total_idle_time,
            "avg_cpu": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
            "avg_memory": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "avg_bandwidth": sum(bandwidth_usage) / len(bandwidth_usage) if bandwidth_usage else 0,
            "grad_divergence": sum(grad_divergences) / len(grad_divergences) if grad_divergences else 0
        })
        
        step_times = []
        cpu_usage = []
        memory_usage = []
        bandwidth_usage = []
        
        dist.barrier()
        
        df = pd.DataFrame(log_data)
        df.to_csv(f"{logs_dir}/worker_{rank}_samples_{sample_size}.csv", index=False)
    
    try:
        dist.barrier()
    except Exception as e:
        print(f"Warning: Barrier error during cleanup: {e}")
    
    return df

# Worker process function for distributed training
def worker(rank, world_size, epochs, sample_size, port, logs_dir="tensor_parallel_logs"):
    try:
        setup_distributed(rank, world_size, port)
        
        train_tensor_parallel(rank, world_size, epochs, sample_size, logs_dir=logs_dir)
        
        completion_tensor = torch.zeros(1)
        for other_rank in range(world_size):
            if other_rank != rank:
                try:
                    dist.send(completion_tensor, dst=other_rank)
                except:
                    pass
        
        try:
            dist.barrier(timeout=datetime.timedelta(seconds=5))
            dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: Error during cleanup for worker {rank}: {e}")
        
        print(f"Worker {rank} completed successfully")
        
    except Exception as e:
        print(f"Exception in worker {rank}: {str(e)}")
    
    sys.exit(0)

# Run tensor parallel training with multiple processes
def run_tensor_parallel(world_size=5, epochs=5, sample_size=1000, logs_dir="tensor_parallel_logs"):
    os.makedirs(logs_dir, exist_ok=True)
    
    mp.set_start_method('spawn', force=True)
    
    port = find_free_port()
    print(f"Using port {port} for tensor parallel training")
    
    # Record the actual training start time
    training_start_time = time.time()
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, epochs, sample_size, port, logs_dir))
        p.start()
        processes.append(p)
    
    # Calculate a dynamic timeout based on sample size
    # Increase the base timeout significantly to handle longer training times
    max_wait_time = max(400, int(400 * (sample_size / 1000)))
    print(f"Using timeout of {max_wait_time} seconds for sample size {sample_size}")
    
    wait_start_time = time.time()
    
    for p in processes:
        remaining_time = max(0, max_wait_time - (time.time() - wait_start_time))
        p.join(timeout=remaining_time)
    
    # Record the actual training end time before cleanup
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    # Cleanup processes - this shouldn't count toward training time
    for p in processes:
        if p.is_alive():
            print(f"Forcefully terminating process {p.pid}")
            p.terminate()
            time.sleep(1)
    
    print(f"Tensor-parallel training completed in {training_duration:.2f} seconds")
    
    time.sleep(2)
    
    all_results = []
    for rank in range(world_size):
        try:
            df = pd.read_csv(f"{logs_dir}/worker_{rank}_samples_{sample_size}.csv")
            df['worker'] = rank
            all_results.append(df)
        except Exception as e:
            print(f"Could not read results for worker {rank}: {e}")
    
    if all_results:
        combined_df = pd.concat(all_results)
        # Add the actual total training time to the results
        combined_df['total_training_time'] = training_duration
        combined_df.to_csv(f"{logs_dir}/combined_results_{sample_size}.csv", index=False)
        return combined_df
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=5, help='Number of processes to use')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of samples to use')
    args = parser.parse_args()
    
    run_tensor_parallel(args.world_size, args.epochs, args.sample_size) 