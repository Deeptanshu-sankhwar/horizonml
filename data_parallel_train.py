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
from torch.utils.data.distributed import DistributedSampler
import datetime
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm
import socket
import random
import psutil
import sys

# Find a free port for distributed communication
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

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

# Get data loader with distributed sampling
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
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=0, 
        pin_memory=True
    )

    return dataloader, sampler

# Train the model using data parallelism
def train(rank, model, dataloader, sampler, optimizer, criterion, epochs=5, log_interval=5, logs_dir="data_parallel_logs"):
    os.makedirs(logs_dir, exist_ok=True)
    
    log_data = []
    
    total_comm_time = 0
    total_compute_time = 0
    total_idle_time = 0
    step_times = []
    memory_usage = []
    cpu_usage = []
    
    prev_grads = None
    grad_divergences = []
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        idle_start = time.time()
        dist.barrier()
        total_idle_time += time.time() - idle_start
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Worker {rank}, Epoch {epoch+1}")):
            step_start = time.time()
            
            cpu_usage.append(psutil.cpu_percent(interval=None))
            memory_usage.append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)  # MB
            
            images = images.to("cpu")
            labels = labels.to("cpu")
            
            compute_start = time.time()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            comm_start = time.time()
            loss.backward()
            compute_time = comm_start - compute_start
            total_compute_time += compute_time
            
            optimizer.step()
            comm_time = time.time() - comm_start
            total_comm_time += comm_time
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
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
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            idle_start = time.time()
            dist.barrier()
            total_idle_time += time.time() - idle_start
        
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        
        if rank == 0:
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
            "grad_divergence": sum(grad_divergences) / len(grad_divergences) if grad_divergences else 0
        })
        
        step_times = []
        cpu_usage = []
        memory_usage = []
        
        df = pd.DataFrame(log_data)
        df.to_csv(f"{logs_dir}/worker_{rank}_samples_{len(dataloader.dataset)}.csv", index=False)
        
        dist.barrier()
    
    try:
        dist.barrier()
    except Exception as e:
        print(f"Warning: Barrier error during cleanup: {e}")
    
    return df

# Worker process function for distributed training
def worker(rank, world_size, epochs, sample_size, port, logs_dir="data_parallel_logs"):
    try:
        setup_distributed(rank, world_size, port)
        
        dataloader, sampler = get_dataloader(rank, world_size, batch_size=64, sample_size=sample_size)
        
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model = model.to("cpu")
        
        model = DDP(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print(f"Worker {rank} is starting training...")
        
        train(rank, model, dataloader, sampler, optimizer, criterion, epochs, logs_dir=logs_dir)
        
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

# Run data parallel training with multiple processes
def run_data_parallel(world_size=5, epochs=5, sample_size=1000, logs_dir="data_parallel_logs"):
    os.makedirs(logs_dir, exist_ok=True)
    
    mp.set_start_method('spawn', force=True)
    
    port = find_free_port()
    print(f"Using port {port} for data parallel training")
    
    # Record the actual training start time
    training_start_time = time.time()
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, epochs, sample_size, port, logs_dir))
        p.start()
        processes.append(p)
    
    # Calculate a dynamic timeout based on sample size
    # Base timeout of 120 seconds for 1000 samples, scaling linearly
    max_wait_time = max(120, int(120 * (sample_size / 1000)))
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
    
    print(f"Data-parallel training completed in {training_duration:.2f} seconds")
    
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
    
    run_data_parallel(args.world_size, args.epochs, args.sample_size)
