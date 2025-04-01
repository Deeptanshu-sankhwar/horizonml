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

# Custom ResNet implementation that can be split across processes
class SplitResNet(nn.Module):
    def __init__(self, num_workers=5):
        super(SplitResNet, self).__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.num_workers = num_workers
        self.segments = self._split_model(resnet, num_workers)
        
    def _split_model(self, model, num_workers):
        layers = []
        
        layers.append(nn.Sequential(OrderedDict([
            ('conv1', model.conv1),
            ('bn1', model.bn1),
            ('relu', model.relu),
            ('maxpool', model.maxpool)
        ])))
        
        layers.append(model.layer1)
        layers.append(model.layer2)
        layers.append(model.layer3)
        layers.append(nn.Sequential(OrderedDict([
            ('layer4', model.layer4),
            ('avgpool', model.avgpool),
            ('flatten', nn.Flatten()),
            ('fc', model.fc)
        ])))
        
        if len(layers) < num_workers:
            raise ValueError(f"Cannot split ResNet into {num_workers} segments. Maximum is {len(layers)}.")
        
        segments = []
        layers_per_segment = len(layers) // num_workers
        remainder = len(layers) % num_workers
        
        start_idx = 0
        for i in range(num_workers):
            end_idx = start_idx + layers_per_segment + (1 if i < remainder else 0)
            if start_idx < end_idx:
                segment = nn.Sequential(*layers[start_idx:end_idx])
                segments.append(segment)
            else:
                segments.append(nn.Identity())
            start_idx = end_idx
            
        return segments

# Get expected output shape for each model segment
def get_expected_shape(rank, batch_size=64):
    shapes = [
        [batch_size, 64, 56, 56],    # After initial layers
        [batch_size, 64, 56, 56],    # After layer1
        [batch_size, 128, 28, 28],   # After layer2
        [batch_size, 256, 14, 14],   # After layer3
        [batch_size, 10]             # After final layer (output)
    ]
    
    if rank < len(shapes):
        return shapes[rank]
    return None

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

# Get data loader for model parallel training
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
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )

    return dataloader

# Train the model using model parallelism
def train_model_parallel(rank, world_size, epochs=5, sample_size=1000, log_interval=5, logs_dir="model_parallel_logs"):
    os.makedirs(logs_dir, exist_ok=True)
    
    batch_size = 64
    dataloader = get_dataloader(rank, world_size, batch_size, sample_size)
    
    model = SplitResNet(num_workers=world_size)
    model_segment = model.segments[rank].to(torch.device("cpu"))
    
    if rank == world_size - 1:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_segment.parameters(), lr=0.001)
    
    print(f"Worker {rank} is starting training with model segment {rank}...")
    
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
            
            try:
                if rank == 0:
                    compute_start = time.time()
                    output = model_segment(images)
                    compute_time = time.time() - compute_start
                    total_compute_time += compute_time
                    
                    comm_start = time.time()
                    
                    output_size = torch.tensor(output.size())
                    dist.send(output_size, dst=rank+1)
                    dist.send(output, dst=rank+1)
                    
                    comm_time = time.time() - comm_start
                    total_comm_time += comm_time
                    
                    data_sent = output.element_size() * output.nelement()
                    bandwidth_usage.append(data_sent)
                    
                elif rank < world_size - 1:
                    comm_start = time.time()
                    
                    output_size = torch.zeros(4, dtype=torch.long)
                    dist.recv(output_size, src=rank-1)
                    
                    output = torch.zeros(output_size.tolist())
                    dist.recv(output, src=rank-1)
                    
                    comm_time = time.time() - comm_start
                    total_comm_time += comm_time
                    
                    compute_start = time.time()
                    output = model_segment(output)
                    compute_time = time.time() - compute_start
                    total_compute_time += compute_time
                    
                    comm_start = time.time()
                    
                    output_size = torch.tensor(output.size())
                    dist.send(output_size, dst=rank+1)
                    dist.send(output, dst=rank+1)
                    
                    comm_time = time.time() - comm_start
                    total_comm_time += comm_time
                    
                    data_sent = output.element_size() * output.nelement()
                    bandwidth_usage.append(data_sent)
                    
                else:  # Last worker
                    comm_start = time.time()
                    
                    output_size = torch.zeros(4, dtype=torch.long)
                    dist.recv(output_size, src=rank-1)
                    
                    output = torch.zeros(output_size.tolist())
                    dist.recv(output, src=rank-1)
                    
                    comm_time = time.time() - comm_start
                    total_comm_time += comm_time
                    
                    compute_start = time.time()
                    
                    output = model_segment(output)
                    loss = criterion(output, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    current_grads = []
                    for param in model_segment.parameters():
                        if param.grad is not None:
                            current_grads.append(param.grad.data.clone().flatten())
                    
                    if prev_grads is not None and current_grads:
                        prev_concat = torch.cat(prev_grads)
                        current_concat = torch.cat(current_grads)
                        
                        if prev_concat.shape == current_concat.shape:
                            divergence = torch.norm(prev_concat - current_concat).item()
                            grad_divergences.append(divergence)
                    
                    prev_grads = current_grads
                    
                    optimizer.step()
                    
                    compute_time = time.time() - compute_start
                    total_compute_time += compute_time
                    
                    total_loss += loss.item()
                    
                    _, predicted = torch.max(output, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
            except Exception as e:
                print(f"Error in worker {rank}, batch {batch_idx}: {str(e)}")
                raise e
        
        if rank == world_size - 1:
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
        else:
            log_data.append({
                "epoch": epoch+1,
                "loss": 0,
                "accuracy": 0,
                "epoch_time": time.time() - start_time,
                "avg_step_time": sum(step_times) / len(step_times) if step_times else 0,
                "compute_time": total_compute_time,
                "comm_time": total_comm_time,
                "idle_time": total_idle_time,
                "avg_cpu": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                "avg_memory": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                "avg_bandwidth": sum(bandwidth_usage) / len(bandwidth_usage) if bandwidth_usage else 0,
                "grad_divergence": 0
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
def worker(rank, world_size, epochs, sample_size, port, logs_dir="model_parallel_logs"):
    try:
        setup_distributed(rank, world_size, port)
        
        train_model_parallel(rank, world_size, epochs, sample_size, logs_dir=logs_dir)
        
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

# Run model parallel training with multiple processes
def run_model_parallel(world_size=5, epochs=5, sample_size=1000, logs_dir="model_parallel_logs"):
    os.makedirs(logs_dir, exist_ok=True)
    
    mp.set_start_method('spawn', force=True)
    
    port = find_free_port()
    print(f"Using port {port} for model parallel training")
    
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
    
    print(f"Model-parallel training completed in {training_duration:.2f} seconds")
    
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
    
    run_model_parallel(args.world_size, args.epochs, args.sample_size) 