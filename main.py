import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from data_parallel_train import run_data_parallel
from layer_model_parallel_train import run_model_parallel
from tensor_parallel_train import run_tensor_parallel

# Set up plotting style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Run benchmarks for all three parallelism approaches
def run_benchmarks(sample_sizes=[1000, 10000, 50000], world_size=5, epochs=5):
    results = {
        'data_parallel': {},
        'model_parallel': {},
        'tensor_parallel': {}
    }
    
    for sample_size in sample_sizes:
        print(f"\n{'='*50}")
        print(f"Running benchmarks with {sample_size} samples")
        print(f"{'='*50}")
        
        print(f"\nStarting Data Parallel training with {sample_size} samples...")
        data_parallel_results = run_data_parallel(world_size, epochs, sample_size)
        # Get the actual training time from the results
        data_parallel_time = data_parallel_results['total_training_time'].iloc[0]
        print(f"Data Parallel training completed in {data_parallel_time:.2f} seconds")
        
        print(f"\nStarting Model Parallel training with {sample_size} samples...")
        model_parallel_results = run_model_parallel(world_size, epochs, sample_size)
        # Get the actual training time from the results
        model_parallel_time = model_parallel_results['total_training_time'].iloc[0]
        print(f"Model Parallel training completed in {model_parallel_time:.2f} seconds")
        
        print(f"\nStarting Tensor Parallel training with {sample_size} samples...")
        tensor_parallel_results = run_tensor_parallel(world_size, epochs, sample_size)
        
        # Check if tensor parallel training completed successfully
        if tensor_parallel_results is not None:
            tensor_parallel_time = tensor_parallel_results['total_training_time'].iloc[0]
            print(f"Tensor Parallel training completed in {tensor_parallel_time:.2f} seconds")
            
            # Check if all epochs were completed
            max_epoch = tensor_parallel_results['epoch'].max()
            if max_epoch < epochs:
                print(f"Warning: Tensor Parallel training only completed {max_epoch} out of {epochs} epochs")
        else:
            print("Tensor Parallel training failed to complete")
            tensor_parallel_time = 0
        
        results['data_parallel'][sample_size] = data_parallel_results
        results['model_parallel'][sample_size] = model_parallel_results
        results['tensor_parallel'][sample_size] = tensor_parallel_results
    
    return results

# Generate comparison graphs from benchmark results
def generate_comparison_graphs(results, output_dir="benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    sample_sizes = list(results['data_parallel'].keys())
    world_size = 5
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(12, 8))
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size][results['model_parallel'][sample_size]['worker'] == world_size-1]
        tp_data = results['tensor_parallel'][sample_size] if results['tensor_parallel'][sample_size] is not None else None
        
        plt.plot(dp_data.groupby('epoch')['accuracy'].mean(), 
                 label=f'Data Parallel ({sample_size} samples)', 
                 marker='o')
        plt.plot(mp_data['epoch'], mp_data['accuracy'], 
                 label=f'Model Parallel ({sample_size} samples)', 
                 marker='x')
        if tp_data is not None:
            plt.plot(tp_data.groupby('epoch')['accuracy'].mean(), 
                     label=f'Tensor Parallel ({sample_size} samples)', 
                     marker='s')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison: Data vs Model vs Tensor Parallel')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_comparison.png")
    
    # 2. Loss Comparison
    plt.figure(figsize=(12, 8))
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size][results['model_parallel'][sample_size]['worker'] == world_size-1]
        tp_data = results['tensor_parallel'][sample_size] if results['tensor_parallel'][sample_size] is not None else None
        
        plt.plot(dp_data.groupby('epoch')['loss'].mean(), 
                 label=f'Data Parallel ({sample_size} samples)', 
                 marker='o')
        plt.plot(mp_data['epoch'], mp_data['loss'], 
                 label=f'Model Parallel ({sample_size} samples)', 
                 marker='x')
        if tp_data is not None:
            plt.plot(tp_data.groupby('epoch')['loss'].mean(), 
                     label=f'Tensor Parallel ({sample_size} samples)', 
                     marker='s')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison: Data vs Model vs Tensor Parallel')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_comparison.png")
    
    # 3. Training Time Comparison
    plt.figure(figsize=(12, 8))
    dp_times = []
    mp_times = []
    tp_times = []
    
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size][results['model_parallel'][sample_size]['worker'] == world_size-1]
        tp_data = results['tensor_parallel'][sample_size] if results['tensor_parallel'][sample_size] is not None else None
        
        dp_times.append(dp_data['epoch_time'].mean())
        mp_times.append(mp_data['epoch_time'].mean())
        if tp_data is not None:
            tp_times.append(tp_data['epoch_time'].mean())
        else:
            tp_times.append(0)
    
    width = 0.25
    x = np.arange(len(sample_sizes))
    
    plt.bar(x - width, dp_times, width, label='Data Parallel')
    plt.bar(x, mp_times, width, label='Model Parallel')
    plt.bar(x + width, tp_times, width, label='Tensor Parallel')
    
    plt.xlabel('Sample Size')
    plt.ylabel('Average Epoch Time (s)')
    plt.title('Training Time Comparison')
    plt.xticks(x, sample_sizes)
    plt.legend()
    plt.savefig(f"{output_dir}/training_time_comparison.png")
    
    # 4. Communication vs Computation Time
    plt.figure(figsize=(18, 6))
    
    sample_labels = []
    dp_compute = []
    dp_comm = []
    mp_compute = []
    mp_comm = []
    tp_compute = []
    tp_comm = []
    
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size]
        tp_data = results['tensor_parallel'][sample_size] if results['tensor_parallel'][sample_size] is not None else None
        
        sample_labels.append(str(sample_size))
        dp_compute.append(dp_data['compute_time'].mean())
        dp_comm.append(dp_data['comm_time'].mean())
        mp_compute.append(mp_data['compute_time'].mean())
        mp_comm.append(mp_data['comm_time'].mean())
        if tp_data is not None:
            tp_compute.append(tp_data['compute_time'].mean())
            tp_comm.append(tp_data['comm_time'].mean())
        else:
            tp_compute.append(0)
            tp_comm.append(0)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.bar(sample_labels, dp_compute, label='Compute Time')
    ax1.bar(sample_labels, dp_comm, bottom=dp_compute, label='Communication Time')
    ax1.set_title('Data Parallel: Compute vs Communication Time')
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Time (s)')
    ax1.legend()
    
    ax2.bar(sample_labels, mp_compute, label='Compute Time')
    ax2.bar(sample_labels, mp_comm, bottom=mp_compute, label='Communication Time')
    ax2.set_title('Model Parallel: Compute vs Communication Time')
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Time (s)')
    ax2.legend()
    
    ax3.bar(sample_labels, tp_compute, label='Compute Time')
    ax3.bar(sample_labels, tp_comm, bottom=tp_compute, label='Communication Time')
    ax3.set_title('Tensor Parallel: Compute vs Communication Time')
    ax3.set_xlabel('Sample Size')
    ax3.set_ylabel('Time (s)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/compute_vs_comm_comparison.png")
    
    # 5. CPU Utilization
    plt.figure(figsize=(12, 8))
    
    dp_cpu = []
    mp_cpu = []
    tp_cpu = []
    
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size]
        tp_data = results['tensor_parallel'][sample_size] if results['tensor_parallel'][sample_size] is not None else None
        
        dp_cpu.append(dp_data['avg_cpu'].mean())
        mp_cpu.append(mp_data['avg_cpu'].mean())
        if tp_data is not None:
            tp_cpu.append(tp_data['avg_cpu'].mean())
        else:
            tp_cpu.append(0)
    
    width = 0.25
    x = np.arange(len(sample_sizes))
    
    plt.bar(x - width, dp_cpu, width, label='Data Parallel')
    plt.bar(x, mp_cpu, width, label='Model Parallel')
    plt.bar(x + width, tp_cpu, width, label='Tensor Parallel')
    
    plt.xlabel('Sample Size')
    plt.ylabel('Average CPU Utilization (%)')
    plt.title('CPU Utilization Comparison')
    plt.xticks(x, sample_sizes)
    plt.legend()
    plt.savefig(f"{output_dir}/cpu_utilization_comparison.png")
    
    # 6. Memory Usage
    plt.figure(figsize=(12, 8))
    
    dp_mem = []
    mp_mem = []
    tp_mem = []
    
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size]
        tp_data = results['tensor_parallel'][sample_size] if results['tensor_parallel'][sample_size] is not None else None
        
        dp_mem.append(dp_data['avg_memory'].mean())
        mp_mem.append(mp_data['avg_memory'].mean())
        if tp_data is not None:
            tp_mem.append(tp_data['avg_memory'].mean())
        else:
            tp_mem.append(0)
    
    width = 0.25
    x = np.arange(len(sample_sizes))
    
    plt.bar(x - width, dp_mem, width, label='Data Parallel')
    plt.bar(x, mp_mem, width, label='Model Parallel')
    plt.bar(x + width, tp_mem, width, label='Tensor Parallel')
    
    plt.xlabel('Sample Size')
    plt.ylabel('Average Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.xticks(x, sample_sizes)
    plt.legend()
    plt.savefig(f"{output_dir}/memory_usage_comparison.png")
    
    # 7. Idle Time
    plt.figure(figsize=(12, 8))
    
    dp_idle = []
    mp_idle = []
    tp_idle = []
    
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size]
        tp_data = results['tensor_parallel'][sample_size] if results['tensor_parallel'][sample_size] is not None else None
        
        dp_idle.append(dp_data['idle_time'].mean())
        mp_idle.append(mp_data['idle_time'].mean())
        if tp_data is not None:
            tp_idle.append(tp_data['idle_time'].mean())
        else:
            tp_idle.append(0)
    
    width = 0.25
    x = np.arange(len(sample_sizes))
    
    plt.bar(x - width, dp_idle, width, label='Data Parallel')
    plt.bar(x, mp_idle, width, label='Model Parallel')
    plt.bar(x + width, tp_idle, width, label='Tensor Parallel')
    
    plt.xlabel('Sample Size')
    plt.ylabel('Average Idle Time (s)')
    plt.title('Worker Idle Time Comparison')
    plt.xticks(x, sample_sizes)
    plt.legend()
    plt.savefig(f"{output_dir}/idle_time_comparison.png")
    
    # 8. Overall Performance Comparison (Radar Chart)
    plt.figure(figsize=(15, 10))
    
    metrics = ['Accuracy', 'Training Time', 'Compute Time', 
               'Communication Time', 'Memory Usage', 'CPU Usage']
    
    largest_sample = max(sample_sizes)
    dp_data = results['data_parallel'][largest_sample]
    mp_data = results['model_parallel'][largest_sample]
    tp_data = results['tensor_parallel'][largest_sample] if results['tensor_parallel'][largest_sample] is not None else None
    
    # Find max values for normalization
    max_accuracy = max(dp_data['accuracy'].max(), 
                       mp_data[mp_data['worker'] == world_size-1]['accuracy'].max(),
                       tp_data['accuracy'].max() if tp_data is not None else 0)
    max_epoch_time = max(dp_data['epoch_time'].mean(), 
                         mp_data['epoch_time'].mean(), 
                         tp_data['epoch_time'].mean() if tp_data is not None else 0)
    max_compute_time = max(dp_data['compute_time'].mean(), 
                           mp_data['compute_time'].mean(), 
                           tp_data['compute_time'].mean() if tp_data is not None else 0)
    max_comm_time = max(dp_data['comm_time'].mean(), 
                        mp_data['comm_time'].mean(), 
                        tp_data['comm_time'].mean() if tp_data is not None else 0)
    max_memory = max(dp_data['avg_memory'].mean(), 
                     mp_data['avg_memory'].mean(), 
                     tp_data['avg_memory'].mean() if tp_data is not None else 0)
    max_cpu = max(dp_data['avg_cpu'].mean(), 
                  mp_data['avg_cpu'].mean(), 
                  tp_data['avg_cpu'].mean() if tp_data is not None else 0)
    
    dp_metrics = [
        dp_data['accuracy'].max() / 100,
        1 - (dp_data['epoch_time'].mean() / max_epoch_time if max_epoch_time > 0 else 0),
        1 - (dp_data['compute_time'].mean() / max_compute_time if max_compute_time > 0 else 0),
        1 - (dp_data['comm_time'].mean() / max_comm_time if max_comm_time > 0 else 0),
        1 - (dp_data['avg_memory'].mean() / max_memory if max_memory > 0 else 0),
        1 - (dp_data['avg_cpu'].mean() / max_cpu if max_cpu > 0 else 0)
    ]
    
    mp_metrics = [
        mp_data[mp_data['worker'] == world_size-1]['accuracy'].max() / 100,
        1 - (mp_data['epoch_time'].mean() / max_epoch_time if max_epoch_time > 0 else 0),
        1 - (mp_data['compute_time'].mean() / max_compute_time if max_compute_time > 0 else 0),
        1 - (mp_data['comm_time'].mean() / max_comm_time if max_comm_time > 0 else 0),
        1 - (mp_data['avg_memory'].mean() / max_memory if max_memory > 0 else 0),
        1 - (mp_data['avg_cpu'].mean() / max_cpu if max_cpu > 0 else 0)
    ]
    
    tp_metrics = None
    if tp_data is not None:
        tp_metrics = [
            tp_data['accuracy'].max() / 100,
            1 - (tp_data['epoch_time'].mean() / max_epoch_time if max_epoch_time > 0 else 0),
            1 - (tp_data['compute_time'].mean() / max_compute_time if max_compute_time > 0 else 0),
            1 - (tp_data['comm_time'].mean() / max_comm_time if max_comm_time > 0 else 0),
            1 - (tp_data['avg_memory'].mean() / max_memory if max_memory > 0 else 0),
            1 - (tp_data['avg_cpu'].mean() / max_cpu if max_cpu > 0 else 0)
        ]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    dp_metrics += dp_metrics[:1]
    mp_metrics += mp_metrics[:1]
    if tp_metrics:
        tp_metrics += tp_metrics[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    ax.plot(angles, dp_metrics, 'o-', linewidth=2, label='Data Parallel')
    ax.fill(angles, dp_metrics, alpha=0.25)
    ax.plot(angles, mp_metrics, 'o-', linewidth=2, label='Model Parallel')
    ax.fill(angles, mp_metrics, alpha=0.25)
    if tp_metrics:
        ax.plot(angles, tp_metrics, 'o-', linewidth=2, label='Tensor Parallel')
        ax.fill(angles, tp_metrics, alpha=0.25)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title(f'Overall Performance Comparison ({largest_sample} samples)')
    ax.legend(loc='upper right')
    
    plt.savefig(f"{output_dir}/overall_performance_comparison.png")
    
    print(f"All comparison graphs saved to {output_dir} directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks for distributed training')
    parser.add_argument('--sample_sizes', type=int, nargs='+', default=[1000, 10000, 50000],
                        help='Sample sizes to benchmark (default: 1000 10000 50000)')
    parser.add_argument('--world_size', type=int, default=5,
                        help='Number of processes to use (default: 5)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='Directory to save benchmark results (default: benchmark_results)')
    
    args = parser.parse_args()
    
    results = run_benchmarks(args.sample_sizes, args.world_size, args.epochs)
    generate_comparison_graphs(results, args.output_dir)
    
    print("Benchmarking complete!")
