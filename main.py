import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from data_parallel_train import run_data_parallel
from layer_model_parallel_train import run_model_parallel

# Set up plotting style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Run benchmarks for both data parallel and model parallel training
def run_benchmarks(sample_sizes=[1000, 10000, 50000], world_size=5, epochs=5):
    results = {
        'data_parallel': {},
        'model_parallel': {}
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
        
        results['data_parallel'][sample_size] = data_parallel_results
        results['model_parallel'][sample_size] = model_parallel_results
    
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
        
        plt.plot(dp_data.groupby('epoch')['accuracy'].mean(), 
                 label=f'Data Parallel ({sample_size} samples)', 
                 marker='o')
        plt.plot(mp_data['epoch'], mp_data['accuracy'], 
                 label=f'Model Parallel ({sample_size} samples)', 
                 marker='x')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison: Data Parallel vs Model Parallel')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_comparison.png")
    
    # 2. Loss Comparison
    plt.figure(figsize=(12, 8))
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size][results['model_parallel'][sample_size]['worker'] == world_size-1]
        
        plt.plot(dp_data.groupby('epoch')['loss'].mean(), 
                 label=f'Data Parallel ({sample_size} samples)', 
                 marker='o')
        plt.plot(mp_data['epoch'], mp_data['loss'], 
                 label=f'Model Parallel ({sample_size} samples)', 
                 marker='x')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison: Data Parallel vs Model Parallel')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_comparison.png")
    
    # 3. Training Time Comparison
    plt.figure(figsize=(12, 8))
    dp_times = []
    mp_times = []
    
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size][results['model_parallel'][sample_size]['worker'] == world_size-1]
        
        dp_times.append(dp_data['epoch_time'].mean())
        mp_times.append(mp_data['epoch_time'].mean())
    
    width = 0.35
    x = np.arange(len(sample_sizes))
    
    plt.bar(x - width/2, dp_times, width, label='Data Parallel')
    plt.bar(x + width/2, mp_times, width, label='Model Parallel')
    
    plt.xlabel('Sample Size')
    plt.ylabel('Average Epoch Time (s)')
    plt.title('Training Time Comparison')
    plt.xticks(x, sample_sizes)
    plt.legend()
    plt.savefig(f"{output_dir}/training_time_comparison.png")
    
    # 4. Communication vs Computation Time
    plt.figure(figsize=(15, 10))
    
    sample_labels = []
    dp_compute = []
    dp_comm = []
    mp_compute = []
    mp_comm = []
    
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size]
        
        sample_labels.append(str(sample_size))
        dp_compute.append(dp_data['compute_time'].mean())
        dp_comm.append(dp_data['comm_time'].mean())
        mp_compute.append(mp_data['compute_time'].mean())
        mp_comm.append(mp_data['comm_time'].mean())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
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
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/compute_vs_comm_comparison.png")
    
    # 5. CPU Utilization
    plt.figure(figsize=(12, 8))
    
    dp_cpu = []
    mp_cpu = []
    
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size]
        
        dp_cpu.append(dp_data['avg_cpu'].mean())
        mp_cpu.append(mp_data['avg_cpu'].mean())
    
    width = 0.35
    x = np.arange(len(sample_sizes))
    
    plt.bar(x - width/2, dp_cpu, width, label='Data Parallel')
    plt.bar(x + width/2, mp_cpu, width, label='Model Parallel')
    
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
    
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size]
        
        dp_mem.append(dp_data['avg_memory'].mean())
        mp_mem.append(mp_data['avg_memory'].mean())
    
    width = 0.35
    x = np.arange(len(sample_sizes))
    
    plt.bar(x - width/2, dp_mem, width, label='Data Parallel')
    plt.bar(x + width/2, mp_mem, width, label='Model Parallel')
    
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
    
    for sample_size in sample_sizes:
        dp_data = results['data_parallel'][sample_size]
        mp_data = results['model_parallel'][sample_size]
        
        dp_idle.append(dp_data['idle_time'].mean())
        mp_idle.append(mp_data['idle_time'].mean())
    
    width = 0.35
    x = np.arange(len(sample_sizes))
    
    plt.bar(x - width/2, dp_idle, width, label='Data Parallel')
    plt.bar(x + width/2, mp_idle, width, label='Model Parallel')
    
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
    
    dp_metrics = [
        dp_data['accuracy'].max() / 100,
        1 - (dp_data['epoch_time'].mean() / max(dp_data['epoch_time'].mean(), mp_data['epoch_time'].mean())),
        1 - (dp_data['compute_time'].mean() / max(dp_data['compute_time'].mean(), mp_data['compute_time'].mean())),
        1 - (dp_data['comm_time'].mean() / max(dp_data['comm_time'].mean(), mp_data['comm_time'].mean())),
        1 - (dp_data['avg_memory'].mean() / max(dp_data['avg_memory'].mean(), mp_data['avg_memory'].mean())),
        1 - (dp_data['avg_cpu'].mean() / max(dp_data['avg_cpu'].mean(), mp_data['avg_cpu'].mean()))
    ]
    
    mp_metrics = [
        mp_data[mp_data['worker'] == world_size-1]['accuracy'].max() / 100,
        1 - (mp_data['epoch_time'].mean() / max(dp_data['epoch_time'].mean(), mp_data['epoch_time'].mean())),
        1 - (mp_data['compute_time'].mean() / max(dp_data['compute_time'].mean(), mp_data['compute_time'].mean())),
        1 - (mp_data['comm_time'].mean() / max(dp_data['comm_time'].mean(), mp_data['comm_time'].mean())),
        1 - (mp_data['avg_memory'].mean() / max(dp_data['avg_memory'].mean(), mp_data['avg_memory'].mean())),
        1 - (mp_data['avg_cpu'].mean() / max(dp_data['avg_cpu'].mean(), mp_data['avg_cpu'].mean()))
    ]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    dp_metrics += dp_metrics[:1]
    mp_metrics += mp_metrics[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    ax.plot(angles, dp_metrics, 'o-', linewidth=2, label='Data Parallel')
    ax.fill(angles, dp_metrics, alpha=0.25)
    ax.plot(angles, mp_metrics, 'o-', linewidth=2, label='Model Parallel')
    ax.fill(angles, mp_metrics, alpha=0.25)
    
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
