# HorizonML

![image](./assets/horizonml.png)

A Hybrid Model Parallelism Framework for Distributed Training on Edge Devices. HorizonML enables efficient training of machine learning models across heterogeneous edge devices using distributed model parallelism, optimizing computation, communication, and resource allocation.

## Overview

This repository contains implementations of two distributed training approaches:

1. **Data Parallelism**: Distributes data across multiple workers, with each worker having a complete copy of the model.
2. **Model Parallelism**: Splits the model across multiple workers, with each worker processing the same data but different parts of the model.

Both implementations use PyTorch's distributed communication primitives and are designed to work on CPU devices.

## Project Structure

- `data_parallel_train.py`: Implementation of data parallel training
- `layer_model_parallel_train.py`: Implementation of model parallel training (layer-wise)
- `main.py`: Benchmarking script to compare both approaches
- `analyze_results.py`: Additional analysis tools for the benchmark results

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- pandas
- matplotlib
- seaborn
- psutil

## Running the Code

### Data Parallel Training

To run data parallel training independently, use the data_parallel_train.py script with the following parameters:
- `--world_size`: Number of processes/workers to use (default: 5)
- `--epochs`: Number of training epochs (default: 5)
- `--sample_size`: Number of samples to use from CIFAR-10 (default: 1000)

Results will be saved in the `data_parallel_logs` directory.

### Model Parallel Training

To run model parallel training independently, use the layer_model_parallel_train.py script with the following parameters:
- `--world_size`: Number of processes/workers to use (default: 5)
- `--epochs`: Number of training epochs (default: 5)
- `--sample_size`: Number of samples to use from CIFAR-10 (default: 1000)

Results will be saved in the `model_parallel_logs` directory.

### Benchmarking Both Approaches

To run benchmarks comparing both approaches, use the main.py script with these parameters:
- `--sample_sizes`: List of sample sizes to benchmark (default: 1000 10000 50000)
- `--world_size`: Number of processes/workers to use (default: 5)
- `--epochs`: Number of training epochs (default: 5)
- `--output_dir`: Directory to save benchmark results (default: benchmark_results)

This will run both training approaches with the specified sample sizes and generate comparison graphs in the `benchmark_results` directory.

## Benchmark Metrics

The benchmarking compares the following metrics:

1. **Accuracy**: Model accuracy on the training data
2. **Loss**: Training loss
3. **Training Time**: Time per epoch
4. **Computation vs. Communication Time**: Breakdown of time spent
5. **CPU Utilization**: Average CPU usage
6. **Memory Usage**: Average memory consumption
7. **Worker Idle Time**: Time workers spend waiting
8. **Gradient Divergence**: For data parallel training
9. **Bandwidth Usage**: For model parallel training
10. **Overall Performance**: Radar chart comparing all metrics

## Implementation Details

### Data Parallelism

- Uses PyTorch's DistributedDataParallel (DDP)
- Each worker has a complete copy of the model
- Data is split among workers using DistributedSampler
- Gradients are synchronized across workers during backpropagation

### Model Parallelism

- Custom implementation that splits ResNet18 into segments
- Each worker processes a different segment of the model
- Data flows through the model segments in a pipeline fashion
- Only the last worker computes the loss and performs backpropagation

## Notes

- Both implementations are designed for CPU training
- The code uses the "gloo" backend for distributed communication
- For larger sample sizes, expect longer training times
- The model used is ResNet18 from torchvision

## Troubleshooting

- If you encounter port conflicts, the code will automatically find a free port
- If processes don't terminate properly, you might need to manually kill them
- For any "address already in use" errors, wait a few moments before retrying

## Command Examples

### Running Data Parallel Training

```bash
python data_parallel_train.py --world_size 5 --epochs 5 --sample_size 1000
```

### Running Model Parallel Training

```bash
python layer_model_parallel_train.py --world_size 5 --epochs 5 --sample_size 1000
```

### Running Benchmarks

```bash
python main.py --sample_sizes 1000 10000 50000 --world_size 5 --epochs 5 --output_dir benchmark_results
```
