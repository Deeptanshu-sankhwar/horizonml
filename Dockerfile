# Use an official PyTorch base image
FROM pytorch/pytorch:latest

# Set the working directory inside the container
WORKDIR /app

# Install necessary dependencies
RUN pip install torch torchvision numpy grpcio protobuf pandas seaborn matplotlib

# Copy project files into the container
COPY . /app

# Set the entrypoint for the training script
CMD ["python", "train.py"]
