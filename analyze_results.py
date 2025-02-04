import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Step 1: Load all training logs
log_files = glob.glob("training_logs_worker_*.csv")
df_list = [pd.read_csv(file) for file in log_files]
df = pd.concat(df_list)  # Combine all worker logs

# Step 2: Compute average loss, accuracy, and training time per epoch
df_avg = df.groupby("Epoch").mean().reset_index()

# Step 3: Plot Loss vs Epoch
plt.figure(figsize=(8,5))
sns.lineplot(x=df_avg["Epoch"], y=df_avg["Loss"], marker="o", label="Avg Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()

# Step 4: Plot Accuracy vs Epoch
plt.figure(figsize=(8,5))
sns.lineplot(x=df_avg["Epoch"], y=df_avg["Accuracy"], marker="o", label="Avg Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy Over Epochs")
plt.legend()
plt.savefig("accuracy_plot.png")
plt.show()

# Step 5: Plot Training Time Per Epoch
plt.figure(figsize=(8,5))
sns.barplot(x=df_avg["Epoch"], y=df_avg["Time"])
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.title("Training Time Per Epoch")
plt.savefig("time_plot.png")
plt.show()

# Save processed data
df_avg.to_csv("processed_training_data.csv", index=False)
print("Analysis complete. Processed data saved.")
