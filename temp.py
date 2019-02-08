import os

for i in range(len(os.listdir("train_data/")) - 1, 9000, -1):
    os.rename(f"train_data/train_{i}.tfrecord",
              f"val_data/val_{i-8999}.tfrecord")
