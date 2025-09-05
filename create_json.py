import os
import json
import random

from glob import glob

base_dir = "/home/ge.polymtl.ca/p122983/moco_dmri/sourcedata/"

# Parameters for splitting
train_ratio = 0.80
val_ratio = 0.10
test_ratio = 0.10

# List all subject folders automatically
all_subjects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
all_subjects.sort()
random.shuffle(all_subjects)

# Split subjects
n = len(all_subjects)
n_train = int(train_ratio * n)
n_val = int(val_ratio * n)
train_subjects = all_subjects[:n_train]
val_subjects = all_subjects[n_train:n_train + n_val]
test_subjects = all_subjects[n_train + n_val:]

def build_entries(subject_list):
    entries = []
    for sub in subject_list:
        dwi_folder = os.path.join(base_dir, sub, "dwi")
        moving_files = glob(os.path.join(dwi_folder, "aug_*.nii.gz"))
        for moving_path in moving_files:
            fixed_path = moving_path.replace("aug_", "").replace(".nii.gz", "_dwi_mean.nii.gz")
            entries.append({"moving": os.path.relpath(moving_path), "fixed": os.path.relpath(fixed_path)})
    return entries

dataset_dict = {
    "training": build_entries(train_subjects),
    "validation": build_entries(val_subjects),
    "testing": build_entries(test_subjects)
}

# Save JSON
out_path = os.path.join(base_dir, "dataset.json")
with open(out_path, "w") as f:
    json.dump(dataset_dict, f, indent=2)

print("dataset.json created successfully!")
print(f"Training subjects: {train_subjects}")
print(f"Validation subjects: {val_subjects}")
print(f"Testing subjects: {test_subjects}")
