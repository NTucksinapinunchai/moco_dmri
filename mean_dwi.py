import os
import glob
import subprocess

# ---------------------------
# Load data
# ---------------------------
data_dir = "/home/ge.polymtl.ca/p122983/moco_dmri/sourcedata/"

subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])
# subfolders = [os.path.join(data_dir, "sub-mgh")]

for subfolder in subfolders:
    print(f"Processing {subfolder} ...")

    # Find NIfTI file
    nii_files = glob.glob(os.path.join(subfolder, "dwi", "sub-*dwi.nii.gz"))
    if len(nii_files) == 0:
        raise FileNotFoundError("No NIfTI file found in folder.")
    input_img = nii_files[0]
    print("Found NIfTI:", input_img)

    # Find bvec file
    bvec_files = glob.glob(os.path.join(subfolder, "dwi", "*.bvec"))
    if len(bvec_files) == 0:
        raise FileNotFoundError("No bvec file found in folder.")
    bvec_file = bvec_files[0]
    print("Found bvec:", bvec_file)

    # Find bvec file
    bval_files = glob.glob(os.path.join(subfolder, "dwi", "*.bval"))
    if len(bval_files) == 0:
        raise FileNotFoundError("No bval file found in folder.")
    bval_file = bval_files[0]
    print("Found bval:", bvec_file)

    basename = os.path.basename(input_img)
    out_dir = os.path.join(subfolder, "dwi")

    SCT_SEP = [
        "sct_dmri_separate_b0_and_dwi",
        "-i", input_img,
        "-bvec", bvec_file,
        "-bval", bval_file,
        "-ofolder", out_dir
    ]

    try:
        subprocess.run(SCT_SEP, check=True)
        print("Success!")
    except subprocess.CalledProcessError as e:
        print("Error", e)