import os
import glob
import nibabel as nib
import numpy as np
import torchio as tio

from scipy.ndimage import affine_transform

# ---------------------------
# Load data
# ---------------------------
data_dir = "/Users/kenggkkeng/Desktop/SCT/moco_dmri/data/"

subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])
# subfolders = [os.path.join(data_dir, "sub-tokyoSigna2")]

for subfolder in subfolders:
    print(f"Processing {subfolder} ...")

    # Find NIfTI file inside the current subfolder
    nii_files = glob.glob(os.path.join(subfolder, "dwi", "sub-*dwi.nii.gz"))
    if len(nii_files) == 0:
        raise FileNotFoundError(f"No NIfTI file found in {subfolder}")
    input_img = nii_files[0]
    print("Found NIfTI:", input_img)

    # Find bvec file inside the current subfolder
    bvec_files = glob.glob(os.path.join(subfolder, "dwi", "*.bvec"))
    if len(bvec_files) == 0:
        raise FileNotFoundError(f"No bvec file found in {subfolder}")
    bvec_file = bvec_files[0]
    print("Found bvec:", bvec_file)

    basename = os.path.basename(input_img)
    output_img = os.path.join(subfolder, "dwi", "aug_" + basename)

    img = nib.load(input_img)
    data = img.get_fdata().astype(np.float32)
    H, W, D, T = data.shape
    print("Input shape:", data.shape)

    bvecs = np.loadtxt(bvec_file)
    grad_norms = np.linalg.norm(bvecs, axis=0)
    is_dwi = grad_norms > 0
    print("B0 volumes indices:", np.where(~is_dwi)[0])
    print("DWI volumes indices:", np.where(is_dwi)[0])

    spine_axis = 2  # slice axis = axial slices through C-spine
    aug_data = np.zeros_like(data)

    # ---------------------------
    # Custom slice-wise transform
    # ---------------------------
    class RandSliceWiseAffine:
        def __init__(self, max_rot=5, max_shift=2, prob=0.9, axis=spine_axis):
            self.max_rot = max_rot
            self.max_shift = max_shift
            self.prob = prob
            self.axis = axis

        def __call__(self, vol: np.ndarray) -> np.ndarray:
            out = vol.copy()
            if np.random.rand() > self.prob:
                return out

            H, W, D = out.shape
            if self.axis != 2:
                out = np.moveaxis(out, self.axis, -1)
                H, W, D = out.shape

            for idx in range(D):
                angle = np.random.uniform(-self.max_rot, self.max_rot)
                tx = np.random.uniform(-self.max_shift, self.max_shift)
                ty = np.random.uniform(-self.max_shift, self.max_shift)

                theta = np.deg2rad(angle)
                rot = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
                center = np.array([H / 2, W / 2])
                inv_rot = np.linalg.inv(rot)
                offset = center - inv_rot @ center + np.array([tx, ty])

                out[:, :, idx] = affine_transform(
                    out[:, :, idx], inv_rot, offset=offset,
                    order=0, mode="nearest"
                )

            if self.axis != 2:
                out = np.moveaxis(out, -1, self.axis)

            return out

    # -----------------------
    # Define transforms
    # -----------------------
    slicewise_tf = RandSliceWiseAffine(max_rot=5, max_shift=2, prob=0.9, axis=spine_axis)
    # artifact_tf = tio.Compose([
    #     tio.RandomBiasField(coefficients=(0.1, 0.5), p=0.5),
    #     tio.RandomGhosting(num_ghosts=4, axes=2, p=0.5),
    #     tio.RandomSpike(num_spikes=2, intensity=(0.1, 0.5), p=0.5),
    # ])

    # -----------------------
    # Apply augmentation volume-by-volume
    # -----------------------
    for t in range(T):
        vol = data[:, :, :, t]
        aug_vol = slicewise_tf(vol)
        aug_data[:, :, :, t] = aug_vol
        print(f"Augmented volume {t + 1}/{T}")

    # aug_data = artifact_tf(aug_data)
    # print("Add MR Artifacts")

    # orig_min, orig_max = data.min(), data.max()
    # aug_min, aug_max = aug_data.min(), aug_data.max()
    # aug_data = (aug_data - aug_min) / (aug_max - aug_min + 1e-8)  # normalize [0,1]
    # aug_data = aug_data * (orig_max - orig_min) + orig_min

    # -----------------------
    # Save augmented 4D data
    # -----------------------
    aug_img = nib.Nifti1Image(aug_data, img.affine, img.header)
    nib.save(aug_img, output_img)
    print(f"Saved augmented dataset: {output_img}")