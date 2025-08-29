import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import os

# ------------------------
# Parameters
# ------------------------
CSV_FILES = {
    "glycerin": "csv/WaterDielectricMeasurementSummary.csv",   # replace with actual file if separate
    "triton": "csv/WaterDielectricMeasurementSummary.csv",     # adjust if multiple files
    "water": "csv/WaterDielectricMeasurementSummary.csv",
}
TARGET_FREQ = 3.0  # GHz
IMG_SIZE = 128
PIXEL_SIZE_CM = 0.15
N_PHANTOMS = 6
OUTFILE = "umbid_synthetic_phantoms.h5"

# ------------------------
# Helpers
# ------------------------
def load_dielectric_csv(filepath):
    """Reads dielectric CSV (freq, permittivity, conductivity)."""
    df = pd.read_csv(filepath)
    # Try to detect columns
    df.columns = [c.strip().lower() for c in df.columns]
    freq_col = [c for c in df.columns if "freq" in c][0]
    eps_col = [c for c in df.columns if "perm" in c][0]
    sig_col = [c for c in df.columns if "cond" in c or "sigma" in c][0]
    df = df[[freq_col, eps_col, sig_col]].rename(columns={freq_col:"freq", eps_col:"eps", sig_col:"sigma"})
    return df

def get_stats_near(df, f0=3.0, bw=0.1):
    """Return mean and std of eps/sigma near f0 GHz."""
    sub = df[(df["freq"] >= f0-bw) & (df["freq"] <= f0+bw)]
    return dict(
        eps_mean=sub["eps"].mean(), eps_std=sub["eps"].std(),
        sig_mean=sub["sigma"].mean(), sig_std=sub["sigma"].std()
    )

def truncnorm_sample(mean, std, low, high, size=None):
    # Ensure std is positive
    std = max(std, 1e-3)
    a, b = (low - mean)/std, (high - mean)/std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

# ------------------------
# Build tissue distributions
# ------------------------
def build_tissue_params(csvs):
    mats = {}
    for k, f in csvs.items():
        df = load_dielectric_csv(f)
        mats[k] = get_stats_near(df, f0=TARGET_FREQ)
    
    # Rough blending (can be calibrated more carefully)
    fat = dict(eps=(mats["glycerin"]["eps_mean"], mats["glycerin"]["eps_std"]),
               sig=(mats["glycerin"]["sig_mean"], mats["glycerin"]["sig_std"]),
               eps_range=(4,15), sig_range=(0.1,0.6))
    fib = dict(eps=(mats["triton"]["eps_mean"], mats["triton"]["eps_std"]),
               sig=(mats["triton"]["sig_mean"], mats["triton"]["sig_std"]),
               eps_range=(20,40), sig_range=(0.8,2.0))
    tum = dict(eps=(mats["water"]["eps_mean"], mats["water"]["eps_std"]),
               sig=(mats["water"]["sig_mean"], mats["water"]["sig_std"]),
               eps_range=(30,60), sig_range=(1.5,3.0))
    return {"fat":fat, "fib":fib, "tum":tum}

# ------------------------
# Phantom generator
# ------------------------
def generate_phantom(params):
    H, W = IMG_SIZE, IMG_SIZE
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W//2, H//2
    mask_breast = ((xx-cx)**2/(0.4*W)**2 + (yy-cy)**2/(0.48*H)**2) <= 1
    
    epsilon = np.zeros((H,W))
    sigma   = np.zeros((H,W))
    
    # Fill fat (background)
    eps_f = truncnorm_sample(*params["fat"]["eps"], *params["fat"]["eps_range"])
    sig_f = truncnorm_sample(*params["fat"]["sig"], *params["fat"]["sig_range"])
    epsilon[mask_breast] = eps_f
    sigma[mask_breast] = sig_f
    
    # Add fibroglandular islands
    mask_fib = np.zeros((H,W), dtype=bool)
    for _ in range(5):
        rx, ry = np.random.randint(20,40), np.random.randint(15,30)
        cx_, cy_ = np.random.randint(40,88), np.random.randint(40,88)
        fib = ((xx-cx_)**2/rx**2 + (yy-cy_)**2/ry**2) <= 1
        fib = fib & mask_breast
        mask_fib |= fib
    eps_g = truncnorm_sample(*params["fib"]["eps"], *params["fib"]["eps_range"])
    sig_g = truncnorm_sample(*params["fib"]["sig"], *params["fib"]["sig_range"])
    epsilon[mask_fib] = eps_g
    sigma[mask_fib] = sig_g
    
    # Add tumour
    tum_radius_cm = np.random.uniform(0.5,1.5)
    tum_radius_px = int(tum_radius_cm / PIXEL_SIZE_CM)
    tx, ty = np.random.randint(50,78), np.random.randint(50,78)
    mask_tumour = ((xx-tx)**2 + (yy-ty)**2) <= tum_radius_px**2
    eps_t = truncnorm_sample(*params["tum"]["eps"], *params["tum"]["eps_range"])
    sig_t = truncnorm_sample(*params["tum"]["sig"], *params["tum"]["sig_range"])
    epsilon[mask_tumour] = eps_t
    sigma[mask_tumour] = sig_t
    
    # Noise for heterogeneity
    epsilon[mask_breast] *= np.random.normal(1.0,0.05, size=epsilon.shape)[mask_breast]
    sigma[mask_breast]   *= np.random.normal(1.0,0.05, size=sigma.shape)[mask_breast]
    
    return dict(epsilon=epsilon, sigma=sigma, mask_breast=mask_breast.astype(np.uint8),
                mask_tumour=mask_tumour.astype(np.uint8),
                meta=dict(pixel_size_cm=PIXEL_SIZE_CM,
                          tum_rad_cm=tum_radius_cm,
                          tum_x_cm=(tx-cx)*PIXEL_SIZE_CM,
                          tum_y_cm=(ty-cy)*PIXEL_SIZE_CM))

# ------------------------
# Main
# ------------------------
def main():
    params = build_tissue_params(CSV_FILES)
    with h5py.File(OUTFILE, "w") as h5:
        for i in range(N_PHANTOMS):
            ph = generate_phantom(params)
            g = h5.create_group(f"phantom_{i:03d}")
            g.create_dataset("epsilon", data=ph["epsilon"].astype(np.float32))
            g.create_dataset("sigma", data=ph["sigma"].astype(np.float32))
            g.create_dataset("mask_breast", data=ph["mask_breast"])
            g.create_dataset("mask_tumour", data=ph["mask_tumour"])
            for k,v in ph["meta"].items():
                g.attrs[k] = v
    print(f"Saved {N_PHANTOMS} phantoms to {OUTFILE}")
    
    # Quick plot
    example = generate_phantom(params)
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    ax[0].imshow(example["epsilon"], cmap="viridis"); ax[0].set_title("Permittivity (ε')")
    ax[1].imshow(example["sigma"], cmap="inferno"); ax[1].set_title("Conductivity (σ)")
    ax[2].imshow(example["mask_tumour"], cmap="gray"); ax[2].set_title("Tumour mask")
    plt.show()

if __name__ == "__main__":
    main()
