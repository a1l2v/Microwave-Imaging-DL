import numpy as np
import h5py
import matplotlib.pyplot as plt

# -----------------------------
# Constants & utilities
# -----------------------------
c0   = 299_792_458.0
eps0 = 8.854187817e-12
mu0  = 4e-7*np.pi

def rfft_db(x, dt):
    """Return freqs (Hz) and |X(f)| in dB from a real time-series x sampled at dt."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), dt)
    mag = np.abs(X) + 1e-15
    return f, 20*np.log10(mag/np.max(mag))

def load_phantom(h5file, group="phantom_000"):
    """
    Load epsilon (relative), sigma (S/m) from Section-1 HDF5.
    Outside the breast mask, fill with air (eps_r=1, sigma=0).
    Returns eps_r, sigma, dx, dy (meters), and metadata dict.
    """
    with h5py.File(h5file, "r") as h5:
        g = h5[group]
        eps = g["epsilon"][()]   # relative permittivity inside breast, 0 outside
        sig = g["sigma"][()]     # conductivity inside breast, 0 outside
        mask_breast = g["mask_breast"][()].astype(bool)
        meta = {k: g.attrs[k] for k in g.attrs}

    # Pixel size is in cm in your generator
    dx = dy = float(meta["pixel_size_cm"]) * 1e-2  # meters

    # ensure outside breast is air
    eps_r = np.ones_like(eps, dtype=float)
    sig_m = np.zeros_like(sig, dtype=float)
    eps_r[mask_breast] = np.clip(eps[mask_breast], 1.0, None)  # keep >= 1
    sig_m[mask_breast] = np.clip(sig[mask_breast], 0.0, None)

    return eps_r, sig_m, dx, dy, meta

def make_absorber_sigma(shape, npml=12, sigma_max=5.0):
    """
    Simple graded lossy layer on all four sides.
    Returns sigma_extra array (S/m) to add to material sigma.
    """
    ny, nx = shape
    sigma_extra = np.zeros((ny, nx), dtype=float)

    # polynomial grading profile
    def grade(n, N):
        r = (N - n + 1) / N
        return r**3

    # left/right
    for i in range(npml):
        s = sigma_max * grade(i+1, npml)
        sigma_extra[:, i]      += s
        sigma_extra[:, nx-1-i] += s
    # top/bottom
    for j in range(npml):
        s = sigma_max * grade(j+1, npml)
        sigma_extra[j, :]      += s
        sigma_extra[ny-1-j, :] += s

    return sigma_extra

def gaussian_pulse(t, t0, spread):
    return np.exp(-((t - t0)**2) / (2*spread**2))

# -----------------------------
# 2D TEz FDTD core
# -----------------------------
def fdtd_2d_tez(eps_r, sigma, dx, dy, nsteps=2500, s_c=0.5,
                src_x=4, rx_x=None, y_probe=None, absorber_sigma=None):
    """
    Minimal TEz solver with lossy boundary.
    eps_r, sigma: material maps (ny, nx). Conductivity in S/m.
    Returns: dict with time traces at TX probe (near source) and RX probe (far).
    """
    ny, nx = eps_r.shape
    if rx_x is None: rx_x = nx - 5
    if y_probe is None: y_probe = ny // 2

    # Courant-limited dt
    dt = s_c / (c0 * np.sqrt((1/dx**2) + (1/dy**2)))

    # Fields
    Ez = np.zeros((ny, nx), dtype=float)
    Hx = np.zeros((ny, nx-1), dtype=float)
    Hy = np.zeros((ny-1, nx), dtype=float)

    # Total conductivity (material + absorber)
    if absorber_sigma is None:
        absorber_sigma = np.zeros_like(eps_r)
    sigma_tot = sigma + absorber_sigma

    # Update coeffs for Ez (with conduction term)
    Ca = (1 - (sigma_tot * dt) / (2 * eps0 * eps_r)) / (1 + (sigma_tot * dt) / (2 * eps0 * eps_r))
    Cb = (dt / (eps0 * eps_r)) / (1 + (sigma_tot * dt) / (2 * eps0 * eps_r))

    # Recording
    tx_trace = []
    rx_trace = []

    # Source time function (wideband Gaussian)
    t0 = 80
    spread = 20

    for n in range(nsteps):
        # --- Update H fields (Yee staggered) ---
        # Hx(i,j+1/2) update (curl Ez wrt y)
        Hx[:, :] -= (dt / mu0) * (Ez[:, 1:] - Ez[:, :-1]) / dy
        # Hy(i+1/2,j) update (curl Ez wrt x)
        Hy[:, :] += (dt / mu0) * (Ez[1:, :] - Ez[:-1, :]) / dx

        # --- Update Ez field ---
        # Curl H = dHy/dx - dHx/dy on Ez nodes
        curlH = np.zeros_like(Ez)
        curlH[1:-1,1:-1] = ((Hy[1:,1:-1] - Hy[:-1,1:-1]) / dx) - ((Hx[1:-1,1:] - Hx[1:-1,:-1]) / dy)

        Ez = Ca * Ez + Cb * curlH

        # --- Soft source (additive) on a vertical line at src_x ---
        Ez[y_probe, src_x] += gaussian_pulse(n, t0, spread)

        # --- Record probes ---
        tx_trace.append(Ez[y_probe, src_x+2])   # just to the right of source
        rx_trace.append(Ez[y_probe, rx_x])      # near right boundary

    return {
        "dt": dt,
        "tx_time": np.array(tx_trace),
        "rx_time": np.array(rx_trace),
    }

# -----------------------------
# Build S-parameters from two runs
# -----------------------------
def compute_sparams(ref, tot, band=(1e9, 8e9)):
    """
    ref, tot: dicts from fdtd_2d_tez() for reference & scatter runs.
    Returns freqs (Hz), S11, S21 as complex spectra and magnitudes in dB.
    """
    assert np.isclose(ref["dt"], tot["dt"])
    dt = ref["dt"]

    # Incident (at TX) and transmitted (at RX) in reference
    inc_tx = ref["tx_time"]
    inc_rx = ref["rx_time"]

    # Total (with phantom)
    tot_tx = tot["tx_time"]
    tot_rx = tot["rx_time"]

    # Reflected = total_at_tx - incident_at_tx
    refl_tx = tot_tx - inc_tx
    trans_rx = tot_rx  # total at RX is already what's transmitted through

    # FFT
    F = np.fft.rfftfreq(len(inc_tx), dt)
    I_tx = np.fft.rfft(inc_tx)
    I_rx = np.fft.rfft(inc_rx)
    R_tx = np.fft.rfft(refl_tx)
    T_rx = np.fft.rfft(trans_rx)

    # Avoid divide-by-zero
    eps = 1e-15
    S11 = R_tx / (I_tx + eps)
    S21 = T_rx / (I_rx + eps)

    # Band-select (1–8 GHz default)
    fmin, fmax = band
    band_idx = (F >= fmin) & (F <= fmax)

    return {
        "freq": F[band_idx],
        "S11": S11[band_idx],
        "S21": S21[band_idx],
        "S11_dB": 20*np.log10(np.abs(S11[band_idx]) + eps),
        "S21_dB": 20*np.log10(np.abs(S21[band_idx]) + eps),
    }

# -----------------------------
# End-to-end runner
# -----------------------------
def main():
    # --- Load one phantom from Section-1 output ---
    eps_r, sigma, dx, dy, meta = load_phantom("umbid_synthetic_phantoms.h5", "phantom_000")

    ny, nx = eps_r.shape

    # --- Build simple absorber (PML-like graded loss) ---
    absorber_sigma = make_absorber_sigma((ny, nx), npml=12, sigma_max=8.0)

    # --- Simulation config ---
    nsteps = 3000
    src_x = 4
    rx_x  = nx - 6
    y_probe = ny // 2

    # --- Run reference (homogeneous free-space) ---
    eps_r_ref = np.ones_like(eps_r)
    sigma_ref = np.zeros_like(sigma)
    ref = fdtd_2d_tez(eps_r_ref, sigma_ref, dx, dy, nsteps=nsteps,
                      src_x=src_x, rx_x=rx_x, y_probe=y_probe,
                      absorber_sigma=absorber_sigma)

    # --- Run scatter (with phantom) ---
    tot = fdtd_2d_tez(eps_r, sigma, dx, dy, nsteps=nsteps,
                      src_x=src_x, rx_x=rx_x, y_probe=y_probe,
                      absorber_sigma=absorber_sigma)

    # --- Compute S-parameters ---
    sp = compute_sparams(ref, tot, band=(1e9, 8e9))

    # --- Plots ---
    plt.figure(figsize=(8,4))
    plt.plot(sp["freq"]/1e9, sp["S11_dB"], label="|S11| (dB)")
    plt.plot(sp["freq"]/1e9, sp["S21_dB"], label="|S21| (dB)")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Magnitude (dB)"); plt.grid(True); plt.legend(); plt.title("Estimated S-parameters")
    plt.show()

    # Optional: inspect time traces
    # import matplotlib.pyplot as plt
    # plt.figure(); plt.plot(ref["tx_time"], label="TX incident"); plt.plot(tot["tx_time"], label="TX total"); plt.legend(); plt.title("Time traces at TX"); plt.show()
    # plt.figure(); plt.plot(ref["rx_time"], label="RX incident (ref)"); plt.plot(tot["rx_time"], label="RX total"); plt.legend(); plt.title("Time traces at RX"); plt.show()

if __name__ == "__main__":
    main()
