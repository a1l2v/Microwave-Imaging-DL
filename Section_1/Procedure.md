Here’s a detailed breakdown of your flowchart with *purpose, **input, and **output* for each step:

---

### ✅ *Step A: Start – Define Simulation Domain (2D/3D Grid + PML Boundaries)*

*Why this step is needed:*

* To create a *computational domain* where electromagnetic simulations will happen.
* The grid represents the breast cross-section or volume, and PML (Perfectly Matched Layer) boundaries *absorb outgoing waves*, preventing reflections that would corrupt results.

*Input:*

* Domain size (e.g., 128×128 grid for 2D or 128×128×128 for 3D).
* Resolution (grid spacing, e.g., 1 mm).
* PML configuration (thickness, absorption profile).

*Output:*

* An *empty computational grid* ready to assign tissue properties and tumor inclusions.

---

### ✅ *Step B: Generate Breast Phantom (Assign Dielectric Properties)*

*Why this step is needed:*

* To mimic the electrical properties of real breast tissues. Different tissues (fat, glandular tissue) have different *permittivity (ϵ)* and *conductivity (σ)* values that affect wave propagation.

*Input:*

* Tissue property distributions from empirical data (MERIT or Lazebnik dataset):

  * Fat: ϵ ≈ 9, σ ≈ 0.1 S/m.
  * Glandular tissue: higher permittivity and conductivity.

*Output:*

* A *base breast phantom map* where each grid cell has (ϵ, σ) values representing normal tissue.

---

### ✅ *Step C: Add Tumor Models (Random Size and Location)*

*Why this step is needed:*

* To simulate cancerous regions within the breast model for algorithm testing.
* Tumors have *higher permittivity and conductivity* than surrounding tissue, making them detectable in microwave imaging.

*Input:*

* Tumor size range (e.g., 5–20 mm diameter).
* Shape type (spherical or irregular).
* Location constraints (within phantom, not overlapping boundaries).

*Output:*

* *Updated dielectric map* including tumor regions with high contrast in (ϵ, σ) compared to background.

---

### ✅ *Step D: Generate Dielectric Maps (Permittivity & Conductivity)*

*Why this step is needed:*

* To finalize the *phantom representation* that will serve as *input to PINN or FDTD simulation*.
* These maps represent the *true ground truth* for training and validating imaging algorithms later.

*Input:*

* Phantom geometry (from Step B).
* Tumor inclusions (from Step C).

*Output:*

* *Dielectric maps* (e.g., 128×128 arrays for ϵ and σ).
* Metadata: tumor size, location, and labels for supervised ML training.

---

### ✅ *Step E: PINN/FDTD Simulation (Compute EM Fields or S-parameters)*

*Why this step is needed:*

* To simulate *how electromagnetic waves interact with the breast phantom*, which is the essence of microwave imaging.
* Normally done using FDTD (expensive), but *PINN accelerates this by approximating the physics while enforcing Maxwell’s equations*.

*Input:*

* Dielectric maps (ϵ, σ) from Step D.
* Simulation settings: frequency range (1–8 GHz), number of antennas, positions, excitation signals.

*Output:*

* *Electromagnetic field distributions (E and H)* inside the domain, OR
* *S-parameters (scattering matrix)* for all transmitter–receiver pairs over the frequency range.

  * Size: N × N × K (N = number of antennas, K = frequency points).

---

### ✅ *Why this sequence matters:*

* You *cannot simulate EM fields* without first defining the geometry and material properties (Steps A–D).
* *PINN is a forward solver, so it needs **dielectric maps as input* and produces *wave response data (fields or S-parameters)* for further imaging steps.

flowchart TD
    A[Start: Define Simulation Domain] --> B[Generate Breast Phantom]
    B --> C[Add Tumor Models]
    C --> D[Generate Dielectric Maps]
    D --> E[PINN/FDTD Simulation]
