# Tracking double descent: hidden-geometry sweeps of a toy superposition model

A replication and extension of the toy model in [Henighan et al. (2023), *Superposition, Memorization, and Double Descent*](https://transformer-circuits.pub/2023/toy-double-descent/index.html). A small ReLU-output autoencoder is trained on sparse synthetic data for many training-set sizes T, and the geometry of its hidden vectors (the 2D or 3D representation of every training point) is tracked across the transition from memorizing data points to generalizing features.

## Live interactive dashboards

- [2D sweep (m = 2)](https://ezeepeezee.github.io/double-descent-hidden-geometry-sweeps/m_2/dashboard_2d.html)
- [3D sweep (m = 3)](https://ezeepeezee.github.io/double-descent-hidden-geometry-sweeps/m_3/dashboard_3d.html)

The data files are large, so the first load can take a while, especially on mobile.

### What the dashboards show

Each frame is one training-set size T. The main window plots the hidden vectors h_i = W x_i of all T training points. A side chart shows the test loss across the whole sweep with a marker at the current T, and the training and test loss for the current T are printed on the plot.

- **Navigation:** slider, previous/next, play with speed and loop, Auto-fit (rescale to the current frame, or use one fixed scale for the whole sweep), Reset View, and Save Snapshot (PNG).
- **2D:** scroll to zoom, drag to pan.
- **3D:** drag to rotate, scroll to zoom, right-drag or shift-drag to pan. Save Animation records the sweep as a video (MP4 or WebM, depending on the browser).

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── 2d/
│   ├── run_m_2.ipynb       # m = 2 sweep, plus export for the dashboard
│   ├── sweep.h5            # raw results (300 values of T)
│   ├── data.js             # the same results, exported for the dashboard (~51 MB)
│   └── dashboard_2d.html
└── 3d/
    ├── run_m_3.ipynb       # m = 3 sweep, plus export for the dashboard
    ├── sweep_merged.h5     # raw results (320 values of T)
    ├── data.js             # the same results, exported for the dashboard (~80 MB)
    └── dashboard_3d.html
```

## The experiment

**Model.** A one-layer autoencoder with tied weights and a ReLU output: h = W x and x̂ = ReLU(Wᵀ h + b), where W has shape m × n. The hidden vector of a training point is h_i = W x_i.

**Data.** n = 10,000 features. Each feature is active with probability 1 − S, with sparsity S = 0.999 (about 10 active features per point). Active values are drawn uniformly from [0, 1], and each point is normalized to unit L2 norm. For every T the training set is generated from a fixed per-T seed. Test loss is measured on 5,000 fresh points.

**Training.** Full-batch training on all T points with mean squared error over every entry. AdamW with peak learning rate 1e-3 and weight decay 1e-2, for 50,000 steps (2,500 linear warm-up steps, then cosine decay to zero). For each T, three models are trained from seeds 0 to 2 and the one with the lowest final training loss is kept.

| | 2D sweep | 3D sweep |
|---|---|---|
| Hidden size m | 2 | 3 |
| Training-set sizes T | 25 to 7,500, stride 25 (300 values) | 25 to 8,000, stride 25 (320 values) |
| Features n | 10,000 | 10,000 |
| Sparsity S | 0.999 | 0.999 |
| Seeds per T | 3 | 3 |
| Training steps | 50,000 | 50,000 |

## Data files

**HDF5 (`sweep.h5`, `sweep_merged.h5`).** One group per T at `results/T_<T>`. The dataset `H` holds the hidden vectors of the T training points (shape T × m, gzip-compressed). Each group has the attributes `T`, `train_loss`, `test_loss`, `D_feat`, `D_data`, and `wall_time` (seconds).

```python
import h5py

with h5py.File("2d/sweep.h5", "r") as f:
    names = sorted(f["results"], key=lambda k: int(k.split("_")[1]))
    for name in names[:3]:
        g = f["results"][name]
        print(name, g["H"].shape, dict(g.attrs))
```

**`data.js`.** The dashboards read this file, not the HDF5 files. It defines `const SWEEP = [...]` with one entry per T containing `T`, `H1`, `H2` (and `H3` in 3D), `train_loss`, `test_loss`, `D_feat`, and `D_data`. To regenerate it, run the export cell at the end of the notebook. It is written next to the notebook.

## How the data was produced

**2D (m = 2).** One continuous sweep over T = 25 to 7,500 in steps of 25, run on a single RTX 2080 Ti in about 120 hours.

**3D (m = 3).** Because of compute limits, the sweep was not run in one session. An initial run failed after T = 1495, and its partial results were not used. The data was then generated in two sessions with identical settings (T = 25 to 5,000 and T = 5,025 to 8,000), and the two output files were merged into `sweep_merged.h5`. Only the merged file is included in this repository. `run_m_3.ipynb` shows the consolidated configuration (T = 25 to 8,000), which has not been run end to end in this form. The merged file was checked to contain all 320 values of T (25 to 8,000, stride 25) with m = 3. Because every T uses a fixed dataset seed and fixed model seeds, results should not depend on which session produced them, up to GPU nondeterminism.

## Running the code

**Notebooks.** Install the dependencies with `pip install -r requirements.txt` and open the notebook for the dimension you want. Recomputing a full sweep needs a
