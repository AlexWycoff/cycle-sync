# Cycle-Sync: camera location estimation from images

This repository implements the **Cycle-Sync** camera location solver
(Li, Shi and Lerman, NeurIPS 2025) together with an end-to-end pipeline that starts
from a set of photographs and ends with estimated camera positions.

Given images of a scene, the pipeline matches features between every pair of images,
estimates the direction from each camera to the other, and then solves for all camera
locations at once while automatically discounting pairs whose directions are wrong.

---

## Contents

1. [The algorithm](#1-the-algorithm)
2. [From images to camera locations](#2-from-images-to-camera-locations)
3. [Repository layout](#3-repository-layout)
4. [Installation](#4-installation)
5. [Getting the data](#5-getting-the-data)
6. [Running the tests](#6-running-the-tests)
7. [Using Cycle-Sync on your own data](#7-using-cycle-sync-on-your-own-data)
8. [Results](#8-results)
9. [Citations](#9-citations)
10. [License](#10-license)

---

## 1. The algorithm

**Problem.** There are `n` cameras with unknown locations `t_1, ..., t_n`. For some pairs
of cameras `(i, j)` we are given a unit vector `γ_ij` that should point from camera `i` to camera `j`. Many of these directions are corrupted, because feature matching and pose estimation fail on some image pairs. The goal is to recover all locations up to a global translation and scale.

**Idea.** Around any triangle of cameras `i, j, k`, the true displacements add up
to zero. A clean direction therefore fits consistently into the triangles it belongs to,
while a corrupted one does not. Cycle-Sync uses this *cycle consistency* to decide how
much to trust each edge.

**Steps.**

1. **Initial trust scores (T-AAB).** For each edge and each triangle through it, measure
   how far the edge's direction is from the set of directions that would close the
   triangle (the AAB inconsistency, Shi and Lerman 2018). Average these over triangles,
   giving more weight to triangles whose other two edges look reliable, and repeat a few
   times. Edges with low scores get high initial weights.
2. **Weighted least squares.** Solve for locations and edge lengths `α_ij ≥ 1` that
   minimize the weighted sum of `‖t_j − t_i − α_ij γ_ij‖²`, with the locations centered at
   the origin. The constraint `α_ij ≥ 1` fixes the scale and rules out the collapsed
   solution where all cameras coincide.
3. **Cycle scores.** Using the current estimate, compute each edge's residual, and for
   every triangle through the edge check whether the edge's direction, scaled by the
   *estimated* distances, closes the triangle. Triangles through low-residual edges count
   more.
4. **Reweighting.** Blend the residual and the cycle score, relying more on the cycle
   score in later iterations, and convert the result to a weight with a robust
   Welsch-type function. Return to step 2.

After about 20 iterations, corrupted edges carry almost no weight and the locations are
determined by the consistent ones.

## 2. From images to camera locations

The notebook `Cycle Sync Test.ipynb` runs the whole pipeline on an ETH3D scene:

1. **Calibration.** Camera intrinsics and ground-truth poses are read from the COLMAP
   calibration that ships with the dataset.
2. **Features.** SIFT features are computed once per image, after resizing each image so
   its longer side is 2000 pixels (keeping the aspect ratio). The intrinsics are scaled to
   match.
3. **Matching.** Every pair of images is matched, and ambiguous matches are removed with
   Lowe's ratio test.
4. **Relative pose.** For each pair, the essential matrix is estimated with the 5-point
   algorithm inside RANSAC, and the relative rotation and translation are recovered with a
   cheirality check (`relpose` in `helper_funcs.py`).
5. **Pair selection.** Pairs with fewer than 100 inliers are discarded. For courtyard this
   keeps about 290 of the 703 pairs, and the resulting graph stays connected.
6. **Directions.** Each relative translation is rotated into the world frame
   (`world_direction`) to give `γ_ij`.
7. **Location estimation and scoring.** Cycle-Sync estimates the locations, which are then
   compared with the ground truth after a robust scale-and-translation alignment. A plain
   least squares estimate is computed as a baseline.

## 3. Repository layout

```
README.md
requirements.txt
.gitignore
Cycle_Sync/
  Cycle Sync Test.ipynb
  cycle_sync.py
  helper_funcs.py
  test.py
  test_synthetic.py
  test_case.py
  object/constructor.py
  utils/
    params.py
    indexing.py
    triangles.py
    aab.py
    reweight.py
    cycle_inconsistency.py
    irls.py
    cvxopt_quadprog.py
SfM_data_clean/                (Setup explained in section 5)
```

## 4. Installation

Python 3.12 is recommended; all dependencies have prebuilt packages for it.

```
pip install -r requirements.txt
```

To run the notebook, either open it in VS Code with the **Python** and **Jupyter**
extensions installed and select your Python as the kernel, or start Jupyter from the
code folder:

```
cd Cycle_Sync
python -m notebook
```

## 5. Getting the data

The test uses the **courtyard** scene from the ETH3D high-resolution multi-view benchmark
(Schöps et al., CVPR 2017). The data belongs to its authors and is released under the
CC BY-NC-SA 3.0 license; see https://www.eth3d.net for the full terms.

**Option A: courtyard only (recommended).** Download `courtyard.zip` from this
repository's releases page:
https://github.com/YOUR-USERNAME/YOUR-REPO/releases

**Option B: the full ETH3D archive.** Download `multi_view_training_dslr_undistorted.7z`
from https://www.eth3d.net/datasets. It contains all training scenes and is much larger;
only the `courtyard` folder is needed.

Place the scene in `SfM_data_clean/` at the top of this repository:

```
SfM_data_clean/
  courtyard/
    dslr_calibration_undistorted/     cameras.txt, images.txt, points3D.txt
    images/
      dslr_images_undistorted/        38 JPG images
```

The notebook reads the data through the relative path `../SfM_data_clean/courtyard`,
so the folder must sit next to `Cycle_Sync/`, not inside it.

## 6. Running the tests

From the `Cycle_Sync` folder:

```
python test.py
python test_synthetic.py
```

- `test.py` recovers four cameras at the corners of a square from exact directions. The
  last line, `RMS alignment error`, should be around `1e-9` or smaller.
- `test_synthetic.py` generates 50 random cameras with 0% to 60% of directions replaced
  by random vectors. It should end with `All synthetic checks passed.`

Then open `Cycle Sync Test.ipynb`, restart the kernel and run all cells. It takes about
10-15 minutes, mostly for feature matching. The output of the last cell is the result:

```
Median error / scene radius (robust alignment), and % of cameras within 0.1:
Cycle-Sync:                  0.0121  (71%)
Plain least squares baseline: 0.7393  (3%)
```

The first number is the median distance between estimated and true camera positions,
divided by the typical distance of a camera from the scene center. Values vary slightly
between runs because RANSAC is randomized.

## 7. Using Cycle-Sync on your own data

```python
from cycle_sync import Cycle_Sync
from utils.indexing import tijmat_from_dict

tijMat = tijmat_from_dict(AdjMat, t_dict)
t_est, out = Cycle_Sync(AdjMat, tijMat, {'seed': 0})
```

`t_est` is a `3 x n` array of locations. `out` also contains the edge scales `alph`,
the run time `TotalTime`, the initial T-AAB scores `IRAABVec`, the final weights `wVec`,
residuals `rVec`, cycle scores `sVec`, the least squares objective at each iteration
`costVec`, and the solver status at each iteration `statusList`.

Main parameters (defaults in `utils/params.py`):

| Key | Default | Meaning |
|---|---|---|
| `WLSiters` | 20 | Cycle-Sync iterations |
| `tau1` | 20 | sharpness of the initial weights from T-AAB |
| `tau4` | 4 | sharpness of the robust weight function |
| `beta` | 20 | how strongly triangle scores favor reliable neighboring edges |
| `flam` | `t/(t+10)` | schedule for blending residuals and cycle scores |
| `sinmin` | 0.6 | minimum triangle angle (as a sine) used by T-AAB |
| `nsample` | 200 | triangles sampled per edge for T-AAB |
| `maxtri` | 100 | maximum triangles per edge for cycle scores |
| `seed` | `None` | random seed for reproducible runs |

## 8. Results

ETH3D courtyard, 38 cameras, directions computed from the images by the pipeline above:

| Method | Median error of scene radius | Cameras within 0.1 of scene radius |
|---|---|---|
| Cycle-Sync | 0.012 to 0.014 | 68% to 71% |
| Plain least squares | 0.60 to 0.74 | 0% to 3% |

## 9. Citations

Please cite the Cycle-Sync paper when using this code:

```bibtex
@inproceedings{li2025cyclesync,
  title     = {Cycle-Sync: Robust Global Camera Pose Estimation through Enhanced Cycle-Consistent Synchronization},
  author    = {Li, Shaohan and Shi, Yunpeng and Lerman, Gilad},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

## 10. License

Cycle-Sync is provided for educational and research use under the license terms included in the LICENSE files.
Please contact Prof. Gilad Lerman (lerman@umn.edu) for more information.