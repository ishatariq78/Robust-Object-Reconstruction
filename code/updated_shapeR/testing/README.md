# testing — original ShapeR vs the trained mask model

A clean, self-contained test bench. Three files.

| file | what it does |
|---|---|
| `augment.py` | makes the synthetic test data: occlusion, and a real 3D object standing on the target. Also holds the camera maths and its self-test. |
| `run_test.py` | runs BOTH models on every condition and saves the meshes |
| `compare.py` | renders visual comparison sheets, one row per object |

## The two models

- **ORIGINAL** — the untouched `ShapeRDenoiser`, the same class `../../infer_shape.py` loads.
- **TRAINED** — the same checkpoint plus the mask-guided fine-tune, fed SAM2 masks as trained.

Both get identical images, identical SAM2 masks and the identical seed in every condition.

## The conditions

| condition | what changes in the data |
|---|---|
| `clean` | nothing |
| `occluded` | a grey bar/blob hides a measured fraction of the object in every photo; SLAM dots behind it are dropped |
| `onobject` | a real 3D object stands on the target's own flat surface, rendered through each view's camera, occluding it where it is in front |

Only image pixels change (plus the point-projection mask that follows from occlusion). The SLAM
point cloud, ground-truth mesh, cameras, crop and caption are never touched.

## Run

```bash
sbatch updated_shapeR/testing/run_test.slurm
# or
python updated_shapeR/testing/run_test.py --input_pkls data/BNB1045__wovenstool.pkl
python updated_shapeR/testing/compare.py
```

## Things that are easy to get wrong, handled here

- **Camera self-test gates the run.** Our projection is checked against ShapeR's own point mask
  (median error in pixels) before anything is generated; a wrongly placed object would silently
  invalidate the test.
- **One camera per object for every render**, framed on its ground truth at metric scale. Framing
  the GT from a scaled mesh makes it look twice the size of the predictions.
- **The manifest merges.** Running on one object does not erase the others' records.
- **Achieved, not requested.** Occlusion level and object size are measured per view and reported.
