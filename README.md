# **3D-AVA: ZED-to-GaussianAvatar Data Pipeline**

## 🧠 Overview

**3D-AVA** is a modular pipeline that transforms **ZED camera recordings** into high-quality, SMPL-based human representations suitable for training **GaussianAvatar** — a state-of-the-art neural rendering model based on 3D Gaussians.

It extracts and packages synchronized **RGB frames**, **depth data**, **keypoints**, **segmentation masks**, **camera parameters**, and **SMPL parameters** — all formatted for neural avatar generation.

![Workflow](./assets/main-workflow.png)

---

## 🛰️ **Step 1: Capture — ZED Camera → SVO File**

The **ZED stereo camera** records synchronized **RGB-D video**, depth maps, and inertial data, saved in ZED’s proprietary `.svo` format.

✅ **Output:**

* `.svo` file (contains RGB frames, depth, IMU, and timestamped metadata)

---

## 🧩 **Step 2: Data Extraction — SVO → Structured Data**

Using the ZED SDK, the pipeline extracts synchronized data streams from `.svo` into usable formats:

| Output              | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `Frames`            | RGB images extracted per frame                            |
| `BBox`              | Bounding boxes from the Object Detection module           |
| `KeyPoints`         | 2D keypoints from the Body Tracking module (ZED38 format) |
| `Camera Intrinsics` | Per-frame intrinsic + extrinsic matrices                  |

Each output supports a downstream module and enables frame-wise annotation consistency.

🖼 **Sample Output:**

<div style="display: flex; gap: 20px;">
  <img src="./assets/sample-keypoints.png" width="45%"/>
  <img src="./assets/sample-bbox.png" width="45%"/>
</div>

---

## 🖼️ **Step 3: Segmentation — Segment Anything (SAM)**

Uses Meta AI’s **Segment Anything Model (SAM)** to generate fine-grained human silhouettes.

🧾 **Input:**

* RGB `Frames`
* Corresponding `Bounding Boxes`

🎯 **Output:**

* Binary `Masks` (segmenting the human subject)

🖼 **Example Output:** 

<img src="./assets/sample-mask.png" width="40%"/>

---

## 🧍 **Step 4: SMPL Estimation — EasyMocap**

**EasyMocap** reconstructs a full **3D SMPL model** from image-space annotations.

📥 **Input:**

* `Frames`, `BBox`, and `KeyPoints`

🧠 **Process:**

* Uses optimization and inverse kinematics to regress the SMPL model

📤 **Output:**

* `betas`: Body shape (10D)
* `poses`: Joint pose angles (axis-angle format)
* `Rh`: Global orientation
* `Th`: Global translation

This is the **canonical 3D representation** used by GaussianAvatar for consistent geometry and animation.

---

## 🧊 **Step 5: Packaging for GaussianAvatar**

The pipeline assembles all assets required to train **GaussianAvatar**:

| Asset                       | Purpose                                          |
| --------------------------- | ------------------------------------------------ |
| `SMPL Params (.npz/.pth)`   | Canonical body geometry and motion               |
| `Segmentation Masks (.png)` | silhouette supervision for photorealism |
| `Camera Intrinsics (.npy)`  | Accurate projection of 3D Gaussians to 2D frame  |
| `RGB Frames (.png)`         | Ground truth supervision of color and structure  |

---

## 🎯 **Final Output: Training-Ready Dataset**

At the end, the following are exported:

📦 **Directory structure:**

```
└── subject_name/
    ├── images/           # RGB frames
    ├── masks/            # Segmentation masks 
    ├── smpl_parms.pth    # Pose, shape, global orientation, translation
    ├── cam_intrinsics.npy
```

## 🧪 Experimental Outcome & Reflection

After implementing the full **3D-AVA pipeline** and generating a dataset using ZED recordings, I used the output to train the **GaussianAvatar** model.

### ❌ Initial Attempt (ZED-Based Dataset):

Despite substantial effort to run and adapt the **official GaussianAvatar source code**, the training results using the **ZED + EasyMocap + SAM pipeline** were **unsatisfactory**:

* SMPL alignment issues with frames — likely due to **EasyMocap misregistration** or **camera mismatch**
* Time-intensive debugging due to the complexity and lack of modularity in the original codebase
* Produce empty rendered images 
---

### ✅ Successful Attempt (Alternate Dataset: M4)

To validate the training pipeline, I switched to an official dataset called **M4** (part of the original GaussianAvatar repo or community benchmarks).

* Successfully trained **GaussianAvatar** end-to-end using M4 data 180 Epochs
* Managed to **animate the avatar** with clean geometry and realistic rendering
* Demonstrated the full cycle from training to rendering and motion

### Demo of Rendered Human
https://github.com/user-attachments/assets/c368982c-98bc-448e-8285-429230581af9
