


# Official Document for MOTIF Hand 

[![arXiv](https://img.shields.io/badge/ArXiv-2506.19201-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2506.19201) [![web](https://img.shields.io/badge/Web-Motif_Hand-blue.svg?style=plastic)](https://slurm-lab-usc.github.io/motif-hand/) [![license](https://img.shields.io/badge/LICENSE-Apache--2.0-white.svg?style=plastic)](./LICENSE)

## Table of Contents

- [Hardware Design](#hardware-design)
- [Software Modules](#software-modules)
- [Reconstruction](#reconstruction)
- [Citation](#citation)


## Hardware Design

https://github.com/Liu-wenhao-1223/motifhand-hardware_system

[Bill of Materials (BOM) table](https://docs.google.com/spreadsheets/d/1oYU1SgGldSbIuVMtG71lnZIiXFAJuYSa/edit?pli=1&gid=717379252#gid=717379252)

This subproject covers:

```
00_Reference/                  # Component Data Sheet
01_Function/                   # Function diagram of the System
02_Hardware/                   # Schematic diagram and PCB
03_Firmware/                   # Source code based on STM32H7
04_Software/                   # Tools for Calibration and Communication test                 
05_Mechanical/                 # 3D models of the Circuits and the modified Hand Structure
```




## Software Modules

https://github.com/slurm-lab-usc/motif-hand-software-modules

This subproject covers:
- 📊 Data collection from onboard tactile and inertial sensors
- 🦾 ROS2-based control architecture
- 🖥️ Real-time visualization tools

```
  Control PC                   Raspberry Pi                Hand
┌─────────────┐              ┌──────────────┐         ┌────────────┐
│ Arm Control │─────ZMQ─────>│ Sensor Server│         │ ROS2 Node  │
│             │              │ - Thermal    │         │            │
│ Trajectory  │<────ZMQ──────│ - ToF        │<────────│ FSR/IMU    │
│ Recorder    │              │ - RGB Camera │  Serial │ Recording  │
└─────────────┘              └──────────────┘         └────────────┘
       │                            │                        │
       └────────────────────────────┴────────────────────────┘
                         Data Visualization
```



After running the software modules above, you will obtain all necessary modalities for learning.

---

## Post-Processing Pipeline

To reconstruct geometry and align thermal information to the 3D scene, please follow the steps below.

### 1. Install Reconstruction Tools

**Structure-from-Motion (SfM)**  
Use [InstantSfM](https://github.com/cre185/InstantSfM) for camera pose estimation and sparse/dense reconstruction.

**Mesh Reconstruction**  
After SfM, use [robogs](https://github.com/louhz/robogs) for mesh reconstruction and 3D Gaussian processing.

### 2. Install Python Dependencies

Make sure the required Python packages are installed:

```bash
pip install -r requirements.txt
```


### Thermal Color Registration & Post-Processing

Set your data and output paths (example):
```bash
DATA_ROOT=/path/to/sequence
OUTPUT_ROOT=/path/to/output
```

Then run the thermal–color registration and post-processing script:

```bash
python your_script.py \
  --points_ply ${DATA_ROOT}/shortcanpoints.ply \
  --data_dir ${DATA_ROOT}/result \
  --heatmap_dir ${DATA_ROOT}/output_images \
  --heatmap_save_dir ${DATA_ROOT}/processed_thermal \
  --input_colored_pcd ${DATA_ROOT}/colored_cloud_final_raw_K.ply \
  --output_dir ${OUTPUT_ROOT} \
  --output_name heat.ply
```

This will produce a recolored point cloud (e.g., heat.ply) with thermally informed coloring suitable for downstream learning.


## Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{zhou2025motifhand,
    title={The MOTIF Hand: A Robotic Hand for Multimodal Observations with Thermal, Inertial, and Force Sensors},
    author={Zhou, Hanyang and Lou, Haozhe and Liu, Wenhao and Zhao, Enyu and Wang, Yue and Seita, Daniel},
    booktitle={International Symposium on Experimental Robotic (ISER)},
    year={2025},
    url={https://arxiv.org/abs/2506.19201}
    }
```