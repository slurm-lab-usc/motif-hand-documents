


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

## Reconstruction

for the code of reconstruction, you can follow this repo: 
https://github.com/louhz/robogs




## Code of thermal map reprojection and color assignment




The Thermal map  will first need to be processed by SAM2 and sift alignment.

Then utilize the heatmap reprojection code to assign and filter color to generate
thermal point cloud. 
https://github.com/louhz/robogs/blob/main/robogs/mesh_util/heatmap.py


### Code for SAM2 reprojection and assignment 




## Code for learn from human will be released soon 



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