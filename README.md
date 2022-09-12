# Deep learning for modelling three dimensional magnetic static field

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## The following steps can be applied to reproduce the results of this research.

- Download and extract the weights of the deep learning model from https://osf.io/download/qshvw:

**In linux**
```bash
!wget https://osf.io/download/qshvw
!unzip qshvw
```

The folder contains the weights and "saved_model_nograd" should appear in the current working directory.

**Create new environment and install dependencies using Conda**

```bash
conda env create -f environment.yml
conda activate Obj3D
```

- Reproduce results
**Samples of generated data**
```bash
python sample_plot.py
```
**Result validation**
```bash
python test.py --component=0 # --component = 0, 1, 2 for axial, azimuthal and radial field components
```
<img src="images/Axial_component.png">

