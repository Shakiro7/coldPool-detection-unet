# Cold Pool Detection U-Net

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8376599.svg)](https://doi.org/10.5281/zenodo.8376599)

This repository contains the code implementation for the segmentation of convective cold pools in simulated cloud and rainfall fields as described in the peer-reviewed article:

**Hoeller, J., Fiévet, R., Engelbrecht, E., & Haerter, J. O. (2024). U-Net segmentation for the detection of convective cold pools from cloud and rainfall fields. Journal of Geophysical Research: Atmospheres, 129, e2023JD040126.**  
📄 [Read the full article here](https://doi.org/10.1029/2023JD040126)

---

## 🧊 Overview

Convective cold pools (CPs) play a central role in organizing deep convection and influencing extreme weather events. However, their large-scale detection has long been limited due to the lack of suitable near-surface observations, especially over remote regions.

This repository provides a deep learning–based segmentation method that overcomes this limitation by relying only on cloud top temperature and rainfall fields — two variables globally available from geostationary satellites. By training U-Net architectures on high-resolution simulations, the method learns to recognize CP signatures based purely on their cloud and rainfall imprints.

In the future, this method might enable systematic, large-scale cold pool detection from spaceborne data, offering new opportunities to study convective organization across the tropics and beyond.

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/Shakiro7/coldPool-detection-unet.git
cd coldPool-detection-unet
```

### 2. Create a Virtual Environment *(optional but recommended)*

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Inference with the Pretrained Models

The datasets and pretrained models used in the publication are hosted on Zenodo:

📦 [Zenodo Dataset (10.5281/zenodo.8376598)](https://zenodo.org/records/8376599)

To generate sample predictions based on the data from the `input`-folder, download at least one of the pretrained neural networks (`*.pt`) and place it in the project directory. Update the paths in the `example_prediction.py` accordingly and run the script to perform segmentation:

```bash
python example_prediction.py
```

The script will predict and output the cold pool segmentation masks based on the input fields and the selected neural network.

---

## 📖 Citation

If you use this code or data in your work, please cite the following:

Paper
```bibtex
@article{https://doi.org/10.1029/2023JD040126,
author = {Hoeller, Jannik and Fiévet, Romain and Engelbrecht, Edward and Haerter, Jan O.},
title = {U-Net Segmentation for the Detection of Convective Cold Pools From Cloud and Rainfall Fields},
journal = {Journal of Geophysical Research: Atmospheres},
volume = {129},
number = {1},
pages = {e2023JD040126},
keywords = {cold pool detection, neural network, U-Net, segmentation, cloud and rainfall fields, convective organization},
doi = {https://doi.org/10.1029/2023JD040126},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2023JD040126},
eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2023JD040126},
note = {e2023JD040126 2023JD040126},
year = {2024}
}
```

Software and dataset
```bibtex
@software{hoeller_2023_8376599,
  author       = {Hoeller, Jannik and
                  Fiévet, Romain and
                  Engelbrecht, Edward and
                  Haerter, Jan},
  title        = {Cold Pool Detection U-Net},
  month        = sep,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.8376599},
  url          = {https://doi.org/10.5281/zenodo.8376599},
}
```

---

## 📬 Contact

For questions, suggestions, or collaborations, feel free to reach out.
