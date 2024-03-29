# PERGAMO

![Teaser](readme_assets/teaser.png "Teaser image")

[[Project website](https://mslab.es/projects/PERGAMO)]
[[Dataset](https://urjc-my.sharepoint.com/:f:/g/personal/andres_casado_urjc_es/EuNAwoSGWD5HtT6AsgL8vJcByupY0Tsx4n95vVlh0CDKsw)]
[[Video](https://www.youtube.com/watch?v=giaHHW6R6pk)]

## Abstract

> Clothing plays a fundamental role in digital humans. Current approaches to animate 3D garments are mostly based on
> realistic physics simulation, however, they typically suffer from two main issues: high computational run-time cost,
> which hinders their development; and simulation-to-real gap, which impedes the synthesis of specific real-world cloth
> samples. To circumvent both issues we propose PERGAMO, a data-driven approach to learn a deformable model for 3D
> garments from monocular images. To this end, we first introduce a novel method to reconstruct the 3D geometry of
> garments from a single image, and use it to build a dataset of clothing from monocular videos. We use these 3D
> reconstructions to train a regression model that accurately predicts how the garment deforms as a function of the
> underlying body pose. We show that our method is capable of producing garment animations that match the real-world
> behaviour, and generalizes to unseen body motions extracted from motion capture dataset.

## Install instructions

### Python dependencies

IGL only supports and recommends the use of Anaconda. However, the environment can be set up using only `pip`
by installing the IGL bindings from source.

The general steps are as follows:

1. Install PyTorch according to your system ( https://pytorch.org/get-started/locally/ )
2. See the `requirements.txt` file to check the needed packages
    - This is usually done with `pip install -r requirements.txt`, but Anaconda may have a different way of doing things
3. Install IGL bindings ( https://github.com/libigl/libigl-python-bindings )
4. Install Kaolin ( https://kaolin.readthedocs.io/en/latest/notes/installation.html )

### Models

- You can download the weights from
  [OneDrive](https://urjc-my.sharepoint.com/:f:/g/personal/andres_casado_urjc_es/EuNAwoSGWD5HtT6AsgL8vJcByupY0Tsx4n95vVlh0CDKsw)
  . Place the `weights` folder from OneDrive into the `data` folder of this repository.
- PERGAMO needs SMPL. You can download it from [SMPL](https://smpl.is.tue.mpg.de/). Rename the file from
  `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `smpl_neutral.pkl` and save it under `data/smpl/`.

## Running the project

To run the reconstruction, please check out `run_recons.sh`.

To run the regression, there are 2 sets of 3 scripts. Please check out `run_regression.sh` to see how it works.

### Visualizing regression results

The output is generated under `data` (`test_sequence` for AMASS scripts, `train/validation_sequence` for reconstructed
scripts).

To visualize using Blender, load the `.obj` file with the option `Geometry > Keep Vert Order`. Then, add a `Mesh Cache`
modifier to the loaded mesh. Change the type to `PC2` and then load the `.pc2` file adjacent to the `.obj`.

## Datasets

You can download a dataset from
[OneDrive](https://urjc-my.sharepoint.com/:f:/g/personal/andres_casado_urjc_es/EuNAwoSGWD5HtT6AsgL8vJcByupY0Tsx4n95vVlh0CDKsw)
.

### Structure

Each data set has the following folder hierarchy:

```
DataDanXXXXX
├─ clips (video files)
| ├─ dan-X01.mp4
| ├─ dan-X02.mp4
| ├─ ...
├─ reconstruction_input
| ├─ dan-X01
| | ├─ dan-X01 (video frames)
| | ├─ dan-X01_expose
| | ├─ dan-X01_parsing
| | ├─ dan-X01_pifu
| | ├─ dan-X01_smpl
| ├─ dan-X02
| | ├─ ...
| ├─ ...
├─ reconstruction_output (reconstructed garment meshes)
| ├─ dan-X01
| ├─ dan-X02
| ├─ ...
├─ regressor_training_data
├─ train_sequences
| ├─ meshes (reconstructed garment meshes in Tpose)
| | ├─ dan-X01
| | ├─ dan-X02
| | ├─ ...
| ├─ poses (encoded poses using the SoftSMPL encoding)
| | ├─ dan-X01
| | ├─ dan-X02
| | ├─ ...
├─ validation_sequences (same structure as train)
├─ ...
```

### For reconstruction

Datasets for the reconstruction script are made by processing each frame with:

- ExPose (output is SMPL-X, they need to be converted to SMPL too)
- PifuHD
- Self-Correction-Human-Parsing

The necessary files are provided in the reconstruction_input folder. We also provide reconstructed meshes for each
dataset (reconstruction_input folder) and the same meshes in Tpose space (inside the meshes folder on
regressor_training_data).

### For training

Our regressors predict wrinkles (vertex displacements with respect to a template mesh) from SMPL poses encoding using
the SoftSMPL encoding. We provide such encoded poses for the DataDanGrey dataset and also the scripts to generate such
encoding from arbitrary SMPL paramteres.

### For regression

You can use AMASS sequences by placing the `.npz` files under `data/test_sequence`.

Alternatively, you can run the regression on sequences of SMPL poses saved as `.pkl` files. Check the set
of `reconstructed` scripts.

## Citation

```
@article {casado2022pergamo,
    journal = {Computer Graphics Forum (Proc. of SCA), 2022},
    title = {{PERGAMO}: Personalized 3D Garments from Monocular video},
    author = {Casado-Elvira, Andrés and Comino Trinidad, Marc and Casas, Dan},
    year = {2022}
}
```
