# PERGAMO

## Install instructions

IGL only supports and recommends the use of Anaconda. However, the environment can be set up using only `pip`
by installing the IGL bindings from source.

The general steps are as follows:

1. Install PyTorch according to your system
2. See the `requirements.txt` file to check the needed packages
    - This is usually done with `pip install -r requirements.txt`, but Anaconda may have a different way of doing things
3. Install IGL bindings ( https://github.com/libigl/libigl-python-bindings )
4. Install Kaolin ( https://kaolin.readthedocs.io/en/latest/notes/installation.html )

## Datasets

You can download two datasets from [OneDrive](https://urjc-my.sharepoint.com/:f:/r/personal/andres_casado_urjc_es/Documents/PERGAMO_public?csf=1&web=1&e=ObIEZ3).

DatosBuff contains BUFF sequences, while DatosDan contains our own data.

Datasets are made by processing each frame with:

- ExPose (output is SMPL-X, they need to be converted to SMPL too)
- PifuHD
- Self-Correction-Human-Parsing

## Running the project

To run the reconstruction, please check out `run_recons.sh`.

To run the regression, there are 2 sets of 3 scripts. Please check out `run_regression.sh` to see how it works.

