# ECE901_Project: On the Possibility of NLOS Imaging Using Spatial Decomposition

![Image of Octree NLOS Reconstruction](https://github.com/elbrandt/ECE901_Project/blob/main/Paper/images/officescene_octree.PNG)

## Installation

### Set up Python environment (using Conda)
```
$ cd PyNLOS
$ conda env create --name pynlos --file env.yml
$ conda activate pynlos
```

### Download Dataset:

The datasets used in this project can be found [here](https://doi.org/10.6084/m9.figshare.8084987).

## Reconstructing at a particular resolution
Show help:
```
(pynlos) $ python backproj.py -h
usage: Backproj.py [-h] [-r RESOLUTION] [-o OUTFILE] [-np] infile

Performs NLOS Backprojection to reconstruct a NLOS scene

positional arguments:
  infile                the *.mat file that contains the input dataset

optional arguments:
  -h, --help            show this help message and exit
  -r RESOLUTION, --resolution RESOLUTION
                        the reconstruction resolution (in meters)
  -o OUTFILE, --outfile OUTFILE
                        filename to save the reconstructed cube (in numpy
                        format)
  -np, --noplot         do not plot output
```

Example reconstruction:

```
(pynlos) $ python Backproj.py officescene.mat -r 0.16 -o my_scene.npy
```

## Exploring a 3D Reconstruction

Show help:
```
(pynlos) $ python nlos_viz.py --help
usage: nlos_viz.py [-h] infile

Visualize a 3D NLOS Cube

positional arguments:
  infile      the *.mat or *.npy file that contains the reconstruction

optional arguments:
  -h, --help  show this help message and exit

While Running:
  T: Increase Threshold
  t: Decrease Threshold
  S: Increase threshold scaling factor
  s: Decrease threshold scaling factor
  O: Toggle Octree visualization On/Off
  D: Increase Octree depth
  d: Decrease Octree depth
  r: Reset view
```

Example visualization:
```
(pynlos) $ python nlos_viz.py officescene_0.010000_W.npy
```