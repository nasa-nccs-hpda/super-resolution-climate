
# super-resolution-climate

Super Resolution for Weather/Climate Data: Test and Development Framework.

## Environment

If mamba is not available, install [miniforge](https://github.com/conda-forge/miniforge).
Execute the following to set up a conda environment for super-resolution-climate:

    >  mamba create -n sres python=3.11
    >  mamba activate sres
    >  mamba install -c conda-forge dask scipy xarray netCDF4 ipywidgets=7.8 jupyterlab=4.0 jupyterlab_widgets ipykernel=6.29 ipympl=0.9 ipython=8.26
    >  mamba install -c pytorch -c nvidia -c conda-forge litdata pytorch lightning lightning-utilities torchvision torchaudio pytorch-cuda cuda-python
    >  pip install parse  nvidia-dali-cuda120
    >  pip install hydra-core --upgrade
    >  ipython kernel install --user --name=sres

## Setup

    > git clone https://github.com/nasa-nccs-hpda/super-resolution-climate.git
    > cd super-resolution-climate/
    > setup.sh

## Configuration

This project uses [hydra](https://hydra.cc) for workflow configuration.  All configuration files are found in the super-resolution-climate/config directory.
Each workflow configuraration is composed of several sections, each with a separate config file. For example, in the sample script [train-rcan-swot-2.2v.py](./scripts/train-rcan-swot-2.2v.py), 
the *configuration* dict specifies the name of the config file to be used for each section, i.e. the *task* section is configured with the file [config/task/swot-2.2v.yaml](./config/task/swot-2.2v.yaml). 
The *ccustom* dict is used to override individual config values.  The *cname* parameter specifies the name of the root config file (e.g. [config/sres.yaml](./config/sres.yaml) )

## Training

The scripts under *super-resolution-climate/scripts* are used to train various super-resolution networks with various configurations. The notebook 
[super-resolution-climate/notebooks/plot_training.ipynb](./notebooks/plot_training.ipynb) is used to display a plot of 
loss vs. epochs for the configured training instance.

## Inference

The notebooks in [super-resolution-climate/notebooks](./notebooks) that plot the training results 
will execute the inference engine and display the result whenever the time or tile 
indices change (via sliders at the bottom).  The notebook [plot_result_tiles.ipynb](./notebooks/plot_result_tiles.ipynb) is 
used to explore the super-resolution results for individual tiles, and notebook [plot_result_images.ipynb](./notebooks/plot_result_images.ipynb) is 
used to display the assembled tiles for each region.











