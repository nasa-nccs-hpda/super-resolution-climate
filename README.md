
# super-resolution-climate

Super Resolution for Weather/Climate Data: Test and Development Framework.

## Conda environment

    >   * mamba create -n sres python=3.11
    >   * mamba activate sres
    >   * mamba install -c conda-forge dask scipy xarray netCDF4 ipywidgets=7.8 jupyterlab=4.0 jupyterlab_widgets ipykernel=6.29 ipympl=0.9 ipython=8.26
    >   * mamba install -c pytorch -c nvidia -c conda-forge litdata pytorch lightning lightning-utilities torchvision torchaudio pytorch-cuda cuda-python
    >   * pip install parse  nvidia-dali-cuda120
    >   * pip install hydra-core --upgrade
    >   * ipython kernel install --user --name=sres

## Setup

    > git clone https://github.com/nasa-nccs-hpda/super-resolution-climate.git
    > cd super-resolution-climate/
    > setup.sh

## Configuration

This project uses [hydra](https://hydra.cc) for workflow configuration.  All configuration files are found in the super-resolution-climate/config directory.
Each workflow configuraration is composed of several sections, each with a separate config file. For example, in the sample script [train-rcan-swot-2.2v.py](./scripts/train-rcan-swot-2.2v.py), 
the *configuration* dict specifies the name of the config file to be used for each section, e.g. the *task* section is configured with the file [config/task/swot-2.2v.yaml](./config/task/swot-2.2v.yaml) 
The *ccustom* dict is used to override individual config values.  The *cname* parameter specifies the name of the root config file (e.g. [config/sres.yaml](./config/sres.yaml) )

## Inference

Run the jupyter notebook: super-resolution-climate/notebooks/plot_results.ipynb

This notebook executes the inference engine and displays the result 
whenever the time or subtile indices change (via sliders at the bottom).










