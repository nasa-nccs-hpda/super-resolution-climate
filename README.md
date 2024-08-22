
# FMod

Super Resolution Test and Development Framework.

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
    > cd super-resolution-climate/notebooks/
    > ln -s ../sres ./sres
    > cd ../scripts
    > ln -s ../sres ./sres

## Inference

    Run the jupyter notebook: super-resolution-climate/notebooks/plot_results.ipynb

    This notebook executes the inference engine and displays the result 
    whenever the time or subtile indices change (via sliders at the bottom).










