import os
import time
import xarray as xa
from sres.base.util.config import ConfigContext, cfg, config
from sres.base.util.logging import lgm, exception_handled, log_timing
from sres.controller.config import TSet, ResultStructure
from typing import Any, Dict, List, Tuple

def save_inference_results( varname: str, data_structure: ResultStructure, var_results: Dict[str ,xa.DataArray], var_losses: Dict[str ,float] ):
	dset = xa.Dataset(data_vars=var_results, attrs=var_losses)
	results_path = f"{cfg().platform.results}/inference/{config()['dataset']}/{config()['task']}/{varname}.{data_structure.value}.nc"
	os.makedirs( os.path.dirname(results_path), exist_ok=True )
	print( f"Saving inference results to: {results_path}")
	dset.to_netcdf( results_path, "w")

def load_inference_results( varname: str, data_structure: ResultStructure ) -> xa.Dataset:
	results_path = f"{cfg().platform.results}/inference/{config()['dataset']}/{config()['task']}/{varname}.{data_structure.value}.nc"
	dset: xa.Dataset = xa.open_dataset( results_path )
	print(f"Loading inference results from: {results_path}")
	dset.attrs['varname'] = varname
	return dset