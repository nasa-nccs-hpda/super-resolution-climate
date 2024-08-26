import os
import time
import xarray as xa
from sres.base.util.config import ConfigContext, cfg, config
from sres.base.util.logging import lgm, exception_handled, log_timing
from sres.controller.config import TSet, ResultStructure
from typing import Any, Dict, List, Tuple, Mapping

def results_path(varname: str, data_structure: ResultStructure):
	downsample_factor = float(cfg().task.data_downsample)
	dss = "" if (downsample_factor == 1.0) else f"_ds-{downsample_factor:.2f}"
	results_path = f"{cfg().platform.results}/inference/{config()['dataset']}/{config()['task']}/{varname}.{data_structure.value}{dss}.nc"
	os.makedirs(os.path.dirname(results_path), exist_ok=True)
	return results_path

def save_inference_results( varname: str, data_structure: ResultStructure, var_results: Dict[str ,xa.DataArray], var_losses: Dict[str ,float] ):
	dset = xa.Dataset(data_vars=var_results, attrs=dict(losses=var_losses) )
	rpath = results_path(varname, data_structure)
	print(f"Saving inference results to: {rpath}")
	dset.to_netcdf( rpath, "w")

def load_inference_result_dset( varname: str, data_structure: ResultStructure ) -> xa.Dataset:
	rpath = results_path(varname, data_structure)
	dset: xa.Dataset = xa.open_dataset( rpath )
	print(f"Loading inference results from: {rpath}")
	dset.attrs['varname'] = varname
	return dset

def load_inference_results( varname: str, data_structure: ResultStructure ) ->Tuple[Mapping[str ,xa.DataArray],Dict[str ,float]]:
	inference_result_dset: xa.Dataset = load_inference_result_dset( varname, data_structure)
	losses: Dict[str ,float] = inference_result_dset.attrs['losses']
	results = inference_result_dset.data_vars
	return results, losses