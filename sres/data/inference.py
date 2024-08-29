import os
import time, glob
import xarray as xa
from pathlib import Path
from sres.base.util.config import ConfigContext, cfg, config
from sres.base.util.logging import lgm, exception_handled, log_timing
from sres.controller.config import TSet, ResultStructure
from typing import Any, Dict, List, Tuple, Mapping, Union

def results_path(varname: str, timestep: int|str, data_structure: ResultStructure, **kwargs ):
	remove = kwargs.get('remove', False)
	print( f"cfg.task.keys: {list(cfg().task.keys())}")
	downsample_factor = float(cfg().task.data_downsample)
	dss = "" if (downsample_factor == 1.0) else f"_ds-{downsample_factor:.2f}"
	results_path = f"{cfg().platform.results}/inference/{config()['dataset']}/{config()['task']}/{varname}-{timestep}.{data_structure.value}{dss}.nc"
	os.makedirs(os.path.dirname(results_path), exist_ok=True)
	if remove and os.path.exists(results_path): os.remove(results_path)
	return results_path

def time_indices(varname: str, data_structure: ResultStructure)-> List[int]:
	glob_path =  results_path( varname, "*", data_structure)
	return [ int( Path(fn).stem.split(".")[0].split("-")[1] ) for fn in glob.glob(glob_path) ]

def save_inference_results( varname: str, data_structure: ResultStructure, var_results: Dict[str ,xa.DataArray], timestep: int, var_losses: Dict[str ,float] ):
	var_results['input'] = var_results['input'].rename( y='ys', x='xs')
	dset = xa.Dataset(data_vars=var_results, attrs=dict(loss_keys=list(var_losses.keys()), loss_values=list(var_losses.values())))
	rpath = results_path( varname, timestep, data_structure, remove=True )
	print(f"Saving inference results to: {rpath}, contents:")
	for  rtype, rdata in var_results.items():
		print(f" ** {rtype}{rdata.dims}{rdata.shape}")
	dset.to_netcdf( rpath, "w")

def load_inference_result_dset( varname: str, data_structure: ResultStructure, timestep: int ) -> xa.Dataset:
	rpath = results_path(varname, timestep, data_structure)
	dset: xa.Dataset = xa.open_dataset( rpath )
	print(f"Loading inference results from: {rpath}")
	dset.attrs['varname'] = varname
	return dset

@exception_handled
def load_inference_results( varname: str, data_structure: ResultStructure, timestep: int ) ->Tuple[Dict[str ,xa.DataArray],Dict[str ,float]]:
	print(f"load_inference_results for {varname}, {timestep}, {data_structure}")
	inference_result_dset: xa.Dataset = load_inference_result_dset( varname, data_structure, timestep )
	print(f"inference_result_dset: vars={list(inference_result_dset.data_vars.keys())}, attrs={inference_result_dset.attrs}")
	loss_data: List[List[str|float]] = [ inference_result_dset.attrs[aname] for aname in ['loss_keys','loss_values']]
	losses: Dict[str,float] = dict(zip(*loss_data))
	print( f" Loaded losses: {losses}")
	results: Dict[str ,xa.DataArray] = dict(inference_result_dset.data_vars)
	results['input'] = results['input'].rename(ys='y', xs='x')
	return results, losses