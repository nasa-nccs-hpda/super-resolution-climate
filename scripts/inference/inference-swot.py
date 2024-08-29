from typing import Any, Dict, List, Tuple, Type, Optional, Union
from sres.controller.workflow import WorkflowController
import xarray as xa
from sres.controller.config import TSet, ResultStructure
images_data: Dict[str,Dict[str,xa.DataArray]]
eval_losses: Dict[str,Dict[str,float]]

cname: str = "sres"
model: str = 'rcan-10-20-64'
ccustom: Dict[str,Any] = {}
time_index_bounds = [ 0, 10 ]
data_structure: ResultStructure = ResultStructure.Tiles

configuration = dict(
	task = "SST-tiles-48",
	dataset = "swot_20-20e",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration, interp_loss=True )
controller.initialize( cname, model, **ccustom )

for timestep in list(range(*time_index_bounds)):
	inference_data, eval_losses = controller.inference( timestep, data_structure, save=True )

	print( f"Inference results for {configuration['dataset']}:{configuration['task']} timestep={timestep}, format={data_structure.value}:")
	for vname in inference_data.keys():
		print( f" * Variable {vname}:")
		var_data: Dict[str, xa.DataArray] = inference_data[vname]
		for dtype, darray in var_data.items():
			print( f"   -> {dtype+':':<8} array{darray.dims}{darray.shape}")




