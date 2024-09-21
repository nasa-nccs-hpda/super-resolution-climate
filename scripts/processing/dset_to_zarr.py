from typing import Any, Dict, List, Tuple, Type, Optional, Union
from sres.controller.workflow import WorkflowController
import xarray as xa
from datetime import datetime, timedelta
images_data: Dict[str,Dict[str,xa.DataArray]]
eval_losses: Dict[str,Dict[str,float]]

cname: str = "sres"
model: str = 'rcan-10-20-64'
ccustom: Dict[str,Any] = {}

configuration = dict(
	task = "SST-tiles-48",
	dataset = "swot_20-20e",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration )
controller.initialize( cname, model, **ccustom )
controller.to_zarr( )
