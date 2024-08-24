from typing import Any, Dict, List, Tuple, Type, Optional, Union
from sres.controller.workflow import WorkflowController

cname: str = "sres"
model: str = 'rcan-10-20-64'
ccustom: Dict[str,Any] = { 'task.data_downsample': 1 }
timestep: int = 0

configuration = dict(
	task = "SST-tiles-48",
	dataset = "swot_20-20e",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration, interp_loss=True )
controller.initialize( cname, model, **ccustom )

images_data, eval_losses = controller.inference( model, timestep, ccustom, save=True )


