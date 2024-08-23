from typing import Any, Dict, List, Tuple, Type, Optional, Union
from sres.controller.workflow import WorkflowController

cname: str = "sres"
model: str = 'rcan-10-20-64'
ccustom: Dict[str,Any] = { 'task.nepochs': 100, 'task.lr': 1e-4 }
refresh =  False
timestep: int = 0

configuration = dict(
	task = "swot-1.1v",
	dataset = "swot_20-20e",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration, refresh_state=refresh, interp_loss=True )
images_data, eval_losses = controller.inference( model, timestep, ccustom, save=True )


