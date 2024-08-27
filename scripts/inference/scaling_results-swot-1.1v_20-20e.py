from typing import Any, Dict, List, Tuple, Type, Optional, Union
from sres.base.util.config import ConfigContext, cfg
import numpy as np
from sres.controller.workflow import WorkflowController
import xarray as xa
from sres.controller.config import TSet, ResultStructure
images_data: Dict[str,Dict[str,xa.DataArray]]
eval_losses: Dict[str,Dict[str,float]]

cname: str = "sres"
model: str = 'rcan-10-20-64'
nts: int = 10
varname = "SST"
downsample_values = [1.0, 1.05] # [ 1.1, 1.3, 1.5, 2.0, 2.5, 3.0 ]
data_structure: ResultStructure = ResultStructure.Image

configuration = dict(
	task = f"{varname}-tiles-48",
	dataset = "swot_20-20e",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration, interp_loss=True )
downsample_results = {}
for data_downsample in downsample_values:
	ccustom: Dict[str,Any] = { 'task.data_downsample': data_downsample, 'task.xyflip': False }
	with ConfigContext( cname, model=model, **ccustom) as cc:
		controller.init_context( cc, model )
		loss_pct = []
		for its in range(nts):
			print( f"\n Run inference, data_downsample={data_downsample}, its={its}")
			infdata, evlosses = controller.inference( its, data_structure, save=False )
			inference_data: Dict[str,xa.DataArray] = infdata[varname]
			eval_losses: Dict[str,float] = evlosses[varname]
			model_loss, interp_loss = eval_losses['model'], eval_losses['interpolated']
			loss_pct.append( (model_loss/interp_loss)*100.0 )
		mean_loss_pct = np.array( loss_pct ).mean()
		downsample_results[data_downsample] =  mean_loss_pct

print( f"\n Scaling Results for {configuration['dataset']}:{configuration['task']}")
print( f" downsample_factor vs mean_loss_pct ")
for data_downsample, mean_loss_pct in downsample_results.items():
	print( f" {data_downsample:.2f}   {mean_loss_pct:.2f}")





