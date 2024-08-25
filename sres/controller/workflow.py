import os
import time
import xarray as xa
from sres.base.util.config import ConfigContext, cfg, config
from sres.controller.dual_trainer import ModelTrainer
from sres.base.util.logging import lgm, exception_handled, log_timing
from sres.base.gpu import save_memory_snapshot
from sres.base.io.loader import TSet, srRes
from typing import Any, Dict, List, Tuple
from sres.view.plot.tiles  import ResultTilePlot
from sres.view.plot.images import ResultImagePlot
from sres.view.plot.training import TrainingPlot
from sres.view.plot.base import Plot

class WorkflowController(object):

	def __init__(self, cname: str, configuration: Dict[str,Any], **kwargs):
		self.cname = cname
		self.seed = kwargs.get('seed', int( time.time()/60 ) )
		self.refresh_state = kwargs.get('refresh_state', False )
		self.interp_loss = kwargs.get('interp_loss', False)
		self.config: ConfigContext = None
		self.trainer: ModelTrainer = None
		self.plot: Plot = None
		self.model = None
		ConfigContext.set_defaults( **configuration )

	def train(self, models: List[str], **kwargs):
		for model in models:
			with ConfigContext(self.cname, model=model, **kwargs) as cc:
				try:
					self.config = cc
					self.trainer = ModelTrainer(cc)
					self.trainer.train(refresh_state=self.refresh_state, seed=self.seed, interp_loss=self.interp_loss)
				except Exception as e:
					lgm().exception( "Exception while training model: %s" % str(e) )
					save_memory_snapshot()

				lgm().log(f"Completed training model: {model}")

	def inference(self, timestep: int,  **kwargs)-> Tuple[Dict[str,Dict[str,xa.DataArray]], Dict[str,Dict[str,float]]]:
			images_data, eval_losses = self.trainer.process_image(TSet.Validation, timestep, interp_loss=True, update_model=True, **kwargs)
			if kwargs.get('save', True):
				self.save_results(images_data, eval_losses)
			return images_data, eval_losses

	def save_results(self, inference_data: Dict[str,Dict[str,xa.DataArray]], inference_losses: Dict[str,Dict[str,float]] ):
		for vname in inference_data.keys():
			var_results: Dict[str,xa.DataArray] = inference_data[vname]
			var_losses: Dict[str,float] =  inference_losses[vname]
			dset = xa.Dataset(data_vars=var_results, attrs=var_losses)
			results_path = f"{cfg().platform.results}/inference/{config()['dataset']}/{config()['task']}/{vname}.nc"
			os.makedirs( os.path.dirname(results_path), exist_ok=True )
			print( f"Saving inference results to: {results_path}")
			dset.to_netcdf( results_path, "w")

	def initialize(self, cname, model, **kwargs ):
		self.model = model
		self.config = ConfigContext.activate_global( cname, model=model, **kwargs )
		self.trainer = ModelTrainer( self.config )

	def get_result_tile_view(self, tset: TSet, **kwargs):
		self.plot = ResultTilePlot( self.trainer, tset, **kwargs)
		return self.plot.plot()

	def get_result_image_view(self, tset: TSet, **kwargs):
		self.plot = ResultImagePlot( self.trainer, tset, **kwargs)
		return self.plot.plot()

	def get_training_view(self, **kwargs):
		self.plot = TrainingPlot(self.trainer, **kwargs)
		return self.plot.plot()

	def test(self, model: str, test_name: str, **kwargs):
		with ConfigContext(self.cname, model=model, **kwargs) as cc:
			self.config = cc
			self.trainer = ModelTrainer(cc)
			if test_name == "load_raw_dataset":
				self.trainer.model_manager.sample_input()

