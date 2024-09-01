import time
import xarray as xa
from sres.base.util.config import ConfigContext, cfg, config
from sres.controller.dual_trainer import ModelTrainer
from sres.base.util.logging import lgm, exception_handled, log_timing
from sres.data.inference import save_inference_results, load_inference_results
from sres.base.gpu import save_memory_snapshot
from sres.controller.config import TSet, ResultStructure
from typing import Any, Dict, List, Tuple
from sres.view.plot.tiles  import ResultTilePlot
from sres.view.plot.images import ResultImagePlot
from sres.view.plot.training import TrainingPlot
from sres.view.plot.base import Plot
import argparse

class WorkflowController(object):

	def __init__(self, cname: str, configuration: Dict[str,Any], **kwargs):
		self.cname = cname
		self.seed = kwargs.get('seed', int( time.time()/60 ) )
		self.interp_loss = kwargs.get('interp_loss', False)
		self.config: ConfigContext = None
		self.trainer: ModelTrainer = None
		self.plot: Plot = None
		self.model = None
		ConfigContext.set_defaults(**configuration)


	def train(self, models: List[str], **kwargs):
		for model in models:
			with ConfigContext(self.cname, model=model, **kwargs) as cc:
				try:
					self.config = cc
					args: argparse.Namespace = self.get_args()
					self.trainer = ModelTrainer(cc)
					self.trainer.train( args.nepochs, args.refresh, seed=self.seed, interp_loss=self.interp_loss )
				except Exception as e:
					lgm().exception( "Exception while training model: %s" % str(e) )
					save_memory_snapshot()

				lgm().log(f"Completed training model: {model}")

	def get_args(self) -> argparse.Namespace:
		argparser = argparse.ArgumentParser(description=f'Execute workflow {self.cname}')
		argparser.add_argument('-r', '--refresh', action='store_true', help="Refresh workflow by deleting existing checkpoints and learning stats")
		argparser.add_argument('-ne', '--nepochs', nargs='?', default=cfg().task.nepochs, type=int, help="Number of epochs to run training")
		return argparser.parse_args()

	def inference(self, timestep: int, data_structure: ResultStructure,  **kwargs)-> Tuple[Dict[str,Dict[str,xa.DataArray]], Dict[str,Dict[str,float]] ]:
			varnames = self.trainer.target_variables
			if   data_structure == ResultStructure.Image:
				image_results, eval_results = self.trainer.process_image(TSet.Validation, timestep, interp_loss=True, update_model=True, **kwargs)
			elif data_structure == ResultStructure.Tiles:
				image_results: Dict[str,Dict[str,xa.DataArray]] = {}
				eval_results:  Dict[str,Dict[str,float]] = {}
				condensed_image_results, condensed_eval_results = self.trainer.evaluate( TSet.Validation, time_index=timestep, update_checkpoint=False, update_model=True, **kwargs )
				if len(varnames) == 1:
					image_results = { varnames[0]: { k:v.squeeze() for k,v in condensed_image_results.items() } }
					eval_results =   { varnames[0]: condensed_eval_results }
				else:
					for varname in varnames:
						image_results[varname] = { k: imgdata.sel(channels=varname,drop=True) for k, imgdata in condensed_image_results.items() }
						eval_results[varname] = condensed_eval_results
			else:
				raise Exception( f"Unknown result structure: {data_structure}")
			if kwargs.get('save', True):
				for vname in varnames:
					image_data: Dict[str, xa.DataArray] = image_results[vname]
					eval_loss: Dict[str, float] = eval_results[vname]
					save_inference_results( vname, data_structure, image_data, timestep, eval_loss )
			return image_results, eval_results

	def initialize(self, cname, model, **kwargs ):
		self.model = model
		self.config = ConfigContext.activate_global( cname, model=model, **kwargs )
		lgm().log(f"Initialize WorkflowController({cname}), model={model}, config={kwargs}")
		self.trainer = ModelTrainer( self.config )

	def init_context(self, cc: ConfigContext, model: str ):
		self.model = model
		self.config = cc
		self.trainer = ModelTrainer( self.config )

	def get_result_tile_view(self, tset: TSet, **kwargs):
		self.plot = ResultTilePlot( self.trainer, tset, **kwargs)
		return self.plot.plot()

	def get_result_image_view(self, tset: TSet, varname: str, **kwargs):
		self.plot = ResultImagePlot( self.trainer, tset, varname, **kwargs)
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

