import torch, time, traceback, pickle, shutil
from typing import Any, Dict, List, Optional
from sres.base.util.config import cfg
from sres.base.util.logging import lgm
from torch.optim.optimizer import Optimizer
from torch.nn import Module
from sres.controller.config import TSet, srRes
import os


class CheckpointManager(object):

	def __init__(self, model: Module, optimizer: Optimizer ):
		self._cpaths: Dict[str,str] = {}
		self.model = model
		self.optimizer = optimizer

	def save_checkpoint(self, epoch: int, itime: int, tset: TSet, loss: float, interp_loss: float  ) -> str:
		t0 = time.time()
		checkpoint = dict( epoch=epoch, itime=itime, model_state_dict=self.model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(), loss=loss )
		cpath = self.checkpoint_path(tset)
		if os.path.isfile(cpath):
			shutil.copyfile( cpath, self.checkpoint_path(tset,backup=True) )
		torch.save( checkpoint, cpath )
		lgm().log(f"\n *** SAVE {tset.name} checkpoint, loss={loss:.5f} ({interp_loss:.5f}), to {cpath}, dt={time.time()-t0:.4f} sec", display=True )
		return cpath

	def _load_state(self, tset: TSet ) -> Dict[str,Any]:
		# sdevice = f'cuda:{cfg().pipeline.gpu}' if torch.cuda.is_available() else 'cpu'
		cpath = self.checkpoint_path(tset)
		checkpoint = torch.load( cpath, map_location='cpu' ) # torch.device(sdevice) )
		return checkpoint

	def load_checkpoint( self, tset: TSet = TSet.Train, **kwargs ) -> Optional[Dict[str,Any]]:
		update_model = kwargs.get('update_model', False)
		cppath = self.checkpoint_path( tset )
		train_state = {}
		if os.path.exists( cppath ):
			try:
				train_state = self._load_state( tset )
				lgm().log(f"Loaded model checkpoint from {cppath}", display=True)
				if update_model:
					self.model.load_state_dict( train_state.pop('model_state_dict') )
					self.optimizer.load_state_dict( train_state.pop('optimizer_state_dict') )
			except Exception as e:
				lgm().log(f"Unable to load model from {cppath}: {e}", display=True)
				traceback.print_exc()
				return None
		else:
			print( f"No checkpoint file found at '{cppath}': starting from scratch.")
		print( f" ------ Saving checkpoints to '{cppath}' ------ " )
		return train_state

	def clear_checkpoints( self ):
		for tset in [ TSet.Train, TSet.Validation ]:
			cppath = self.checkpoint_path(tset)
			if os.path.exists(cppath):
				print( f" >> Clearing state: {cppath}")
				os.remove(cppath)

	@classmethod
	def checkpoint_path( cls, tset: TSet, backup=False ) -> str:
		vtset: TSet = TSet.Validation if (tset == TSet.Test) else tset
		cpath = f"{cfg().platform.results}/checkpoints/{cfg().task.training_version}.{vtset.value}"
		if backup: cpath = f"{cpath}.backup"
		os.makedirs(os.path.dirname(cpath), 0o777, exist_ok=True)
		return cpath + '.pt'

