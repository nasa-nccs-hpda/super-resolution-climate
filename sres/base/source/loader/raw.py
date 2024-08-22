import math, os
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal
import time, numpy as np
from sres.data.tiles import TileGrid
import xarray as xa
from sres.base.util.logging import lgm, log_timing
from sres.base.util.config import cfg
from omegaconf import DictConfig, OmegaConf

class SRRawDataLoader(object):

	def __init__(self, config: DictConfig, **kwargs):
		self.config = config
		self.tile_grid = TileGrid()
		self.varnames: Dict[str, str] = self.config.input_variables

	@classmethod
	def get_loader(cls, task_config: DictConfig, **kwargs) -> 'SRRawDataLoader':
		dset: str = task_config.dataset
		if dset.startswith("swot"):
			from sres.base.source.swot.raw import SWOTRawDataLoader
			return SWOTRawDataLoader( task_config, **kwargs )

	def load_timeslice(self, **kwargs) -> xa.DataArray:
		raise NotImplementedError("SRRawDataLoader:load_timeslice")

	def get_batch_time_indices(self, **kwargs) -> xa.DataArray:
		raise NotImplementedError("SRRawDataLoader:get_batch_time_indices")
	@property
	def norm_stats(self) -> xa.Dataset:
		raise NotImplementedError("SRRawDataLoader:norm_stats")

	@property
	def global_norm_stats(self) -> xa.Dataset:
		raise NotImplementedError("SRRawDataLoader:global_norm_stats")
