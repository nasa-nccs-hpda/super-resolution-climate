import xarray as xa, math, os
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
import xarray as xa
import time, numpy as np
from sres.base.util.dates import date_list
from datetime import date
from sres.base.util.logging import lgm, log_timing
from sres.base.util.config import cfg
from omegaconf import DictConfig, OmegaConf
from sres.base.util.ops import remove_filepath
from base.controller.config import TSet, srRes

class SRDataLoader(object):

	def __init__(self, task_config: DictConfig ):
		self.task = task_config
		self.dindxs = []

	def load_global_timeslice(self, vid: str, **kwargs) -> np.ndarray:
		raise NotImplementedError("SRDataLoader:load_global_timeslice")

	def load_timeslice(self, time_index: int, **kwargs) -> xa.DataArray:
		raise NotImplementedError("SRDataLoader:load_timeslice")

	def load_tile_batch(self, tile_range: Tuple[int,int] ) -> Optional[xa.DataArray]:
		raise NotImplementedError("SRDataLoader:load_tile_batch")

	def get_dset_size(self) -> int:
		raise NotImplementedError("SRDataLoader:get_dset_size")

	def load_norm_data(self)-> Dict[str, xa.Dataset]:
		raise NotImplementedError("SRDataLoader:load_norm_data")

	def load_temporal_batch(self, ctile: Dict[str,int], date_range: Tuple[datetime,datetime] ) -> xa.DataArray:
		raise NotImplementedError("SRDataLoader:load_temporal_batch")

	def load_index_batch(self, ctile: Dict[str,int], index_range: Tuple[int,int] ) -> xa.DataArray:
		raise NotImplementedError("SRDataLoader:load_index_batch")

	def load_const_dataset(self, ctile: Dict[str,int] ):
		raise NotImplementedError("SRDataLoader:load_const_dataset")

	def get_batch_time_indices(self):
		raise NotImplementedError("SRDataLoader:get_batch_time_indices")

	@classmethod
	def rcoords( cls, dset: xa.Dataset ):
		c = dset.coords
		return '[' + ','.join([f"{k}:{c[k].size}" for k in c.keys()]) + ']'

	@classmethod
	def get_loader(cls, task_config: DictConfig, tile_size: Dict[str, int], **kwargs) -> 'SRDataLoader':
		dset: str = task_config.dataset
		if dset.startswith("LLC4320"):
			from sres.base.source.s3export.batch import S3ExportDataLoader
			return S3ExportDataLoader( task_config, tile_size, **kwargs )
		elif dset.startswith("swot"):
			from sres.base.source.swot.batch import SWOTDataLoader
			return SWOTDataLoader( task_config, **kwargs )
		elif dset.startswith("merra2"):
			return None

class FMDataLoader(object):

	def load_norm_data(self)-> Dict[str, xa.Dataset]:
		raise NotImplementedError("SRDataLoader:load_norm_data")

	def load_dataset(self, d: date, vres: str):
		raise NotImplementedError("SRDataLoader:load_norm_data")

	def load_const_dataset(self, vres: str):
		raise NotImplementedError("SRDataLoader:load_norm_data")

	def rcoords(self, dset: xa.Dataset):
		raise NotImplementedError("SRDataLoader:rcoords")

	@classmethod
	def get_loader(cls, task_config: DictConfig, ** kwargs):
		pass