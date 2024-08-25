import xarray as xa, math, os, random
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal
from omegaconf import DictConfig, OmegaConf
from sres.base.source.loader.batch import SRDataLoader, FMDataLoader
import xarray as xa
import time, numpy as np
from typing import Dict, List, Optional, Union
from sres.base.util.dates import date_list, date_bounds
from datetime import datetime, date
from sres.base.util.logging import lgm, log_timing
from sres.base.util.config import cfg
from sres.base.io.loader import ncFormat, batchDomain
from base.controller.config import TSet, srRes

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR
predef_norms = [ 'year_progress', 'year_progress_sin', 'year_progress_cos', 'day_progress', 'day_progress_sin', 'day_progress_cos' ]
DAY_PROGRESS = "day_progress"
YEAR_PROGRESS = "year_progress"

class BatchType(Enum):
	Training = 'training'
	Forecast = 'forecast'

class VarType(Enum):
	Constant = 'constant'
	Dynamic = 'dynamic'

def xyflip(batch_data: xa.DataArray) -> xa.DataArray:
	bflip, flip_index = cfg().task.get('xyflip',False), 0
	if bflip:
		flip_index = random.randint(0, 3)
		if flip_index // 2 == 1: batch_data.reindex(x=batch_data.x[::-1])
		if flip_index  % 2 == 1: batch_data.reindex(y=batch_data.y[::-1])
	batch_data.attrs['xyflip'] = flip_index
	return batch_data

def idxarg( **kwargs ) -> Union[datetime,int]:
	if    'start_time' in kwargs: return kwargs['start_time']
	elif 'start_index' in kwargs: return kwargs['start_index']
def index_of_cval(  data: Union[xa.Dataset,xa.DataArray], dim:str, cval: float)-> int:
	coord: np.ndarray = data.coords[dim].values
	cindx: np.ndarray = np.where(coord==cval)[0]
	return cindx.tolist()[0] if cindx.size else -1

def get_index_roi(dataset: xa.Dataset, vres: str) -> Optional[Dict[str,slice]]:
	roi: Optional[Dict[str, List[float]]] = cfg().task.get('roi')
	if roi is None: return None
	cbounds: Dict = {}
	for dim in ['x', 'y']:
		if dim in roi:
			croi: List[float] = roi[dim]
			coord: List[float] = dataset.coords[dim].values.tolist()
			iroi: int =  index_of_cval( dataset, dim, croi[0] )
			crisize = round( (croi[1]-croi[0]) / (coord[1] - coord[0] ) )
			cbounds[dim] = slice( iroi, iroi + crisize )
	return cbounds

def furbish( dset: xa.Dataset ) -> xa.Dataset:
	dvars: Dict = { vname: dvar.squeeze() for vname, dvar in dset.data_vars.items() }
	coords: Dict = { cname: cval for cname, cval in dset.coords.items() }
	attrs: Dict = { aname: aval for aname, aval in dset.attrs }
	attrs['datetime'] = coords.pop('datetime', attrs.get('datetime',None) )
	return xa.Dataset( dvars, coords, attrs )

# def merge_batch( slices: List[xa.Dataset], constants: xa.Dataset ) -> xa.Dataset:
# 	dynamics: xa.Dataset = xa.concat( slices, dim="tiles", coords = "minimal" )
# 	return xa.merge( [dynamics, constants], compat='override' )

def get_target_steps(btype: BatchType):
	if btype == BatchType.Training: return cfg().task.train_steps
	elif btype == BatchType.Forecast: return cfg().task.eval_steps

def get_steps_per_day() -> int:
	hours_per_step = cfg().task.get('hours_per_step',0)
	if hours_per_step == 0: return 0
	steps_per_day: float = 24 / hours_per_step
	assert steps_per_day.is_integer(), "steps_per_day (24/data_timestep) must be an integer"
	return int(steps_per_day)

def get_days_per_batch(btype: BatchType):
	steps_per_day = get_steps_per_day()
	target_steps = get_target_steps( btype )
	batch_steps: int = cfg().task.nsteps_input + len(target_steps)
	if btype == BatchType.Training: return 1 + math.ceil((batch_steps - 1) / steps_per_day)
	elif btype == BatchType.Forecast: return math.ceil(batch_steps / steps_per_day)


def merge_temporal_batch( self, slices: List[xa.Dataset], constants: xa.Dataset) -> xa.Dataset:
	constant_vars: List[str] = self.task_config.get('constants',[])
	cvars = [vname for vname, vdata in slices[0].data_vars.items() if "tiles" not in vdata.dims]
	dynamics: xa.Dataset = xa.concat( slices, dim="tiles", coords = "minimal" )
	dynamics = dynamics.drop_vars(cvars)
	sample: xa.Dataset = slices[0].drop_dims( 'time', errors='ignore' )
	for vname, dvar in sample.data_vars.items():
		if vname not in dynamics.data_vars.keys():
			constants[vname] = dvar
		elif (vname in constant_vars) and ("tiles" in dvar.dims):
			dvar = dvar.mean(dim="tiles", skipna=True, keep_attrs=True)
			constants[vname] = dvar
	dynamics = dynamics.drop_vars(constant_vars, errors='ignore')
	return xa.merge( [dynamics, constants], compat='override' )

def load_predef_norm_data() -> Dict[str,xa.Dataset]:
	root, norms, drop_vars = cfg().platform.model, {}, None
	with open(f"{root}/stats/diffs_stddev_by_level.nc", "rb") as f:
		dset: xa.Dataset = xa.load_dataset(f)
		drop_vars = [ vname for vname in dset.data_vars.keys() if vname not in predef_norms ]
		norms['diffs_stddev_by_level']: xa.Dataset = dset.drop_vars( drop_vars ).compute()
	with open(f"{root}/stats/mean_by_level.nc", "rb") as f:
		dset: xa.Dataset = xa.load_dataset(f)
		drop_vars = [ vname for vname in dset.data_vars.keys() if vname not in predef_norms ]
		norms['mean_by_level']: xa.Dataset = dset.drop_vars( drop_vars ).compute()
	with open(f"{root}/stats/stddev_by_level.nc", "rb") as f:
		dset: xa.Dataset = xa.load_dataset(f)
		drop_vars = [ vname for vname in dset.data_vars.keys() if vname not in predef_norms ]
		norms['stddev_by_level']: xa.Dataset = dset.drop_vars( drop_vars ).compute()
	for nname, norm in norms.items():
		print( f" __________________ {nname} __________________ ")
		for (vname,darray) in norm.data_vars.items():
			print( f"   > {vname}: dims={darray.dims}, shape={darray.shape}, coords={list(darray.coords.keys())}  ")
	return norms

def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
	"""Computes year progress for times in seconds.

	Args:
	  seconds_since_epoch: Times in seconds since the "epoch" (the point at which
		UNIX time starts).

	Returns:
	  Year progress normalized to be in the [0, 1) interval for each time point.
	"""

	# Start with the pure integer division, and then float at the very end.
	# We will try to keep as much precision as possible.
	years_since_epoch = ( seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR) )
	# Note depending on how these ops are down, we may end up with a "weak_type"
	# which can cause issues in subtle ways, and hard to track here.
	# In any case, casting to float32 should get rid of the weak type.
	# [0, 1.) Interval.
	yp = np.mod(years_since_epoch, 1.0).astype(np.float32)
	return yp


def get_day_progress( seconds_since_epoch: np.ndarray, longitude: np.ndarray ) -> np.ndarray:
	"""Computes day progress for times in seconds at each longitude.

	Args:
	  seconds_since_epoch: 1D array of times in seconds since the 'epoch' (the
		point at which UNIX time starts).
	  longitude: 1D array of longitudes at which day progress is computed.

	Returns:
	  2D array of day progress values normalized to be in the [0, 1) inverval
		for each time point at each longitude.
	"""

	# [0.0, 1.0) Interval.
	day_progress_greenwich = ( np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY )

	# Offset the day progress to the longitude of each point on Earth.
	longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
	day_progress = np.mod( day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0 )
	return day_progress.astype(np.float32)


def featurize_progress( name: str, dims: Sequence[str], progress: np.ndarray ) -> Mapping[str, xa.Variable]:
	"""Derives features used by ML models from the `progress` variable.

	Args:
	  name: Base variable name from which features are derived.
	  fdims: List of the output feature dimensions, e.g. ("day", "lon").
	  progress: Progress variable values.

	Returns:
	  Dictionary of xa variables derived from the `progress` values. It
	  includes the original `progress` variable along with its sin and cos
	  transformations.

	Raises:
	  ValueError if the number of feature dimensions is not equal to the number
		of data dimensions.
	"""
	if len(dims) != progress.ndim:
		raise ValueError( f"Number of dimensions in feature {name}{dims} must be equal to the number of dimensions in progress{progress.shape}." )
	else: lgm().log( f"featurize_progress: {name}{dims} --> progress{progress.shape} ")

	progress_phase = progress * (2 * np.pi)
	return {
		name: xa.Variable(dims, progress),
		name + "_sin": xa.Variable(dims, np.sin(progress_phase)),
		name + "_cos": xa.Variable(dims, np.cos(progress_phase)),
	}

class FMBatch:

	def __init__(self, btype: BatchType, date_loader: FMDataLoader, **kwargs):
		self.format = ncFormat( cfg().task.get('nc_format', 'standard') )
		self.date_loader = date_loader
		self.type: BatchType = btype
		self.vres = kwargs.get('vres', "high" )
		self.days_per_batch = get_days_per_batch(btype)
		self.target_steps = get_target_steps(btype)
		self.batch_steps: int = cfg().task.nsteps_input + len(self.target_steps)
		self.constants: xa.Dataset = self.date_loader.load_const_dataset( **kwargs )
		#self.norm_data: Dict[str, xa.Dataset] = self.date_loader.load_norm_data()
		self.current_batch: xa.Dataset = None

	def load(self, d: date, **kwargs):
		bdays = date_list(d, self.days_per_batch)
		time_slices: List[xa.Dataset] = [ self.date_loader.load_dataset(d, self.vres) for d in bdays ]
		self.current_batch: xa.Dataset = merge_temporal_batch(time_slices, self.constants)

	def get_train_data(self,  day_offset: int ) -> xa.Dataset:
		return self.current_batch.isel( tiles=slice(day_offset, day_offset+self.batch_steps) )

	def get_time_slice(self,  day_offset: int) -> xa.Dataset:
		return self.current_batch.isel( tiles=day_offset )

	@classmethod
	def to_feature_array( cls, data_batch: xa.Dataset) -> xa.DataArray:
		features = xa.DataArray(data=list(data_batch.data_vars.keys()), name="features")
		result = xa.concat( list(data_batch.data_vars.values()), dim=features )
		result = result.transpose(..., "features")
		return result

class SRBatch:

	def __init__(self, task_config: DictConfig, tile_size: Dict[str, int], **kwargs):
		self.name = "target"
		self.tile_size: Dict[str, int] = tile_size
		self.data_loader: SRDataLoader = SRDataLoader.get_loader( task_config, tile_size, **kwargs )
		self.current_batch: xa.DataArray = None
		self.current_start_idx: Optional[Union[datetime,int]] = None
		self.current_origin = None
		self.days_per_batch = cfg().task.get('days_per_batch',0)
		self.batch_size = cfg().task.batch_size
		self.batch_domain: batchDomain = batchDomain.from_config( cfg().task.get('batch_domain', 'tiles'))
		self.batch_steps: int = self.days_per_batch * get_steps_per_day()
		self._constants: Optional[xa.Dataset] = None
		#self.norm_data: Dict[str, xa.Dataset] = self.data_loader.load_norm_data()
		self.channels: List[str] = None

	def get_batch_time_indices(self):
		return self.data_loader.get_batch_time_indices()

	def get_dset_size(self):
		return self.data_loader.get_dset_size()

	def load_global_timeslice(self, vid: str, **kwargs) -> np.ndarray:
		return self.data_loader.load_global_timeslice(vid,**kwargs)

	def constants(self, origin: Dict[str,int] )-> xa.Dataset:
		if self._constants is None:
			self._constants: xa.Dataset = self.data_loader.load_const_dataset(origin)
		return self._constants

	def merge_batch(self, slices: List[xa.Dataset]) -> xa.Dataset:
		dynamics: xa.Dataset = xa.concat(slices, dim="tiles", coords="minimal")
		merged: xa.Dataset =  xa.merge([dynamics, self.constants], compat='override')
		return merged

	def load_timeslice(self, ctime: Union[datetime, int], **kwargs) -> xa.DataArray:
		return self.data_loader.load_timeslice(ctime, **kwargs)

	def load_batch(self, ctile: Dict[str,int], ctime: Union[datetime,int]) -> Optional[xa.DataArray]:
		if self.batch_domain == batchDomain.Time:
			if type(ctime) == datetime:
				dates: Tuple[datetime,datetime] = date_bounds(ctime, self.days_per_batch)
				darray: xa.DataArray = self.data_loader.load_temporal_batch( ctile, dates )
				lgm().debug( f"load_batch[{ctime.strftime('%m/%d:%H/%Y')}]: {dates[0].strftime('%m/%d:%H/%Y')} -> {dates[1].strftime('%m/%d:%H/%Y')}, ndates={darray.sizes['time']}")
			elif type(ctime) == int:
				index_range: Tuple[int,int] = (ctime, ctime + self.batch_size )
				darray: xa.DataArray = self.data_loader.load_index_batch( ctile, index_range )
				lgm().log(f"load_batch: {index_range[0]} -> {index_range[1]}, data{darray.dims} shape={darray.shape}, ctile={list(ctile.values())}")
			else: raise Exception( f"'start_coord' in load_batch must be either int or datetime, not {type(ctime)}")
		elif self.batch_domain == batchDomain.Tiles:
			tile_range = (ctile['start'],ctile['end'])
			darray: xa.DataArray = self.data_loader.load_tile_batch( tile_range  )
		else:
			raise Exception(f"Unknown 'batch_domain' in load_batch: {self.batch_domain}")
		if self.channels is None:
			self.channels = darray.coords["channels"].values.tolist()
		return xyflip( darray )


	def load(self, ctile: Dict[str,int], ctime: Union[datetime,int] ) -> Optional[xa.DataArray]:
		t0 = time.time()
		cbatch: xa.DataArray = self.load_batch(ctile, ctime)
		if cbatch is not None:
			self.current_batch = cbatch
			self.current_start_idx = ctime
			self.current_origin = ctile
			xf = cbatch.attrs.get('xyflip',0)
			lgm().log( f" -----> load batch[{ctile}][{self.current_start_idx}]:{self.current_batch.dims}{self.current_batch.shape}[F{xf}], mean={cbatch.values.mean():.2f}, time = {time.time() - t0:.3f} sec" )
		return cbatch


