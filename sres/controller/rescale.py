from sres.base.source.merra2.model import load_const_dataset, load_merra2_norm_data, load_dataset
from sres.base.util.ops import print_norms, vars3d
from xarray.core.resample import DataArrayResample
import xarray as xa, pandas as pd
import numpy as np
from sres.base.util.config import cfg
from sres.base.util.ops import dataset_to_stacked
from typing import List, Union, Tuple, Optional, Dict, Type, Any, Sequence, Mapping, Literal
import math, glob, sys, os, time, traceback
from xarray.core.dataset import DataVariables
from datetime import date
from sres.base.util.ops import get_levels_config, increasing, replace_nans
from sres.base.util.logging import lgm, exception_handled, log_timing
from sres.base.source.merra2.model import merge_batch
from sres.base.io.loader import ncFormat
from enum import Enum
from xarray.core.types import InterpOptions
np.set_printoptions(precision=3, suppress=False, linewidth=150)

class QType(Enum):
	Intensive = 'intensive'
	Extensive = 'extensive'

class DataLoader(object):

	def __init__(self,  **kwargs):
		self.c = cfg().task.coords
		self.format = ncFormat(cfg().task.get('nc_format', 'standard'))
		self.interp_method =  cfg().task.get('interp_method','linear')
		self.corder = ['time', 'z', 'y', 'x']
		self.xext, self.yext = cfg().preprocess.get('xext'), cfg().preprocess.get('yext')
		self.xres, self.yres = cfg().preprocess.get('xres'), cfg().preprocess.get('yres')
		self.levels: Optional[np.ndarray] = get_levels_config(cfg().preprocess)
		self.tstep = str(cfg().preprocess.data_timestep) + "h"
		self.dmap: Dict = cfg().preprocess.dims
		self.upscale_factor: int = cfg().model.get('scale_factor')
		self._constant_data: Dict[str, xa.Dataset] = {}
		self.norm_data: Dict[str, xa.Dataset] = load_merra2_norm_data()

	def close(self):
		for dset in self.norm_data.values(): dset.close()
		for dset in self._constant_data.values(): dset.close()

	def constant_data(self, vres: str ) -> xa.Dataset:
		return self._constant_data.setdefault( vres,  load_const_dataset( vres ) )

	def get_dataset(self, vres: str,  d: date, **kwargs ) -> xa.Dataset:
		time_index = kwargs.pop('time_index',-1)
		dset: xa.Dataset = load_dataset( vres, d ).squeeze( drop=True )
		if time_index >= 0: dset=dset.isel( tiles=time_index, drop=True )
		merged: xa.Dataset = merge_batch( [ dset ], self.constant_data(vres) )
		return merged

	def get_channel_array(self, vres: str, d: date, **kwargs) -> xa.DataArray:
		dset: xa.Dataset = self.get_dataset( vres, d, **kwargs )
		cvars = kwargs.pop('vars', vars3d(dset))
		dset = dset.drop_vars( set(dset.data_vars.keys()) - set(cvars) )
		result: xa.DataArray = self.ds2array( self.normalize( dset, **kwargs ) ).load()
		dset.close()
		return result

	def normalize(self, dset: xa.Dataset, **kwargs ) -> xa.Dataset:
		mean: xa.Dataset = self.norm_data['mean_by_level']
		std: xa.Dataset = self.norm_data['stddev_by_level']
		nvars: Dict[str,xa.DataArray] = {}
		for (vn,var) in dset.data_vars.items():
			if kwargs.get('interp_nan',False):
				var = var.interpolate_na( dim=self.c['x'], method='linear', keep_attrs=True )
				var = var.interpolate_na( dim=self.c['y'], method='linear', keep_attrs=True )
			nvars[vn] = (var-mean[vn])/std[vn]
		return xa.Dataset( nvars, dset.coords, dset.attrs )


	def ds2array( self, dset: xa.Dataset, **kwargs) -> xa.DataArray:
		aux_dims: List[str] =  [ "channels", self.c['t'], self.c['y'], self.c['x'] ]
		merge_dims = kwargs.get('merge_dims', [self.c['z']])
		sizes: Dict[str, int] = {}
		vnames = list(dset.data_vars.keys())
		vnames.sort()
		channels = []
		for vname in vnames:
			dvar: xa.DataArray = dset.data_vars[vname]
			if self.c['z'] in dvar.dims:    channels.extend([f"{vname}~{iL}" for iL in range(dvar.sizes[self.c['z']])])
			else:                           channels.append(vname)
			for (cname, coord) in dvar.coords.items():
				if cname not in (merge_dims + list(sizes.keys())):
					sizes[cname] = coord.size
		sizes.pop('datetime',None)
		darray: xa.DataArray = dataset_to_stacked(dset, sizes=sizes, preserved_dims=tuple(sizes.keys()))
		darray.attrs['channels'] = channels
		if self.c['t'] not in darray.dims: aux_dims.remove( "tiles" )
		return darray.transpose( *aux_dims )

	def interp_axis(self, dvar: xa.DataArray, coords: Dict[str, Any], axis: str):
		assert axis in ['x', 'y'], f"Invalid axis: {axis}"
		res, ext = (self.xres, self.xext) if (axis == 'x') else (self.yres, self.yext)
		if res is not None:
			if ext is None:
				c0 = dvar.coords[axis].values
				if axis == 'x':   self.xext = [c0[0], c0[-1]]
				else:           self.yext = [c0[0], c0[-1]]
			ext1 = ext[1] if axis == 'x' else ext[1] + res / 2
			coords[axis] = np.arange(ext[0], ext1, res)
		elif ext is not None:
			coords[axis] = slice(ext[0], ext[1])

	def interp_axes(self, dvar: xa.DataArray, subsample_coords: Dict[str, Dict[str, np.ndarray]], vres: str):
		coords: Dict[str, Any] = subsample_coords.setdefault(vres, {})
		if (self.levels is not None) and ('z' in dvar.dims):
			coords['z'] = self.levels
		for axis in ['x', 'y']:
			if vres == "high":
				self.interp_axis(dvar, coords, axis)
			else:
				hres_coords: Dict[str, np.ndarray] = subsample_coords['high']
				hres_axis = hres_coords[axis] if axis in hres_coords else dvar.coords[axis]
				coords[axis] = hres_axis[0::self.upscale_factor]

	def subsample_coords(self, dvar: xa.DataArray) -> Dict[str, Dict[str, np.ndarray]]:
		sscoords: Dict[str, Dict[str, np.ndarray]] = {}
		for vres in ["high", "low"]:
			if vres == "high" or self.format == ncFormat.SRES:
				self.interp_axes(dvar, sscoords, vres)
		return sscoords

	def upscale(self, variable: xa.DataArray, global_attrs: Dict, qtype: QType, isconst: bool) -> Dict[str, List[xa.DataArray]]:
		vhires = self.process_attrs( variable, global_attrs )
		if isconst and ("tiles" in variable.dims):
			vhires = vhires.isel(tiles=0, drop=True)
		if 'tiles' in vhires.dims:
			lgm().log( f" @@Resample {variable.name}{variable.dims}: shape={variable.shape}, tstep={self.tstep}")
			resampled: DataArrayResample = vhires.resample( dict(tiles=self.tstep), offset='0h' )
			vhires: xa.DataArray = resampled.mean() if qtype == QType.Intensive else resampled.sum()
		redop = np.mean if qtype == QType.Intensive else np.sum
		vlores: xa.DataArray = vhires
		scale_factor = math.prod( cfg().model.downscale_factors )
		for dim in [ 'x', 'y']:
			cargs = { dim: scale_factor }
			vlores = vlores.coarsen( boundary="trim", coord_func="min", **cargs ).reduce( redop, keep_attrs=True )

		return dict( high=[vhires], low=[vlores] )

	def process_attrs(self, variable: xa.DataArray, attrs: Dict ) -> xa.DataArray:
		cmap: Dict[str, str] = {cn0: cn1 for (cn0, cn1) in self.dmap.items() if cn0 in list(variable.coords.keys())}
		attrs1 = dict(**variable.attrs, **attrs)
		variable: xa.DataArray = variable.rename(**cmap)
		variable.attrs.update(attrs1)
		for missing in ['fmissing_value', 'missing_value', 'fill_value']:
			if missing in variable.attrs:
				missing_value = variable.attrs.pop('fmissing_value')
				variable = variable.where(variable != missing_value, np.nan)
		return replace_nans(variable).transpose(*self.corder, missing_dims="ignore")

	def rescale(self, variable: xa.DataArray, global_attrs: Dict, qtype: QType, isconst: bool) -> Dict[str, List[xa.DataArray]]:
		ncformat: ncFormat = ncFormat(cfg().task.nc_format)
		if ncformat == ncFormat.SRES:   return self.upscale(   variable, global_attrs, qtype, isconst )
		else:                           return self.subsample( variable, global_attrs, qtype, isconst )
	def subsample(self, variable: xa.DataArray, global_attrs: Dict, qtype: QType, isconst: bool) -> Dict[str, List[xa.DataArray]]:
		ssvars: Dict[str, List] = {}
		cmap: Dict[str, str] = {cn0: cn1 for (cn0, cn1) in self.dmap.items() if cn0 in list(variable.coords.keys())}
		print( f"Subsample input: cmap={cmap}, vdims={variable.dims}, vshape={variable.shape}")
		variable: xa.DataArray = variable.rename(**cmap)
		if isconst and ("tiles" in variable.dims):
			variable = variable.isel(tiles=0, drop=True)
		sscoords: Dict[str, Dict[str, np.ndarray]] = self.subsample_coords(variable)
		for vres, vcoord in sscoords.items():
			svars = ssvars.setdefault(vres, [])
			lgm().log(f" **** subsample {variable.name}:{vres}, vc={list(vcoord.keys())}, dims={variable.dims}, shape={variable.shape}, new sizes: { {cn: cv.size for cn, cv in vcoord.items()} }")
			varray: xa.DataArray = self._interp(variable, vcoord, global_attrs, qtype)
			svars.append(varray)
		return ssvars

	def _interp(self, variable: xa.DataArray, vcoord: Dict[str, np.ndarray], global_attrs: Dict, qtype: QType) -> xa.DataArray:
		varray = variable.interp( x=vcoord['x'], assume_sorted=True,  method=self.interp_method ) if 'x' in vcoord else variable
		varray =   varray.interp( y=vcoord['y'], assume_sorted=True,  method=self.interp_method ) if 'y' in vcoord else varray
		varray =   varray.interp( z=vcoord['z'], assume_sorted=False, method=self.interp_method ) if 'z' in vcoord else varray
		if 'time' in varray.dims:
			resampled: DataArrayResample = varray.resample(tiles=self.tstep)
			varray: xa.DataArray = resampled.mean() if qtype == QType.Intensive else resampled.sum()
		varray.attrs.update(global_attrs)
		varray.attrs.update(varray.attrs)
		for missing in ['fmissing_value', 'missing_value', 'fill_value']:
			if missing in varray.attrs:
				missing_value = varray.attrs.pop('fmissing_value')
				varray = varray.where(varray != missing_value, np.nan)
		return replace_nans(varray).transpose(*self.corder, missing_dims="ignore")