import os, glob, torch
from .parse import parse
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Mapping
from .config import cfg
import shutil, xarray as xa
import numpy as np
from collections.abc import Iterable
from sres.base.util.logging import lgm, exception_handled
from torch import Tensor

def nnan(varray: np.ndarray) -> int: return np.count_nonzero( np.isnan( varray.flatten() ) )
def pctnan(varray: np.ndarray) -> str: return f"{nnan(varray) * 100.0 / varray.flatten().shape[0]:.2f}%"

def nnant(varray: Tensor) -> int: return torch.isnan(varray).sum().item()
def pctnant(varray: Tensor) -> str: return f"{nnant(varray)*100.0/torch.numel(varray):.2f}%"

def remove_filepath(filepath: str):
	if os.path.exists(filepath):
		try:
			os.remove(filepath)
			print("Removing file: ", filepath)
		except IsADirectoryError:
			shutil.rmtree(filepath)

ArrayOrTensor = Union[xa.DataArray,Tensor]

def xextent( raster: xa.DataArray ) -> Tuple[float,float,float,float]:
	xc, yc = raster.coords['lon'].values.tolist(), raster.coords['lat'].values.tolist()
	extent = xc[0], xc[-1]+(xc[1]-xc[0]), yc[0], yc[-1]+(yc[1]-yc[0])
	return extent

def dsextent( dset: xa.Dataset ) -> Tuple[float,float,float,float]:
	xc, yc = dset.lon.values.tolist(), dset.lat.values.tolist()
	extent = xc[0], xc[-1]+(xc[1]-xc[0]), yc[0], yc[-1]+(yc[1]-yc[0])
	return extent

def vrange(vdata: xa.DataArray) -> Tuple[float,float]:
	return vdata.min(skipna=True).values.tolist(), vdata.max(skipna=True).values.tolist()

def dsrange(vdata: xa.Dataset) -> Dict[str,Tuple[float,float]]:
	return { vid: vrange(v) for vid,v in vdata.data_vars.items() }

def year2date( year: Union[int,str] ) -> np.datetime64:
	return np.datetime64( int(year) - 1970, 'Y')

def extract_year( filename: str ) -> int:
	for template in cfg().platform.occ_files:
		fields = parse( template, filename )
		if (fields is not None) and ('year' in fields):
			try:     return int(fields['year'])
			except:  pass
	return 0

def extract_species( filename: str ) -> Optional[str]:
	for template in cfg().platform.occ_files:
		fields = parse( template, filename )
		if (fields is not None) and ('species' in fields):
			try:     return fields['species']
			except:  pass
	return None

def get_cfg_dates() -> List[np.datetime64]:
	return [year2date(y) for y in range(*cfg().platform.year_range) ]

def get_obs_dates() -> List[np.datetime64]:
	files = glob.glob(f"{cfg().platform.cov_data_dir}/*.jay")
	years = set([ extract_year(os.path.basename(file)) for file in files])
	return [year2date(y) for y in years if y > 0]

def get_dates( year_range: List[int] ) -> List[np.datetime64]:
	return [ year2date(y) for y in range(*year_range) ]

def get_obs_species() -> List[str]:
	files = glob.glob(f"{cfg().platform.occ_data_dir}/*.jay")
	species = set([ extract_species(os.path.basename(file)) for file in files])
	species.discard(None)
	return list(species)

def obs_dates_for_cov_date( covdate: np.datetime64 ) -> List[np.datetime64]:
	return [ covdate ]

def format_float_list(nval: List[float]) -> List[str]:
	return [ f"{x:.2f}" for x in nval ] if isinstance(nval, Iterable) else  [ str(nval) ]

def print_data_column( target: xa.Dataset, vname: str, **kwargs):
	ptype = kwargs.get("type", "")
	ttest_array: xa.DataArray = target.data_vars[vname]
	iargs = dict( lon=kwargs.get('lon',100), lat=kwargs.get('lat',100), tiles=kwargs.get('tiles',0))
	tdata = ttest_array.isel(**iargs).squeeze().values.tolist()
	print(f" ** {ptype} data column=> {vname}{ttest_array.dims}{ttest_array.shape}: {format_float_list(tdata)}")

def vars3d( target: xa.Dataset ) -> List[str]:
	return [ name for name,dvar in target.data_vars.items() if "level" in dvar.dims ]

def is_float( string: str ) -> bool:
	try: float(string); return True
	except ValueError:  return False

def find_key( d: Dict, v: str ) -> str:
	return list(d.keys())[ list(d.values()).index(v) ]

def is_int( string: str ) -> bool:
	try: int(string);  return True
	except ValueError: return False

def str2num( string: str ) -> Union[float,int,str]:
	try: return int(string)
	except ValueError:
		try: return float(string)
		except ValueError:
			return string

def xmin( v: xa.DataArray ):
	return v.min(skipna=True).values.tolist()

def xmax( v: xa.DataArray ):
	return v.max(skipna=True).values.tolist()

def xrng( v: xa.DataArray ):
	return [ xmin(v), xmax(v) ]

def srng( v: xa.DataArray ):
	return f"[{xmin(v):.5f}, {xmax(v):.5f}]"

def get_levels_config( config: Dict ) -> Optional[np.ndarray]:
	levs = config.get('levels')
	if levs is not None:
		levels = np.array(levs)
		levels.sort()
		return levels
	levr = config.get('level_range')
	if levr is not None:
		levels = np.arange(*levr)
		return levels


def increasing( data: np.ndarray ) -> bool:
	xl = data.tolist()
	return xl[-1] > xl[0]

def replace_nans(level_array: xa.DataArray) -> xa.DataArray:
	if nnan(level_array.values) > 0:
		level_array = level_array.interpolate_na(dim='x', method="linear", fill_value="extrapolate")
		if nnan(level_array.values) > 0:
			level_array = level_array.interpolate_na(dim='y', method="linear", fill_value="extrapolate")
		assert nnan(level_array.values) == 0, "NaNs remaining after replace_nans()"
	return level_array
def format_timedelta( td: np.timedelta64, form: str, strf: bool = True ) -> Union[str, float,int]:
	s = td.astype('timedelta64[s]').astype(np.int32)
	hours, remainder = divmod(s, 3600)
	if form == "full":
		minutes, seconds = divmod(remainder, 60)
		return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))
	elif form == "hr":
		return f'{hours}' if strf else hours
	elif form == "day":
		return f'{hours/24}' if strf else hours/24
	else: raise Exception( f"format_timedelta: unknown form: {form}" )

def xaformat_timedeltas( tds: xa.DataArray, form: str = "hr", strf: bool = True ) -> xa.DataArray:
	return xa.DataArray( [format_timedelta(td,form,strf) for td in tds.values] )

def format_timedeltas( tds: xa.DataArray, form: str = "hr", strf: bool = True ) -> str:
	if tds is None: return " NA "
	return str( xaformat_timedeltas(tds,form,strf).values.tolist() ).replace('"','')

def print_dict( title: str, data: Dict ):
	print( f"\n -----> {title}:")
	for k,v in data.items():
		print( f"   ** {k}: {v}")

def parse_file_parts(file_name):
	return dict(part.split("-", 1) for part in file_name.split("_"))

def sformat( param: str, params: Dict[str,str] ) -> str:
	try: return param.format(**params)
	except KeyError: return param
def pformat( param: Any, params: Dict[str,str] ) -> Union[str,Dict[str,str]]:
	if (type(param) is str) and ('{' in param): return sformat(param, params)
	if type(param) is dict: return { k: sformat(p, params) for k,p in param.items() }
	return param

def print_norms( norms: Dict[str, xa.Dataset] ):
	print(f"\n ----------------------------------------- Norm Data ----------------------------------------- ")
	for norm in norms.keys():
		print(f" >> Norm {norm}:")
		for k, ndata in norms[norm].data_vars.items():
			nval = ndata.values.tolist()
			print(f" >>>> {k}: {format_float_list(nval)}")


def variable_to_stacked( vname: str,  variable: xa.Variable, sizes: Mapping[str, int], preserved_dims: Tuple[str, ...] = ("tiles", "lat", "lon"), ) -> xa.Variable:
    """Converts an xa.Variable to preserved_dims + ("channels",).

	Any dimensions other than those included in preserved_dims get stacked into a
	final "channels" dimension. If any of the preserved_dims are missing then they
	are added, with the data broadcast/tiled to match the sizes specified in
	`sizes`.

	Args:
	  vname: Variable name.
	  variable: An xa.Variable.
	  sizes: Mapping including sizes for any dimensions which are not present in
		`variable` but are needed for the output. This may be needed for example
		for a static variable with only ("lat", "lon") dims, or if you want to
		encode just the latitude coordinates (a variable with dims ("lat",)).
	  preserved_dims: dimensions of variable to not be folded in channels.

	Returns:
	  An xa.Variable with dimensions preserved_dims + ("channels",).
	"""
    stack_to_channels_dims = [ d for d in variable.dims if d not in preserved_dims]
    dims = {dim: variable.sizes.get(dim) or sizes[dim] for dim in preserved_dims}
    lgm().log( f"#variable_to_stacked: {vname}{variable.dims}{variable.shape}: stack to channels {[ f'{d}[{variable.sizes.get(d,sizes.get(d,1))}]' for d in stack_to_channels_dims]}")
    if stack_to_channels_dims:
        variable = variable.stack(channels=stack_to_channels_dims)
    dims["channels"] = variable.sizes.get("channels", 1)
    lgm().log(f"  ****> stacked dvar {vname}{variable.dims}: {variable.shape}, preserved_dims={preserved_dims}")
    result = variable # variable.set_dims(dims)
    return result


def dataset_to_stacked( dataset: xa.Dataset, sizes: Optional[Mapping[str, int]] = None, preserved_dims: Tuple[str, ...] = ("tiles", "lat", "lon") ) -> xa.DataArray:
    """Converts an xa.Dataset to a single stacked array.

	This takes each consistuent data_var, converts it into BHWC layout
	using `variable_to_stacked`, then concats them all along the channels axis.

	Args:
	  dataset: An xa.Dataset.
	  sizes: Mapping including sizes for any dimensions which are not present in
		the `dataset` but are needed for the output. See variable_to_stacked.
	  preserved_dims: dimensions from the dataset that should not be folded in
		the predictions channels.

	Returns:
	  An xa.DataArray with dimensions preserved_dims + ("channels",).
	"""
    data_vars = [variable_to_stacked(name, dataset.variables[name], sizes or dataset.sizes, preserved_dims) for name in sorted(dataset.data_vars.keys())]
    lgm().debug(f"dataset_to_stacked: {len(dataset.data_vars)} data_vars, preserved_dims={preserved_dims}, concat-list size= {len(data_vars)}")
    coords = {dim: coord for dim, coord in dataset.coords.items() if dim in preserved_dims}
    stacked_data = xa.Variable.concat(data_vars, dim="channels")
    lgm().log(f"stacked_data{stacked_data.dims}: shape = {stacked_data.shape}, coords={list(coords.keys())}, dsattrs={list(dataset.attrs.keys())}")
    dims: List = list(stacked_data.dims).copy()
    vdata: np.ndarray = stacked_data.values
    if "channels" not in coords:
        coords["channels"] = np.arange( stacked_data.sizes["channels"], dtype=np.int32 )
    # print( f"\ndataset_to_stacked: vdata{dims}{vdata.shape} coords={list(coords.keys())}\n" )
    return xa.DataArray(data=vdata, dims=dims, coords=coords)


def stacked_to_dataset(
    stacked_array: xa.Variable,
    template_dataset: xa.Dataset,
    preserved_dims: Tuple[str, ...] = ("tiles", "lat", "lon"),
) -> xa.Dataset:
    """The inverse of dataset_to_stacked.

	Requires a template dataset to demonstrate the variables/shapes/coordinates
	required.
	All variables must have preserved_dims dimensions.

	Args:
	  stacked_array: Data in BHWC layout, encoded the same as dataset_to_stacked
		would if it was asked to encode `template_dataset`.
	  template_dataset: A template Dataset (or other mapping of DataArrays)
		demonstrating the shape of output required (variables, shapes,
		coordinates etc).
	  preserved_dims: dimensions from the target_template that were not folded in
		the predictions channels. The preserved_dims need to be a subset of the
		dims of all the variables of template_dataset.

	Returns:
	  An xa.Dataset (or other mapping of DataArrays) with the same shape and
	  type as template_dataset.
	"""
    unstack_from_channels_sizes = {}
    var_names = sorted(template_dataset.keys())
    for name in var_names:
        template_var = template_dataset[name]
        if not all(dim in template_var.dims for dim in preserved_dims):
            raise ValueError(
                f"stacked_to_dataset requires all Variables to have {preserved_dims} "
                f"dimensions, but found only {template_var.dims}.")
        unstack_from_channels_sizes[name] = {
            dim: size for dim, size in template_var.sizes.items()
            if dim not in preserved_dims}

    channels = {name: np.prod(list(unstack_sizes.values()), dtype=np.int64)
        for name, unstack_sizes in unstack_from_channels_sizes.items()}
    total_expected_channels = sum(channels.values())
    found_channels = stacked_array.sizes["channels"]
    if total_expected_channels != found_channels:
        raise ValueError(
            f"Expected {total_expected_channels} channels but found "
            f"{found_channels}, when trying to convert a stacked array of shape "
            f"{stacked_array.sizes} to a dataset of shape {template_dataset}.")

    data_vars = {}
    index = 0
    for name in var_names:
        template_var = template_dataset[name]
        var = stacked_array.isel({"channels": slice(index, index + channels[name])})
        index += channels[name]
        var = var.unstack({"channels": unstack_from_channels_sizes[name]})
        var = var.transpose(*template_var.dims)
        data_vars[name] = xa.DataArray(
            data=var,
            coords=template_var.coords,
            # This might not always be the same as the name it's keyed under; it
            # will refer to the original variable name, whereas the key might be
            # some alias e.g. temperature_850 under which it should be logged:
            name=template_var.name,
        )
    return type(template_dataset)(data_vars)  # pytype:disable=not-callable,wrong-arg-count

def normalize(values: xa.Dataset, scales: xa.Dataset, means: Optional[xa.Dataset], ) -> xa.Dataset:
    def normalize_array(array):
        if array.name is None:
            raise ValueError( "Can't look up normalization constants because array has no name.")
        if means is not None:
            if array.name in means:
                array = array - means[array.name].astype(array.dtype)
            else:
                print('No normalization location found for %s', array.name)
        if scales is not None:
            if array.name in scales:
                array = array / scales[array.name].astype(array.dtype)
            else:
                print('No normalization scale found for %s', array.name)
        return array
    data_vars = { vn: normalize_array(v) for vn, v in values.data_vars.items() }
    return xa.Dataset( data_vars, coords=values.coords, attrs=values.attrs )

def unnormalize(values: xa.Dataset, scales: xa.Dataset, means: Optional[xa.Dataset] ) -> xa.Dataset:
    """Unnormalize variables using the given scales and (optionally) means."""
    def unnormalize_array(array):
        if array.name is None:
            raise ValueError( "Can't look up normalization constants because array has no name.")
        if array.name in scales:
            array = array * scales[array.name].astype(array.dtype)
        else:
            print('No normalization scale found for %s', array.name)
        if means is not None:
            if array.name in means:
                array = array + means[array.name].astype(array.dtype)
            else:
                print('No normalization location found for %s', array.name)
        return array
    data_vars = { vn: unnormalize_array(v) for vn, v in values.data_vars.items() }
    return xa.Dataset( data_vars, coords=values.coords, attrs=values.attrs )
