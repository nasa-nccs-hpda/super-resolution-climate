import numpy as np, xarray as xa, math
import torch, dataclasses
from sres.base.util.config import cfg2meta, cfg
from typing import Iterable, List, Tuple, Union, Optional, Dict, Any, Sequence
from sres.base.util.ops import dataset_to_stacked
from sres.base.gpu import set_device, get_device
from sres.base.util.ops import format_timedeltas

TimedeltaLike = Any  # Something convertible to pd.Timedelta.
TimedeltaStr = str  # A string convertible to pd.Timedelta.

class TensorRole:
    INPUT = "input"
    TARGET = "target"
    PREDICTION = "prediction"

class TensorType:
    DALI = "dali"
    TORCH = "torch"

TargetLeadTimes = Union[
    TimedeltaLike,
    Sequence[TimedeltaLike],
    slice  # with TimedeltaLike as its start and stop.
]

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

DAY_PROGRESS = "day_progress"
YEAR_PROGRESS = "year_progress"
UPSAMPLE_COORDS = {}

def torch_interp_mode( downsample: bool ):
    mode = cfg().task.downsample_mode if downsample else cfg().task.upsample_mode
    if   mode == "linear": return "bilinear"
    elif mode == "cubic": return "bicubic"
    return mode
def get_timedeltas( dset: xa.Dataset ):
    return format_timedeltas( dset.coords["tiles"] )
Tensor = torch.Tensor

def d2xa( dvals: Dict[str,float] ) -> xa.Dataset:
    return xa.Dataset( {vn: xa.DataArray( np.array(dval) ) for vn, dval in dvals.items()} )

def ds2array( dset: xa.Dataset, **kwargs ) -> xa.DataArray:
    coords = cfg().task.coords
    merge_dims = kwargs.get( 'merge_dims', [coords['z'], coords['t']] )
    sizes: Dict[str,int] = {}
    vnames = list(dset.data_vars.keys()); vnames.sort()
    channels = []
    levels: np.ndarray = dset.coords[coords['z']].values
    for vname in vnames:
        dvar: xa.DataArray = dset.data_vars[vname]
        if coords['z'] in dvar.dims:    channels.extend([f"{vname}{int(levels[iL])}" for iL in range(dvar.sizes[coords['z']])])
        else:                           channels.append(vname)
        for (cname, coord) in dvar.coords.items():
            if cname not in (merge_dims + list(sizes.keys())):
                sizes[ cname ] = coord.size
    darray: xa.DataArray = dataset_to_stacked( dset, sizes=sizes, preserved_dims=tuple(sizes.keys()) )
    darray.attrs['channels'] = channels
    return darray.transpose( "tiles", "channels", coords['y'], coords['x'] )

def array2tensor( darray: Union[xa.DataArray,np.ndarray] ) -> Tensor:
    nparray: np.ndarray = darray.values if type(darray) is xa.DataArray else darray
    array_data: np.ndarray = np.ravel(nparray).reshape( nparray.shape )
    return torch.tensor( array_data, device=get_device(), requires_grad=True, dtype=torch.float32 )

def downsample( target_data: Union[xa.DataArray,Tensor], **kwargs) -> Tensor:
    scale_factor = kwargs.get('scale_factor', math.prod(cfg().model.downscale_factors))
    target_tensor: Tensor = array2tensor(target_data) if type(target_data) is xa.DataArray else target_data
    downsampled = torch.nn.functional.interpolate(target_tensor, scale_factor=1.0/scale_factor, mode=torch_interp_mode(True))
    return downsampled

def xa_downsample( input_array: xa.DataArray, **kwargs) -> xa.DataArray:
    scale_factor =  kwargs.get('scale_factor', math.prod(cfg().model.downscale_factors) )
    coords = { cn: input_array.coords[cn][::scale_factor] for cn in ['x', 'y'] }
    downsampled =  input_array.interp(coords=coords, method=cfg().task.downsample_mode, assume_sorted=True)
    return downsampled

def upsample( input_tensor: Tensor ) -> Tensor:
    scale_factor = math.prod(cfg().model.downscale_factors)
    upsampled = torch.nn.functional.interpolate(input_tensor, scale_factor=scale_factor, mode=torch_interp_mode(False))
    return upsampled

def xa_upsample(input_array: xa.DataArray, coords: Dict[str,np.ndarray]) -> xa.DataArray:
    csize = { cn:cv.shape for cn,cv in coords.items() }
    print( f"xa_upsample, input{input_array.shape}, coords{csize}")
    upsampled = input_array.interp(coords=coords, method=cfg().task.upsample_mode, assume_sorted=True)
    return upsampled

    #target_channels = cfg().task.target_variables
    # target_tensor: Tensor = array2tensor(target_data.sel(channel=target_channels))