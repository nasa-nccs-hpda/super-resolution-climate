import math, numpy as np
import xarray as xa
from typing  import List, Tuple, Optional, Dict
from sres.base.util.ops import xaformat_timedeltas
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import ipywidgets as ipw
from sres.base.util.logging import lgm, exception_handled

colors = ["red", "blue", "green", "cyan", "magenta", "yellow", "grey", "brown", "pink", "purple", "orange", "black"]

def flex(weight: int) -> ipw.Layout:
	return ipw.Layout(flex=f'1 {weight} auto', width='auto')
def rms( dvar: xa.DataArray, **kwargs ) -> float:
	varray: np.ndarray = dvar.isel( **kwargs, missing_dims="ignore", drop=True ).values
	return np.sqrt( np.mean( np.square( varray ) ) )

def rmse( diff: xa.DataArray, **kw ) -> xa.DataArray:
	rms_error = np.array( [ rms(diff, tiles=iT, **kw) for iT in range(diff.shape[0]) ] )
	return xa.DataArray( rms_error, dims=['time'], coords={'time': diff.time} )

def cscale( pvar: xa.DataArray, stretch: float = 2.0 ) -> Tuple[float,float]:
	meanv, stdv, minv = pvar.values.mean(), pvar.values.std(), pvar.values.min()
	vmin = max( minv, meanv - stretch*stdv )
	vmax = meanv + stretch*stdv
	return vmin, vmax

def normalize( target: xa.Dataset, vname: str, **kwargs ) -> xa.DataArray:
	statnames: Dict[str,str] = kwargs.get('statnames', dict(mean='mean', std='std'))
	norms: Dict[str,xa.Dataset] = kwargs.pop( 'norms', {} )
	fvar: xa.DataArray = target.data_vars[vname]
	if 'batch' in fvar.dims:  fvar = fvar.squeeze(dim="batch", drop=True)
	if len(norms) == 0: return fvar
	stats: Dict[str,xa.DataArray] = { stat: statdata.data_vars[vname] for stat,statdata in norms.items()}
	return (fvar-stats[ statnames['mean'] ]) / stats[ statnames['std'] ]

@exception_handled
def mplplot_error( target: xa.Dataset, forecast: xa.Dataset, vnames: List[str],  **kwargs ):
	ftime: np.ndarray = xaformat_timedeltas( target.coords['time'], form="day", strf=False ).values
	tvals = list( range( 1, round(float(ftime[-1]))+1, math.ceil(ftime.size/10) ) )
	with plt.ioff():
		fig, ax = plt.subplots(nrows=1, ncols=1,  figsize=[ 9, 6 ], layout="tight")

	for iv, vname in enumerate(vnames):
		tvar: xa.DataArray = normalize(target,vname,**kwargs)
		fvar: xa.DataArray = normalize(forecast,vname,**kwargs)
		error: xa.DataArray = rmse(tvar-fvar).assign_coords(tiles=ftime).rename( time = "time (days)")
		error.plot.line( ax=ax, color=colors[iv], label=vname )

	ax.set_title(f"  Forecast Error  ")
	ax.xaxis.set_major_locator(ticker.FixedLocator(tvals))
	ax.legend()
	return fig.canvas

