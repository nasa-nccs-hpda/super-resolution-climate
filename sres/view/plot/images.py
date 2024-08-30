import torch, numpy as np
import xarray as xa
from typing  import List, Tuple, Optional, Dict
from sres.base.io.loader import batchDomain
from sres.controller.config import TSet, srRes
from sres.base.util.config import cfg
from sres.base.util.array import array2tensor, downsample, upsample, xa_downsample, xa_upsample
import ipywidgets as ipw
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from xarray.core.coordinates import DataArrayCoordinates
from sres.controller.dual_trainer import ModelTrainer
from sres.controller.config import TSet
from sres.view.tile_selection_grid import TileSelectionGrid
from sres.view.plot.widgets import StepSlider
from sres.base.util.logging import lgm, exception_handled
from sres.view.plot.base import Plot

colors = ["red", "blue", "green", "cyan", "magenta", "yellow", "grey", "brown", "pink", "purple", "orange", "black"]

def flex(weight: int) -> ipw.Layout:
	return ipw.Layout(flex=f'1 {weight} auto', width='auto')
def rms( dvar: xa.DataArray, **kwargs ) -> float:
	varray: np.ndarray = dvar.isel( **kwargs, missing_dims="ignore", drop=True ).values
	return np.sqrt( np.mean( np.square( varray ) ) )

def tensor( dvar: xa.DataArray ) -> torch.Tensor:
	return torch.from_numpy( dvar.values.squeeze() )

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

def to_xa( template: xa.DataArray, data: np.ndarray ) -> xa.DataArray:
	return template.copy(data=data.reshape(template.shape))

class ResultImagePlot(Plot):
	def __init__(self, trainer: ModelTrainer, tset: TSet, varname: str, **kwargs):
		super(ResultImagePlot, self).__init__(trainer, **kwargs)
		self.tset: TSet = tset
		self.time_index: int = kwargs.get( 'time_id', 0 )
		self.losses = None
		self.varname = varname
		self.tileId: int = kwargs.get( 'tile_id', 0 )
		eval_results, eval_losses = self.update_tile_data(update_model=True)
		self.images_data: Dict[str, xa.DataArray] = eval_results
		print( f" eval_results[model]{eval_results['model'].dims}{eval_results['model'].shape}" )
		self.tslider: StepSlider = StepSlider('Time:', self.time_index, eval_results['model'].shape[0] )
		self.plot_titles: List[str] = list(self.images_data.keys())
		self.ims: Dict[int,AxesImage] = {}
		self.callbacks = dict(button_press_event=self.select_point)
		self.create_figure( nrows=4, ncols=1, callbacks=self.callbacks, title='SRes Loss Over Training Epochs' )
		self.tslider.set_callback( self.time_update )

	@property
	def sample_target(self) -> xa.DataArray:
		return self.trainer.get_sample_target()

	@property
	def tcoords(self) -> DataArrayCoordinates:
		return self.sample_target.coords

	@property
	def sample_input(self) -> xa.DataArray:
		return self.trainer.get_sample_input()

	@property
	def icoords(self) -> DataArrayCoordinates:
		return self.sample_input.coords

	@property
	def batch_domain(self) -> batchDomain:
		return self.trainer.batch_domain

	def update_tile_data( self, **kwargs ) -> Tuple[Dict[str, xa.DataArray],Dict[str,float]]:
		images_data, eval_losses = self.trainer.process_image( self.tset,  self.time_index, interp_loss=True, **kwargs )
		if len( eval_losses ) > 0:
			self.losses = eval_losses[self.varname]
			lgm().log(f"update_tile_data({self.varname}), time_index={self.time_index} ---> images = {list(images_data[self.varname].keys())}")
			return images_data[self.varname], self.losses

	def select_point(self,event):
		lgm().log(f'Mouse click: button={event.button}, dbl={event.dblclick}, x={event.xdata:.2f}, y={event.ydata:.2f}')

	@property
	def upscale_plot_label(self) -> str:
		return "interpolated"

	@property
	def result_plot_label(self) -> str:
		return "model"

	@exception_handled
	def time_update(self, sindex: int):
		lgm().log(f"\n time_update ---> sindex = {sindex}")
		self.time_index = sindex
		self.images_data, loss = self.update_tile_data()
		self.update_subplots()


	def plot( self ) -> ipw.Box:
		self.generate_subplots()
		panels = [self.fig.canvas, self.tslider]
		return ipw.VBox(panels)

	@property
	def display_time(self) -> str:
		return str(self.time_index)
	#	ctime: datetime = self.time_coords[self.time_index]
	#	return ctime.strftime("%m/%d/%Y:%H")

	def generate_subplots(self):
		self.fig.suptitle(f'Time: {self.display_time}', fontsize=10, va="top", y=1.0)
		for iplot in range(4):
				self.generate_subplot(iplot)
		self.fig.canvas.draw_idle()

	def update_subplots(self):
		self.fig.suptitle(f'Time: {self.display_time}', fontsize=10, va="top", y=1.0)
		for iplot in range(4):
			self.update_subplot(iplot)
		self.fig.canvas.draw_idle()

	#	cbar.clim(vmin, vmax)

	def generate_subplot(self, iplot: int):
		ax: Axes = self.axs[iplot]
		ax.set_aspect(0.5)
		ax.set_xlim([0, 100])
		ax.set_ylim([0, 100])
		ptype: str = self.plot_titles[iplot]
		image: xa.DataArray = self.images_data[ptype]
		vrange = [np.nanmin(image.values), np.nanmax(image.values)]
		lgm().log( f"subplot_image[{ptype}]: image{image.dims}{image.shape}, vrange={vrange}")
		axImage: AxesImage =  image.plot.imshow(ax=ax, x="x", y="y", cmap='jet', yincrease=True, add_colorbar=True ) #, vmin=vrange[0], vmax=vrange[1] )
		ax.set_title( self.get_subplot_title(ptype) )
		self.ims[ iplot ] = axImage

	def update_subplot(self, iplot: int):
		ptype: str = self.plot_titles[iplot]
		image: xa.DataArray = self.images_data[ptype]
		lgm().log( f" >> update_subplot_image[{ptype}]: image{image.dims}{image.shape}, ims: {list(self.ims.keys())}")
		self.ims[iplot].set_data(image.values)
		self.ims[iplot].changed()
		self.ims[iplot].stale = True

	def get_subplot_title(self, ptype: str) -> str:
		loss: float = None
		if   ptype == "interpolated": loss = self.losses.get("interpolated",0.0)
		elif ptype == "output": loss = self.losses.get('model', 0.0)
		return ptype if (loss is None) else f"{ptype}, loss={loss*1000:.3f}"



