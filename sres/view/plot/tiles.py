import torch, numpy as np
import xarray as xa
from typing  import List, Tuple, Optional, Dict, Mapping
from sres.base.io.loader import batchDomain
from sres.controller.config import TSet, srRes
from sres.base.util.config import cfg
from sres.data.tiles import TileGrid
from sres.data.inference import load_inference_results
import ipywidgets as ipw
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from sres.data.inference import time_indices
from xarray.core.coordinates import DataArrayCoordinates
from sres.controller.dual_trainer import ModelTrainer
from sres.controller.config import TSet, ResultStructure
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

class ResultTilePlot(Plot):
	def __init__(self, trainer: ModelTrainer, tset: TSet, **kwargs):
		super(ResultTilePlot, self).__init__(trainer, **kwargs)
		self.tset: TSet = tset
		self.time_index: int = kwargs.get( 'time_id', 0 )
		self.tile_index: int = kwargs.get( 'tile_id', 0 )
		self.tileId: int = kwargs.get( 'tile_id', 0 )
		self.channel: str = kwargs.get( 'channel', trainer.target_variables[0] )
		self.splabels = [['input', self.upscale_plot_label], ['target', self.result_plot_label]]
		eval_results, eval_losses = self.update_tile_data(update_model=True)
		self.images_data: Dict[str, xa.DataArray] = eval_results
		self.losses: Dict[str, float] = eval_losses
		print( f" inference_data({self.channel}) time_index={self.time_index} tile_index={self.tile_index}:" )
		for rtype, rdata in eval_results.items():
			print(f"   ** {rtype}{rdata.dims}{rdata.shape}")
		print( f" eval_results[model]{eval_results['model'].dims}{eval_results['model'].shape}" )
		assert len(self.losses) > 0, "Aborting ResultPlot: Failed evaluation"
		self.tile_grid: TileGrid  = TileGrid()
		self.time_indices = time_indices(self.channel, ResultStructure.Tiles)
		self.time_index = min( self.time_index, len(self.time_indices)-1)
		self.tslider: StepSlider = StepSlider('Time:', self.time_index, len(self.time_indices) )
		self.sslider: StepSlider = StepSlider('Tile:', self.tile_index, cfg().task.batch_size )
		self.plot_titles: List[List[str]] = [ ['input', 'target'], ['interpolated', 'model'] ]
		self.ims = {}
		self.create_figure( nrows=2, ncols=2, sharex=True, sharey=True, title='SRes Loss Over Training Epochs' )
		self.panels = [ self.fig.canvas, self.tslider, self.sslider ]
		self.tslider.set_callback( self.time_update )
		self.sslider.set_callback( self.tile_update )

	@property
	def batch_domain(self) -> batchDomain:
		return self.trainer.batch_domain

	def update_tile_data( self, **kwargs ) -> Tuple[ Optional[ Mapping[str,xa.DataArray] ], Dict[str,float] ]:
		try:
			print( f"update_tile_data: tileId={self.tileId} time_index={self.time_index} ")
			eval_results, eval_losses = load_inference_results( self.channel, ResultStructure.Tiles, self.time_index )
			if len( eval_losses ) > 0:
				self.losses = eval_losses
				self.tile_index = self.tileId
			return eval_results, eval_losses
		except Exception as e:
			lgm().log( f"Exception in update_tile_data: {e}")
			return None, {}

	def select_tile(self, selected_tile: int):
		print(f" ---> selected_tile: {selected_tile}")
		if selected_tile is not None:
			self.tile_index = selected_tile
			self.images_data, loss = self.update_tile_data()
			self.update_subplots()
			lgm().log( f" ---> selected_tile = {selected_tile}")

	@property
	def upscale_plot_label(self) -> str:
		return "interpolated"

	@property
	def result_plot_label(self) -> str:
		return "model"

	def image(self, ir: int, ic: int) -> xa.DataArray:
		itype = self.splabels[ic][ir]
		image = self.images_data[itype]
		image.attrs['itype'] = itype
		return image

	@exception_handled
	def time_update(self, sindex: int):
		self.time_index = self.time_indices[ sindex ]
		lgm().log(f"\n time_update ---> sindex: {sindex} -> {self.time_index}")
		self.images_data, loss = self.update_tile_data()
		self.update_subplots()

	@exception_handled
	def tile_update(self, sindex: int):
		lgm().log( f" <-------------------------- tile_update ---> sindex = {sindex}" )
		self.tileId = sindex
		idata, loss = self.update_tile_data()
		if idata is not None:
			self.images_data = idata
			self.update_subplots()


	def plot( self ) -> ipw.Box:
		self.generate_subplots()
		print( f"Creating widget...")
		return ipw.VBox(self.panels)

	@property
	def display_time(self) -> str:
		return str(self.time_index)
	#	ctime: datetime = self.time_coords[self.time_index]
	#	return ctime.strftime("%m/%d/%Y:%H")

	def generate_subplots(self):
		self.fig.suptitle(f'Time: {self.display_time}, Tile: {self.tile_index}', fontsize=10, va="top", y=1.0)
		for irow in [0, 1]:
			for icol in [0, 1]:
				self.generate_subplot(irow, icol)
		self.fig.canvas.draw_idle()

	def update_subplots(self):
		self.fig.suptitle(f'Time: {self.display_time}, Tile: {self.tile_index}', fontsize=10, va="top", y=1.0)
		for irow in [0, 1]:
			for icol in [0, 1]:
				self.update_subplot(irow, icol)
		self.fig.canvas.draw_idle()

	def generate_subplot(self, irow: int, icol: int):
		ax: Axes = self.axs[irow, icol]
		ax.set_aspect(0.5)
		ts: Dict[str, int] = self.tile_grid.get_full_tile_size()
		ax.set_xlim([0, ts['x']])
		ax.set_ylim([0, ts['y']])
		image: Optional[xa.DataArray] = self.get_subplot_image(irow, icol, ts)
		vrange = cscale(image, 2.0)

		print( f"subplot_image[{irow}, {icol}]: image{image.dims}{image.shape}, vrange={vrange}")
		iplot: AxesImage =  image.plot.imshow(ax=ax, x="x", y="y", cmap='jet', yincrease=True, vmin=vrange[0], vmax=vrange[1])
		ax.set_title( self.get_subplot_title(irow,icol) )
		self.ims[ (irow, icol) ] = iplot

	def update_subplot(self, irow: int, icol: int):
		ts: Dict[str, int] = self.tile_grid.get_full_tile_size()
		image: Optional[xa.DataArray] = self.get_subplot_image(irow, icol, ts)
		if image is not None:
			self.ims[ (irow, icol) ].set_data(image.values)

	def get_subplot_title(self,irow,icol) -> str:
		label = self.plot_titles[irow][icol]
		rmserror = ""
		if irow == 1:
			print( f"get_subplot_title: label={label} losses={self.losses}")
			loss: float = float(self.losses.get(label,0.0))
			rmserror = f", RMSE: {loss*1000:.1f}"
		title = f"{label}{rmserror}"
		return title

	def get_subplot_image(self, irow: int, icol: int, ts: Dict[str, int] ) -> Optional[xa.DataArray]:
		image: xa.DataArray = self.image(irow, icol)
		if 'channels' in image.dims:
			print( f" get_subplot_image: image.dims={image.dims}, channel={self.channel}, image.channels={image.coords['channels'].values.tolist()}")
			image = image.sel( channels=self.channel, drop=True )
		if 'tiles' in image.dims:
			if self.batch_domain == batchDomain.Time:
				batch_time_index = self.time_index % self.trainer.get_ml_input(self.tset).shape[0]
				image = image.isel(tiles=batch_time_index).squeeze(drop=True)
			elif self.batch_domain == batchDomain.Tiles:
				lgm().log( f" Select tile {self.tile_index} from image_data{image.dims}{image.shape}" )
				image = image.isel(tiles=self.tile_index).squeeze(drop=True) if self.tile_index < image.sizes['tiles'] else None
		if image is not None:
			dx, dy = ts['x']/image.shape[-1], ts['y']/image.shape[-2]
			coords = dict( x=np.linspace(-dx/2, ts['x']+dx/2, image.shape[-1] ), y=np.linspace(-dy/2, ts['y']+dy/2, image.shape[-2] ) )
			image = image.assign_coords( coords )
		return image

