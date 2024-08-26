import torch, math
import xarray, traceback, random
from datetime import datetime
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union, Sequence, Optional
from sres.base.util.config import ConfigContext, cfg
from sres.data.tiles import TileIterator
from sres.base.io.loader import batchDomain
from sres.controller.config import TSet, srRes
from sres.base.util.config import cdelta, cfg, cval, get_data_coords, dateindex
from sres.base.gpu import set_device
from sres.base.util.array import array2tensor, downsample, upsample
from sres.data.batch import BatchDataset
from sres.base.util.dates import TimeType
from sres.model.manager import SRModels, ResultsAccumulator
from sres.base.util.logging import lgm
from sres.base.util.ops import pctnan, pctnant
from sres.controller.checkpoints import CheckpointManager
import numpy as np, xarray as xa
from sres.controller.stats import l2loss
import torch.nn as nn
from sres.base.gpu import save_memory_snapshot
import time, csv

Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]
MLTensors = Dict[ TSet, torch.Tensor]

def ttsplit_times( times: List[TimeType]) -> Dict[TSet, List[TimeType]]:
	ttsplit = cfg().task.ttsplit
	start, result, nt = 0, {}, len(times)
	for tset, tset_fraction in ttsplit.items():
		end = start + int(tset_fraction * nt)
		result[TSet(tset)] = times[start:end]
		print( f"Batch times[{tset}]: {len(result[TSet(tset)])}")
		start = end
	return result

def smean( data: xarray.DataArray, dims: List[str] = None ) -> str:
	means: np.ndarray = data.mean(dim=dims).values
	return str( [ f"{mv:.2f}" for mv in means ] )

def sstd( data: xarray.DataArray, dims: List[str] = None ) -> str:
	stds: np.ndarray = data.std(dim=dims).values
	return str( [ f"{mv:.2f}" for mv in stds ] )

def log_stats( name: str, data: xarray.DataArray, dims: List[str], display: bool = True):
	lgm().log(f' * {name} mean: {smean(data, dims)}', display=display)
	lgm().log(f' * {name} std:  { sstd(data, dims)}', display=display)

def mse( data: xarray.DataArray, target: xarray.DataArray, dims: List[str] ) -> xarray.DataArray:
	sdiff: xarray.DataArray = (target - data)**2
	return np.sqrt( sdiff.mean(dim=dims) )

def batch( members: List[xarray.DataArray] ) -> xarray.DataArray:
	return xarray.concat( members, dim="tiles" )

def npa( ts: TensorOrTensors ) -> np.ndarray:
	t = ts if type(ts) == Tensor else ts[-1]
	return t.detach().cpu().numpy()

def denorm( t: Tensor, norm_data: Dict[str,np.ndarray] ) -> np.ndarray:
	normed: np.ndarray = t.detach().cpu().numpy()
	# sshapes = { sn: sd.shape for sn, sd in norm_data.items() if (type(sd)==np.ndarray) }
	# print(f" ~~~~~~~~~~~~~~~~~~~ denorm->norm_data{normed.shape}, sshapes = {sshapes}" )
	if 'mean' in norm_data:
		normed = (normed*norm_data['std']) + norm_data['mean']
	if 'max' in norm_data:
		rng: np.ndarray = norm_data['max']-norm_data['min']
		normed = (normed*rng) + norm_data['min']
	# print(f" ~~~~~~~~~~~~~~~~~~~ denorm_data{normed.shape}: keys={list(norm_data.keys())} mean{norm_data['mean'].shape}={normed.mean():.2f} std{norm_data['std'].shape}={normed.std():.2f} ")
	return normed

def fmtfl( flist: List[float] ) -> str:
	svals = ','.join( [ f"{fv:.4f}" for fv in flist ] )
	return f"[{svals}]"

def shuffle( data: Tensor ) -> Tensor:
	idx = torch.randperm(data.shape[0])
	return data[idx,...]

def tas( ta: Any ) -> str:
	return list(ta) if (type(ta) is torch.Size) else ta

def ts( ts: TensorOrTensors ) -> str:
	if type(ts) == torch.Tensor: return tas(ts.shape)
	else:                        return str( [ tas(t.shape) for t in ts ] )

def unsqueeze( tensor: Tensor ) -> Tensor:
	if tensor.ndim == 2:
		tensor = torch.unsqueeze(tensor, 0)
		tensor = torch.unsqueeze(tensor, 0)
	elif tensor.ndim == 3:
		tensor = torch.unsqueeze(tensor, 1)
	return tensor

def normalize( tensor: Tensor ) -> Tensor:
	tensor = unsqueeze( tensor )
	tensor = tensor - tensor.mean(dim=[2,3], keepdim=True)
	return tensor / tensor.std(dim=[2,3], keepdim=True)

def downscale(self, origin: Dict[str,int] ):
	return { d: v*self.upsample_factor for d,v in origin.items()}

class ModelTrainer(object):

	model_cfg = ['batch_size', 'num_workers', 'persistent_workers' ]

	def __init__(self, cc: ConfigContext ):
		super(ModelTrainer, self).__init__()
		self.model_manager: SRModels = SRModels( set_device() )
		self.context: ConfigContext = cc
		self.device: torch.device = self.model_manager.device
		self.results_accum: ResultsAccumulator = ResultsAccumulator(cc)
		self.domain: batchDomain = batchDomain.from_config(cfg().task.get('batch_domain', 'tiles'))
		self.min_loss = float('inf')
		self.eps = 1e-6
		self._sht, self._isht = None, None
		self.scheduler = None
		self.model = self.model_manager.get_model( )
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg().task.lr, weight_decay=cfg().task.get('weight_decay', 0.0))
		self.checkpoint_manager = CheckpointManager(self.model, self.optimizer)
		self.loss_module: nn.Module = None
		self.layer_losses = []
		self.channel_idxs: torch.LongTensor = None
		self.target_variables = cfg().task.target_variables
		self.downscale_factors = cfg().model.downscale_factors
		self.scale_factor = math.prod(self.downscale_factors)
		self.conform_to_data_grid()
	#	self.grid_shape, self.gridops, self.lmax = self.configure_grid()
		self.input: np.ndarray = {}
		self.target: np.ndarray = {}
		self.product: np.ndarray = {}
		self.interp: np.ndarray = {}
		self.current_losses: Dict[str,float] = {}
		self.time_index: int = -1
		self.tile_index: int = -1
		self.validation_loss: float = float('inf')
		self.upsampled_loss: float = float('inf')
		self.data_timestamps: Dict[TSet,List[Union[datetime, int]]] = {}

	def to_xa(self, data: np.ndarray, upscaled: bool = False) -> xarray.DataArray:
		ustep: int = math.prod(cfg().model.downscale_factors)
		cscale = ustep if upscaled else 1
		coords = dict(tiles=np.arange(data.shape[0]), channels=np.array(self.target_variables))
		coords['y'] = np.arange(0, data.shape[2] * cscale, cscale)
		coords['x'] = np.arange(0, data.shape[3] * cscale, cscale)
		result = xa.DataArray(data.astype(np.float32), dims=['tiles', 'channels', 'y', 'x'], coords=coords)
		return result

	@property
	def model_name(self):
		return self.model_manager.model_name

	def get_dataset(self)-> BatchDataset:
		return self.model_manager.get_dataset()

	def get_sample_input(self, targets_only: bool = True) -> xa.DataArray:
		return self.model_manager.get_sample_input(targets_only)

	def get_sample_target(self) -> xa.DataArray:
		return self.model_manager.get_sample_target()


	# def configure_grid(self, tset: TSet ):
	# 	tar: xarray.DataArray = self.target_dataset(tset).get_current_batch_array()
	# 	grid_shape = tar.shape[-2:]
	# 	gridops = GridOps(*grid_shape,self.device)
	# 	lgm().log(f"SHAPES: target{list(tar.shape)}, (nlat, nlon)={grid_shape}")
	# 	lmax = tar.shape[-2]
	# 	return grid_shape, gridops, lmax

	def conform_to_data_grid(self, **kwargs):
		if cfg().task.conform_to_grid:
			data: xarray.DataArray = self.get_dataset().get_current_batch_array()
			data_origin: Dict[str, float] = get_data_coords(data, cfg().task['origin'])
			dc = cdelta(data)
			lgm().log(f"  ** snap_origin_to_data_grid: {cfg().task['origin']} -> {data_origin}", **kwargs)
			cfg().task['origin'] = data_origin
			cfg().task['extent'] = {dim: float(cval(data, dim, -1) + dc[cfg().task.coords[dim]]) for dim in data_origin.keys()}
			print(f" *** conform_to_data_grid: origin={cfg().task['origin']} extent={cfg().task['extent']} *** ")

	def tensor(self, data: xarray.DataArray) -> torch.Tensor:
		return Tensor(data.values).to(self.device)

	@property
	def loader_args(self) -> Dict[str, Any]:
		return { k: cfg().model.get(k) for k in self.model_cfg }

	def charbonnier(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
		error = torch.sqrt( ((prd - tar) ** 2) + self.eps )
		return torch.mean(error)

	def single_product_loss(self, prd: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
		if cfg().model.loss_fn == 'l2':
		#	print( f"LOSS: prd{list(prd.shape)}.mean={prd.mean()}, tar{list(tar.shape)}.mean={tar.mean()}")
			loss = l2loss(prd, tar)
		elif cfg().model.loss_fn == "charbonnier":
			loss = self.charbonnier(prd, tar)
		else:
			raise Exception("Unknown single-product loss function {}".format(cfg().model.loss_fn))
		return loss

	def get_multiscale_targets(self, hr_targ: Tensor) -> List[Tensor]:
		targets: List[Tensor] = [hr_targ]
		for usf in self.downscale_factors[:-1]:
			targets.append( torch.nn.functional.interpolate(targets[-1], scale_factor=1.0/usf, mode='bilinear') )
		targets.reverse()
		return targets

	def loss(self, products: TensorOrTensors, target: Tensor ) -> Tuple[float,torch.Tensor]:
		sloss, mloss, ptype, self.layer_losses = None, None, type(products), []
		if ptype == torch.Tensor:
			sloss = self.single_product_loss( products, target)
			mloss = sloss
		else:
			sloss = self.single_product_loss(products[-1], target)
			targets: List[Tensor] = self.get_multiscale_targets(target)
			for iL, (layer_output, layer_target) in enumerate( zip(products,targets)):
				layer_loss = self.single_product_loss(layer_output, layer_target)
				#		print( f"Layer-{iL}: Output{list(layer_output.shape)}, Target{list(layer_target.shape)}, loss={layer_loss.item():.5f}")
				mloss = layer_loss if (mloss is None) else (mloss + layer_loss)
				self.layer_losses.append( layer_loss.item() )
		return sloss.item(), mloss

	def load_timeslice(self, ctime: TimeType, **kwargs) -> Optional[xarray.DataArray]:
		return self.get_dataset().load_timeslice( ctime, **kwargs )

	@property
	def batch_domain(self) -> batchDomain:
		return self.get_dataset().batch_domain

	def get_srbatch(self, ctile: Dict[str,int], ctime: TimeType,  **kwargs  ) -> Optional[xarray.DataArray]:
		shuffle: bool = kwargs.pop('shuffle',False)
		btarget:  Optional[xarray.DataArray]  = self.get_dataset().get_batch_array(ctile,ctime,**kwargs)
		if btarget is not None:
			if shuffle:
				batch_perm: Tensor = torch.randperm(btarget.shape[0])
				btarget: xarray.DataArray = btarget[ batch_perm, ... ]
			lgm().log(f" *** target{btarget.dims}{btarget.shape}, mean={btarget.mean():.3f}, std={btarget.std():.3f}")
		return btarget

	def get_ml_input(self, tset: TSet) -> xa.DataArray:
		return  self.to_xa( self.input[tset], True )

	def get_ml_target(self, tset: TSet) -> xa.DataArray:
		return self.to_xa( self.target[tset] )

	def get_ml_product(self, tset: TSet) -> xa.DataArray:
		return self.to_xa( self.product[tset] )

	def get_ml_interp(self, tset: TSet) -> xa.DataArray:
		return self.to_xa( self.interp[tset] )

	def train(self, **kwargs) -> Dict[str, float]:
		if cfg().task['nepochs'] == 0: return {}
		refresh_state = kwargs.get('refresh_state', False)
		interp_loss = kwargs.get('interp_loss', False)
		seed = kwargs.get('seed', 4456)
		lossrec_flush_period = 32

		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.scheduler = kwargs.get('scheduler', None)
		epoch0, itime0, epoch_loss, nepochs, loss_history, eval_losses, tset, interp_sloss = 1, 0, 0.0, cfg().task.nepochs,  [], {}, TSet.Train, 0.0
		train_start = time.time()
		if refresh_state:
			self.checkpoint_manager.clear_checkpoints()
			if self.results_accum is not None:
				self.results_accum.refresh_state()
			print(" *** No checkpoint loaded: training from scratch *** ")
		else:
			train_state = self.checkpoint_manager.load_checkpoint( TSet.Train, update_model=True )
			if self.results_accum is not None:
				self.results_accum.load_results()
			epoch0 = train_state.get('epoch', 1 )
			itime0 = train_state.get( 'itime', 0 )
			epoch_loss = train_state.get('loss', float('inf'))
			nepochs += epoch0

		self.init_data_timestamps()
		for epoch in range(epoch0,nepochs):
			epoch_start = time.time()
			self.model.train()

			lgm().log(f"  ----------- Epoch {epoch}/{nepochs}   ----------- ", display=True )
			binput, boutput, btarget, nts = None, None, None, len(self.data_timestamps[TSet.Train])
			for itime in range (itime0,nts):
				ctime  = self.data_timestamps[TSet.Train][itime]
				timeslice: xa.DataArray = self.load_timeslice(ctime)
				tile_iter = TileIterator.get_iterator( ntiles=timeslice.sizes['tiles'], randomize=True )
				for ctile in iter(tile_iter):
					batch_data: Optional[xa.DataArray] = self.get_srbatch(ctile,ctime)
					if batch_data is None: break
					self.optimizer.zero_grad()
					binput, boutput, btarget = self.apply_network( batch_data )
					lgm().log(f"  ->apply_network: inp{binput.shape} target{ts(btarget)} prd{ts(boutput)}" )
					[sloss, mloss] = self.loss(boutput,btarget)
					tile_iter.register_loss( 'model', sloss )
					if interp_loss:
						binterp = upsample(binput)
						[interp_sloss, interp_multilevel_mloss] = self.loss(btarget, binterp)
						tile_iter.register_loss('interp', interp_sloss)
					stile = list(ctile.values())
					xyf = batch_data.attrs.get('xyflip',0)
					lgm().log(f" ** <{self.model_manager.model_name}> E({epoch:3}/{nepochs}) TIME[{itime:3}:{ctime:4}] TILES[{stile[0]:4}:{stile[1]:4}][F{xyf}]-> Loss= {sloss*1000:6.2f} ({interp_sloss*1000:6.2f}): {(sloss/interp_sloss)*100:.2f}%", display=True)
					mloss.backward()
					self.optimizer.step()


				if binput is not None:   self.input[tset] = binput.detach().cpu().numpy()
				if btarget is not None:  self.target[tset] = btarget.detach().cpu().numpy()
				if boutput is not None:  self.product[tset] = boutput.detach().cpu().numpy()
				[epoch_loss, interp_loss] = [ tile_iter.accumulate_loss(ltype) for ltype in ['model', 'interp']]
				self.checkpoint_manager.save_checkpoint(epoch, itime, TSet.Train, epoch_loss, interp_loss )
				self.results_accum.record_losses( TSet.Train, epoch-1+itime/nts, epoch_loss, interp_loss, flush=((itime+1) % lossrec_flush_period == 0) )

			if self.scheduler is not None:
				self.scheduler.step()

			epoch_time = (time.time() - epoch_start)/60.0
			lgm().log(f'Epoch Execution time: {epoch_time:.1f} min, train-loss: {epoch_loss:.4f}', display=True)
			self.record_eval( epoch, {TSet.Train: epoch_loss}, TSet.Validation )
			save_memory_snapshot()

		train_time = time.time() - train_start
		ntotal_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		self.record_eval( nepochs, {},  TSet.Test )
		print(f'\n -------> Training model with {ntotal_params} wts took {train_time/60:.2f} ({train_time/(60*nepochs):.2f} per epoch) min.')
		self.current_losses = dict( prediction=epoch_loss, **eval_losses )
		return self.current_losses

	def record_eval(self, epoch: int, losses: Dict[TSet,float], tset: TSet, **kwargs ):
		eval_results, eval_losses = self.evaluate( tset, update_model=False, **kwargs )
		if len(eval_losses) > 0:
			if self.results_accum is not None:
				print( f" --->> record {tset.name} eval[{epoch}]: eval_losses={eval_losses}, losses={losses}")
				self.results_accum.record_losses( tset, epoch, eval_losses['model'], eval_losses['interp'] )
			if kwargs.get('flush',True):
				self.results_accum.flush()
		return eval_losses

	def init_data_timestamps(self):
		if len(self.data_timestamps) == 0:
			ctimes: List[TimeType] = self.get_dataset().get_batch_time_coords()
			lgm().log( f"init_data_timestamps: {len(ctimes)} times", display=True)
			self.data_timestamps = ttsplit_times(ctimes)

	def tile_in_batch(self, itile, ctile ):
		if self.tile_index < 0: return True
		if self.batch_domain == batchDomain.Time:
			return self.tile_index == itile
		elif self.batch_domain == batchDomain.Tiles:
			tile_range = range(ctile['start'], ctile['end'])
			return self.tile_index in tile_range

	def process_image(self, tset: TSet, itime: int, **kwargs) -> Tuple[Dict[str,Dict[str,xa.DataArray]], Dict[str,Dict[str,float]]]:
		seed = kwargs.get('seed', 333)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.time_index = itime
		train_state = self.checkpoint_manager.load_checkpoint( TSet.Validation, **kwargs )
		if train_state is None:
			print( "Error loading checkpoint file, skipping evaluation.")
			return {},{}
		self.validation_loss = train_state.get('loss', float('inf'))
		proc_start = time.time()
		lgm().log(f" ##### process_image({tset.value}): time_index={self.time_index} ##### ")
		self.init_data_timestamps()
		batch_model_losses, batch_interp_losses, interp_sloss, ibatch, batches = [], [], 0.0, 0, []
		ctime = self.data_timestamps[TSet.Train][itime]
		timeslice: xa.DataArray = self.load_timeslice(ctime)
		vnames, nvars, cvar = self.target_variables, len(self.target_variables), kwargs.get('var',None)
		output_vars = [ cvar ] if cvar is not None else vnames
		print( f"Loaded timeslice{timeslice.dims}{timeslice.shape}, mean={np.nanmean(timeslice.values)}:.3f")
		tile_iter = TileIterator.get_iterator( ntiles=timeslice.sizes['tiles'] )
		for itile, ctile in enumerate(iter(tile_iter)):
			lgm().log(f"     -----------------    evaluate[{tset.name}]: ctime[{itime}]={ctime}, time_index={self.time_index}, ctile[{itile}]={ctile}", display=True)
			batch_data: Optional[xa.DataArray] = self.get_srbatch(ctile, ctime, shuffle=False)
			if batch_data is None: break
			# print( f" --> batch_data{list(batch_data.shape)} mean={batch_data.values.mean()}")

			binput, boutput, btarget = self.apply_network( batch_data )
			if binput is not None:
				binterp = upsample(binput)
				lgm().log(f"  ->apply_network: inp{ts(binput)} target{ts(btarget)} prd{ts(boutput)} interp{ts(binterp)}", display=True )
				[model_sloss, model_multilevel_loss] = self.loss(boutput, btarget)
				batch_model_losses.append( model_sloss )
				[interp_sloss, interp_multilevel_mloss] = self.loss(binterp,btarget)
				batch_interp_losses.append( interp_sloss )
				xyf = batch_data.attrs.get('xyflip', 0)
				sloss = batch_model_losses[-1]
				lgm().log(f" **  ** <{self.model_manager.model_name}:{tset.name}> BATCH[{ibatch:3}]{batch_data.shape} TIME[{itime:3}:{ctime:4}] TILES{list(ctile.values())}[F{xyf}]-> Loss= {sloss*1000:5.1f} ({interp_sloss*1000:5.1f}): {(sloss/interp_sloss)*100:.2f}%", display=True )
				ibatch = ibatch + 1
				batches.append( dict(input=denorm(binput,batch_data.attrs), target=denorm(btarget,batch_data.attrs), interpolated=denorm(binterp,batch_data.attrs), model=denorm(boutput,batch_data.attrs)) )

		images, losses = {}, {}
		for ivar, vname in enumerate(output_vars):
			images[vname] = self.assemble_images( batches, ivar, timeslice.coords['tiles'].values, timeslice.attrs['grid_shape'] )
			proc_time = time.time() - proc_start
			lgm().log(f" --- batch_model_losses = {batch_model_losses}")
			lgm().log(f" --- batch_interp_losses = {batch_interp_losses}")
			model_loss: float = np.array(batch_model_losses).mean()
			ntotal_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
			lgm().log(f' -------> Exec {tset.value} model with {ntotal_params} wts on {tset.value} tset took {proc_time:.2f} sec, model loss = {model_loss:.4f}')
			losses[vname] = dict( model=model_loss, interpolated=np.array(batch_interp_losses).mean() )
		return images, losses

	def assemble_images(self, batches: List[Dict[str,np.ndarray]], ivar: int, tile_ids: np.ndarray, grid_shape: Dict[str, int] ) -> Dict[str,xa.DataArray]:
		assembled_images: Dict[str,xa.DataArray] = {}
		nb: int =  len(batches)
		itypes: List[str] = list(batches[0].keys())
		print(f"Assembling {nb} batches with tile_idxs{tile_ids.shape}, grid_shape{grid_shape}, itypes={itypes}")

		for ii, image_type in enumerate(itypes):
			bsize, tidx0, tidx1 = None, 0, 0
			block_grid: List[List[np.ndarray]] = None
			for ib in range(nb):
				vbatch: np.ndarray = batches[ib][image_type]
				batch: np.ndarray = vbatch[:,ivar,:,:]
				tile_shape = list(batch.shape[-2:])
				bsize = batch.shape[0]
				if block_grid is None:
					empty_tile = np.full(tile_shape, np.nan)
					block_grid = [[empty_tile]*grid_shape['x'] for i in range(grid_shape['y'])]
				tidx1 = tidx0 + bsize
				# print(f" ** Processing batch[{ii}][{ib}]: shape={batch.shape}, size={bsize}, tids=[{tidx0},{tidx1}]")
				for bidx, tidx in enumerate(range(tidx0, tidx1)):
					tid = int(tile_ids[tidx])
					tc = dict( y=tid//grid_shape['x'], x=tid%grid_shape['x'] )
					block: np.ndarray = batch[bidx].squeeze()
					# print( f" ---> batch[{ib}][{bidx}] tidx={tidx} tid={tid} tc=[{tc['y']},{tc['x']}], block{list(block.shape)}")
					block_grid[ tc['y'] ][ tc['x'] ] = block
				tidx0 = tidx1
			image_data = np.block( block_grid )
			dims, bnds = ['y', 'x'], [0.0,100.0]
			coords = { cn: np.arange( bnds[0],bnds[1],(bnds[1]-bnds[0])/image_data.shape[ic]) for ic,cn in enumerate(dims) }
			assembled_images[image_type] = xa.DataArray(  image_data, dims=dims, coords=coords )

		return assembled_images


	def evaluate(self, tset: TSet, **kwargs) -> Tuple[Dict[str,xa.DataArray],Dict[str,float]]:
		seed = kwargs.get('seed', 333)
		assert tset in [ TSet.Validation, TSet.Test ], f"Invalid tset in training evaluation: {tset.name}"
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		self.time_index = kwargs.get('time_index', self.time_index)
		self.tile_index = kwargs.get('tile_index', self.tile_index)
		save_checkpoint = kwargs.get('save_checkpoint', True)
		train_state = self.checkpoint_manager.load_checkpoint( TSet.Validation, **kwargs )
		if train_state is None:
			print( "Error loading checkpoint file, skipping evaluation.")
			return {}
		self.validation_loss = train_state.get('loss', float('inf'))
		epoch = train_state.get( 'epoch', 0 )
		self.init_data_timestamps()
		proc_start = time.time()
		lgm().log(f" ##### evaluate({tset.value}): time_index={self.time_index}, tile_index={self.tile_index} ##### ")

		batch_model_losses, batch_interp_losses, interp_sloss = [], [], 0.0
		binput, boutput, btarget, binterp, ibatch = None, None, None, None, 0
		for itime, ctime in enumerate(self.data_timestamps[tset]):
			if (self.time_index < 0) or (itime == self.time_index):
				timeslice: xa.DataArray = self.load_timeslice(ctime)
				tile_iter = TileIterator.get_iterator( ntiles=timeslice.sizes['tiles'] )
				for itile, ctile in enumerate(iter(tile_iter)):
					if self.tile_in_batch(itile, ctile):
						lgm().log(f"     -----------------    evaluate[{tset.name}]: ctime[{itime}]={ctime}, time_index={self.time_index}, ctile[{itile}]={ctile}", display=True)
						batch_data: Optional[xa.DataArray] = self.get_srbatch(ctile, ctime)
						if batch_data is None: break
						binput, boutput, btarget = self.apply_network( batch_data )
						binterp = upsample(binput)
						lgm().log(f"  ->apply_network: inp{ts(binput)} target{ts(btarget)} prd{ts(boutput)} interp{ts(binterp)}")
						[model_sloss, model_multilevel_loss] = self.loss(boutput, btarget)
						batch_model_losses.append( model_sloss )
						[interp_sloss, interp_multilevel_mloss] = self.loss(binterp,btarget)
						batch_interp_losses.append( interp_sloss )
						xyf = batch_data.attrs.get('xyflip', 0)
						sloss = batch_model_losses[-1]
						lgm().log(f" **  ** <{self.model_manager.model_name}:{tset.name}> BATCH[{ibatch:3}] TIME[{itime:3}:{ctime:4}] TILES{list(ctile.values())}[F{xyf}]-> Loss= {sloss:5.1f} ({interp_sloss*1000:5.1f}): {(sloss/interp_sloss)*100:.2f}%", display=True )
						ibatch = ibatch + 1
						if self.tile_index >= 0: break
				if self.time_index >= 0: break

		if binput is not None:  self.input[tset] = binput.detach().cpu().numpy()
		if btarget is not None: self.target[tset] = btarget.detach().cpu().numpy()
		if boutput is not None: self.product[tset] = boutput.detach().cpu().numpy()
		if binterp is not None: self.interp[tset] = binterp.detach().cpu().numpy()

		proc_time = time.time() - proc_start
		lgm().log(f" --- batch_model_losses = {batch_model_losses}")
		lgm().log(f" --- batch_interp_losses = {batch_interp_losses}")
		model_loss: float = np.array(batch_model_losses).mean()
		ntotal_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		if tset == TSet.Validation:
			if (model_loss < self.validation_loss) or (self.validation_loss == 0.0):
				if (self.validation_loss > 0.0) and save_checkpoint:
					interp_loss: float = np.array(batch_interp_losses).mean()
					self.checkpoint_manager.save_checkpoint( epoch, 0, TSet.Validation, model_loss, interp_loss )
				self.validation_loss = model_loss
		lgm().log(f' -------> Exec {tset.value} model with {ntotal_params} wts on {tset.value} tset took {proc_time:.2f} sec, model loss = {model_loss:.4f}')
		losses = dict( model=model_loss, interpolated=np.array(batch_interp_losses).mean() )
		results = dict( input=self.get_ml_input(tset), target=self.get_ml_target(tset), model=self.get_ml_product(tset), interpolated=self.get_ml_interp(tset) )
		return  results, losses

	def apply_network(self, target_data: xa.DataArray ) -> Tuple[Tensor,TensorOrTensors,Tensor]:
		icdim = list(target_data.dims).index('channels')
		input_tensor: Tensor = array2tensor( target_data )
		dsample = cfg().task.get('data_downsample',1.0)
		if dsample > 1.0:
			input_tensor =  downsample( input_tensor, scale_factor=dsample )
		target_channels: List[str] = cfg().task.target_variables
		output_tensor: Tensor = input_tensor
		if target_data.shape[icdim] > len(target_channels):
			tindx: Tensor = array2tensor( np.in1d(target_data.coords['channels'], target_channels).nonzero()[0] )
			output_tensor = torch.index_select(input_tensor, icdim, tindx)
		input_tensor = downsample( input_tensor )
		result_tensor: TensorOrTensors = self.model( input_tensor )
		return input_tensor, result_tensor, output_tensor

