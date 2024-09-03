import matplotlib.pyplot as plt
from sres.view.plot.base import Plot
import ipywidgets as ipw, numpy as np
from sres.base.util.logging import lgm, exception_handled, log_timing
from sres.base.io.loader import ncFormat
from sres.controller.config import TSet
from sres.controller.dual_trainer import ModelTrainer

def subsample( data: np.ndarray, step: int):
	end = step * int(len(data) / step )
	return np.mean(data[:end].reshape(-1, step), 1)

class TrainingPlot(Plot):

	def __init__(self, trainer: ModelTrainer, **kwargs ):
		super(TrainingPlot,self).__init__(trainer, **kwargs)
		self.fmt =[ 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'p', 'h', 'v']
		self.trainer.results_accum.load_results()
		self.min_loss = {}
		self.max_points = kwargs.get("max_points", 200)
		self.create_figure( title='Training Loss' )

	@exception_handled
	def plot(self) -> ipw.Box:
		(x, y) = self.trainer.results_accum.get_plot_data()
		for ip, pid in enumerate(x.keys()):
			xp, yp = x[pid], y[pid]
			npts = xp.size
			if npts > self.max_points:
				step = round(npts/self.max_points)
				xp = subsample(xp,step)
				yp = subsample(yp,step)
			self.min_loss[pid] = yp.min() if (yp.size > 0) else 0.0
			self.axs.plot(xp, yp, self.fmt[ip], label=pid)
			print( f"Plotting {xp.size} {pid} points"  )
		self.axs.set_xlabel("Epoch")
		self.axs.set_ylabel("Loss")
		self.axs.set_yscale(self.yscale)
		interp_loss = self.min_loss.get('ref-valid',   0.0)
		model_loss  = self.min_loss.get('model-valid', 0.0)
		self.axs.set_title(f"Model '{self.model}':  Validation Loss = {model_loss*1000:.1f} ({(model_loss/interp_loss)*100:.1f}%)")
		self.axs.legend()
		self.fig.canvas.draw_idle()
		return ipw.VBox( [self.fig.canvas] )