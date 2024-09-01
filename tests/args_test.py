import argparse
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from sres.base.util.config import ConfigContext, cfg, config

cname: str = "sres"
configuration = dict(
	task = "SST-tiles-48",
	dataset = "swot_20-20e",
	pipeline = "sres",
	platform = "explore",
	model = 'rcan-10-20-64'
)
ConfigContext.set_defaults(**configuration)

def get_args(cname) -> argparse.Namespace:
	argparser = argparse.ArgumentParser(description=f'Execute workflow {cname}')
	argparser.add_argument('-r', '--refresh', action='store_true', help="Refresh workflow by deleting existing checkpoints and learning stats")
	argparser.add_argument('-ne', '--nepochs', nargs='?', default=cfg().task.nepochs, type=int, help="Number of epochs to run training")
	return argparser.parse_args()

with ConfigContext( cname ) as cc:
	args = get_args(cname)
	print( "Refresh = " + str(args.refresh))
	print( "NEpochs = " + str(args.nepochs))