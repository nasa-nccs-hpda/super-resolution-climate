from typing import Any, Dict, List, Tuple, Type, Optional, Union
from sres.controller.workflow import WorkflowController
from sres.base.io.loader import ncFormat
from sres.controller.config import TSet

cname = "sres"
model =  'rcan-10-20-64'
varnames = [ "SSS", "SST" ]
regions = [ "", "_southpacific_1200", "_60-20s", "_20-20e", "_20-60n"]
nvars = 2
rid = 3
vid = 1

configuration = dict(
	task = f"swot-{nvars}.{nvars}v",
	dataset = f"swot{regions[rid]}",
	platform = "explore"
)
controller = WorkflowController( cname, configuration )
controller.initialize( cname, model )

controller.get_result_image_view( TSet.Validation, channel=varnames[vid], fsize=8.0)