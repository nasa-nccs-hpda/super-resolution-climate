import torch
import torch.nn as nn
import torch.nn.functional as F
from sres.base.util.config import cfg
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from sres.base.util.logging import lgm, exception_handled, log_timing
from model.common.unet import UNet, DoubleConv
from model.common.common import FModule

def get_model( **config ) -> nn.Module:
	return MSCNN(**config)
class MSCNN(FModule):

    def __init__(self, **kwargs):
        super(MSCNN, self).__init__({}, **kwargs)

        self.inc: nn.Module = DoubleConv( self.nchannels_in, self.nfeatures)
        self.downscale: nn.ModuleList = nn.ModuleList()
        self.upsample: nn.ModuleList = nn.ModuleList()
        self.crossscale: nn.ModuleList = nn.ModuleList()
        self.unet: Optional[UNet] = UNet(self.nfeatures, self.nlayers) if self.nlayers > 0 else None
        for iL, usf in enumerate(self.downscale_factors):
            self.downscale.append(ConvDownscale(self.nfeatures, self.nfeatures, usf))
            self.crossscale.append(Crossscale(self.nfeatures, self.nchannels_out))
            self.upsample.append(Upsample(usf, self.ups_mode))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features, results = self.inc(x), [x]
        if self.unet_depth > 0:
            features = self.unet(features)
        for iL, usf in enumerate(self.downscale_factors):
            features = self.downscale[iL](features)
            xave = self.upsample[iL](results[-1])
            xres = self.crossscale[iL](features)
            results.append( torch.add( xres, xave ) )
        return results[1:]

class Upsampler(nn.Module):
    def __init__(self, downscale_factors: List[int], mode: str ):
        print(f"Upsampler: downscale_factors = {downscale_factors}")
        super(Upsampler, self).__init__()
        self.downscale_factors = downscale_factors
        self.upsample: nn.ModuleList = nn.ModuleList()

        for iL, usf in enumerate(self.downscale_factors):
            self.upsample.append( Upsample(usf,mode) )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        result =  x
        for iL, usf in enumerate(self.downscale_factors):
            result = self.upsample[iL](result)
        return result

class ConvDownscale(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, downscale_fator: int):
        super().__init__()
        self.downscale = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=downscale_fator),
            DoubleConv(out_channels, out_channels )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downscale(x)

class Upsample(nn.Module):

    def __init__(self, downscale_factor: int, mode: str):
        super().__init__()
        self.mode = mode
        self.downscale_factor = downscale_factor
        print( f"Creating Upsample stage, downscale_factor={downscale_factor}, mode={mode}")
        self.up = nn.Upsample( scale_factor=downscale_factor, mode=mode )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

class Crossscale(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Crossscale, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
