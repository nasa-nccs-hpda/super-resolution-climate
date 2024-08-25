from enum import Enum

class ResultStructure(Enum):
	Tiles = 'tiles'
	Image = 'image'

class TSet(Enum):
	Train = 'train'
	Validation = 'valid'
	Test = 'test'
	Upsample = 'upsample'

class srRes(Enum):
	Low = 'lr'
	High = 'hr'
	Raw = 'raw'

	@classmethod
	def from_config(cls, sval: str ) -> 'srRes':
		if sval == "low": return cls.Low
		if sval == "high": return cls.High
		if sval == "raw": return cls.Raw