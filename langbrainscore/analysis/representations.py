"""submodule holding functions related to analyzing outputs from LangBrainScore"""

import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from langbrainscore.interface.encoder import EncoderRepresentations, _Encoder
