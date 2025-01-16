import sys

import numpy as np
import pytest

sys.path.append("./src")
from signatures import *


def test_heat_signatures():
    model = np.load("./data/voxelModel/HCH_PRT.npy")
    code = heat_signature(model, dim=32)
    assert code.shape[0] == 32
    pass
