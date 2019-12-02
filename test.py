import generate_data as gd
import importlib
importlib.reload(gd)

import numpy as np
import pandas as pd


x = gd.GenerateDate()
x.continuous_variable()
x.discrete_variable()

