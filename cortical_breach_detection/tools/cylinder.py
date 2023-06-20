#!/usr/bin/env python3
import logging
import math
from collections import UserList
from pathlib import Path
from time import time
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from deepdrr import geo
from .tool import Tool


log = logging.getLogger(__name__)


class Cylinder(Tool):
    """Generic cylinder."""

    radius = 2.5
    height = 150

    # In anatomical coordinates (LPS when viewed in slicer)
    base = geo.point(2.5, 2.5, height)
    tip = geo.point(2.5, 2.5, 0)
