from typing import Tuple, List
import numpy as np 

from .alarm import Alarm

class Device:
    def __init__(self, id:int, alarms:List[Alarm]) -> None:
        self.id = id
        self.alarms = np.array(alarms)