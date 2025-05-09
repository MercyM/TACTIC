REGISTRY = {}

from .rnn_agent import RNNAgent
from .tactic_agent import TACTICAgent
from .tactic_gc_agent import TACTICAgent_GC
from .tactic_gp_agent import TACTICAgent_GP

REGISTRY["rnn"] = RNNAgent
REGISTRY["tactic"] = TACTICAgent
REGISTRY["tactic_gc"] = TACTICAgent_GC
REGISTRY["tactic_gp"] = TACTICAgent_GP