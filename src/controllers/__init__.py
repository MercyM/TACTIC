REGISTRY = {}

from .basic_controller import BasicMAC
from .lagma_gc_controller import LAGMAMAC_GC
from  .tactic_controller import TACTICMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["lagma_gc_mac"] = LAGMAMAC_GC
REGISTRY["tactic_mac"] = TACTICMAC