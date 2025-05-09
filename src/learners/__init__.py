from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .lagma_learner_gc import LAGMAGCLearner
from .tactic_learner import TACTICLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["lagma_gc_learner"] = LAGMAGCLearner
REGISTRY["tactic_learner"] = TACTICLearner