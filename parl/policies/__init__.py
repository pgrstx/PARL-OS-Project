from .lru import LRUPolicy
from .clock import CLOCKPolicy
from .lfu import LFUPolicy
from .arc import ARCPolicy
from .lirs import LIRSPolicy
from .opt import OPTPolicy
from .hybrid import HybridPolicy

# Action ID → Policy class mapping (used by DQN agent)
POLICY_CLASSES = {
    0: LRUPolicy,
    1: CLOCKPolicy,
    2: LFUPolicy,
    3: ARCPolicy,
    4: HybridPolicy,
}

POLICY_NAMES = {
    0: "LRU",
    1: "CLOCK",
    2: "LFU",
    3: "ARC",
    4: "HYBRID",
}

__all__ = [
    "LRUPolicy", "CLOCKPolicy", "LFUPolicy", "ARCPolicy",
    "LIRSPolicy", "OPTPolicy", "HybridPolicy",
    "POLICY_CLASSES", "POLICY_NAMES",
]
