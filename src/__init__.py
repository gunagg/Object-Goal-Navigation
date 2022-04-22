from typing import Dict
from .policy import (
    Net, Policy,
)
from .policy import (
    Seq2SeqModel
)
from .sem_seg_policy import (
    SemSegSeqModel
)
from .sem_seg_ft_policy import (
    SemSegSeqFTModel
)
POLICY_CLASSES = {
    'Seq2SeqPolicy': Seq2SeqModel,
    'SemSegSeq2SeqPolicy': SemSegSeqModel,
    'SemSegFTSeq2SeqPolicy': SemSegSeqFTModel,
}

__all__ = [
    "Policy", "Net",
]