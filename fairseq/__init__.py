"""Минимальная заглушка fairseq для Windows окружений RVC.

Реализует ровно тот функционал, который требуется скриптам
Retrieval-based-Voice-Conversion-WebUI при извлечении фичей HuBERT.
"""

from . import checkpoint_utils  # noqa: F401
from . import data  # noqa: F401
from . import modules  # noqa: F401

__all__ = ["checkpoint_utils", "data", "modules"]
