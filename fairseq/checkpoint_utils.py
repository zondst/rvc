"""Подмножество API fairseq.checkpoint_utils для HuBERT."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import torch

try:
    import torchaudio  # type: ignore
except Exception as exc:  # pragma: no cover - платформа без torchaudio
    raise ImportError(
        "torchaudio обязателен для встроенной заглушки fairseq. "
        "Установите torchaudio (pip install torchaudio) перед извлечением фичей."
    ) from exc

_LOGGER = logging.getLogger(__name__)


def _load_state_dict(path: Path) -> Tuple[dict, object]:
    """Загружает state_dict HuBERT из чекпойнта fairseq."""

    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        cfg = ckpt.get("cfg") or ckpt.get("args")
    else:
        # На всякий случай: если сохранён чистый state_dict
        state_dict = ckpt
        cfg = None
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Не удалось получить state_dict из {path}")
    return state_dict, cfg


def _extract_normalize_flag(cfg: object) -> bool:
    """Достаёт флаг normalize из сохранённого конфига."""

    if cfg is None:
        return False
    task_cfg = None
    if hasattr(cfg, "task"):
        task_cfg = getattr(cfg, "task")
    elif isinstance(cfg, dict):
        task_cfg = cfg.get("task")
    if task_cfg is None:
        return False
    if isinstance(task_cfg, dict):
        return bool(task_cfg.get("normalize", False))
    return bool(getattr(task_cfg, "normalize", False))


def load_model_ensemble_and_task(
    filenames: Iterable[str],
    suffix: str = "",
    **_: object,
) -> Tuple[List[torch.nn.Module], SimpleNamespace, SimpleNamespace]:
    """Чтение HuBERT checkpoint без зависимостей fairseq.

    Возвращает (models, saved_cfg, task) совместимые с вызовом в WebUI.
    """

    files = list(filenames)
    if not files:
        raise ValueError("Список checkpoint'ов пуст")

    path = Path(files[0])
    if suffix and not path.name.endswith(suffix):
        path = path.with_name(path.name + suffix)
    if not path.exists():
        raise FileNotFoundError(path)

    state_dict, cfg = _load_state_dict(path)

    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _LOGGER.debug("HuBERT state_dict: отсутствуют ключи: %s", sorted(missing))
    if unexpected:
        _LOGGER.debug("HuBERT state_dict: неожиданные ключи: %s", sorted(unexpected))

    normalize = _extract_normalize_flag(cfg)
    saved_cfg = SimpleNamespace(task=SimpleNamespace(normalize=normalize))
    task = SimpleNamespace(cfg=SimpleNamespace(normalize=normalize))

    return [model], saved_cfg, task


__all__ = ["load_model_ensemble_and_task"]
