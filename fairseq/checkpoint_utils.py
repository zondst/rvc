"""Подмножество API fairseq.checkpoint_utils для HuBERT."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import torch
from torch import Tensor, nn

from .data.dictionary import Dictionary

try:
    import torchaudio  # type: ignore
except Exception as exc:  # pragma: no cover - платформа без torchaudio
    raise ImportError(
        "torchaudio обязателен для встроенной заглушки fairseq. "
        "Установите torchaudio (pip install torchaudio) перед извлечением фичей."
    ) from exc

try:  # pragma: no cover - зависит от версии PyTorch
    from torch.serialization import add_safe_globals
except ImportError:  # PyTorch < 2.1
    add_safe_globals = None  # type: ignore[assignment]

if add_safe_globals is not None:
    add_safe_globals([Dictionary])

_LOGGER = logging.getLogger(__name__)


class _FairseqHubertAdapter(nn.Module):
    """Обёртка, эмулирующая интерфейс fairseq для torchaudio HuBERT."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # В fairseq final_proj применяется только для версии v1; Identity достаточно.
        self.final_proj = nn.Identity()

    def __getattr__(self, name: str):  # pragma: no cover - проброс атрибутов
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _mask_to_lengths(mask: Tensor | None, batch: int, total_length: int) -> Tensor | None:
        if mask is None:
            return torch.full((batch,), total_length, dtype=torch.int64)
        if mask.dim() != 2:
            raise ValueError("padding_mask должен иметь 2 измерения")
        return (~mask.bool()).sum(dim=1, dtype=torch.int64)

    def extract_features(
        self,
        *,
        source: Tensor,
        padding_mask: Tensor | None = None,
        output_layer: int | None = None,
        **_: object,
    ) -> Tuple[Tensor, dict]:
        if source.dtype not in (torch.float32, torch.float64):
            source = source.float()
        lengths = self._mask_to_lengths(padding_mask, source.size(0), source.size(1))
        if lengths is not None:
            lengths = lengths.to(source.device)

        num_layers = int(output_layer) if output_layer else None
        layer_outputs, layer_lengths = self.model.extract_features(
            source,
            lengths=lengths,
            num_layers=num_layers,
        )

        if isinstance(layer_outputs, list) and layer_outputs:
            features = layer_outputs[-1]
        else:
            features = source.new_empty(0)

        extra = {
            "layer_results": tuple({"x": layer} for layer in layer_outputs) if isinstance(layer_outputs, list) else (),
            "padding_mask": padding_mask,
            "layer_lengths": layer_lengths,
        }
        return features, extra

    def forward(self, *args, **kwargs):  # pragma: no cover - совместимость
        return self.model(*args, **kwargs)


def _load_state_dict(path: Path) -> Tuple[dict, object]:
    """Загружает state_dict HuBERT из чекпойнта fairseq."""

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    except pickle.UnpicklingError as exc:
        _LOGGER.warning(
            "HuBERT checkpoint %s нельзя загрузить с weights_only=True (%s). "
            "Падает обратно на weights_only=False. Убедитесь, что чекпойнт получен "
            "из доверенного источника.",
            path,
            exc,
        )
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
    base_model = bundle.get_model()

    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
    if missing:
        _LOGGER.debug("HuBERT state_dict: отсутствуют ключи: %s", sorted(missing))
    if unexpected:
        _LOGGER.debug("HuBERT state_dict: неожиданные ключи: %s", sorted(unexpected))

    normalize = _extract_normalize_flag(cfg)
    saved_cfg = SimpleNamespace(task=SimpleNamespace(normalize=normalize))
    task = SimpleNamespace(cfg=SimpleNamespace(normalize=normalize))

    model = _FairseqHubertAdapter(base_model)

    return [model], saved_cfg, task


__all__ = ["load_model_ensemble_and_task"]
