"""Минимальная реализация fairseq.data.dictionary.Dictionary."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


class Dictionary:
    """Заглушка класса ``Dictionary`` совместимая с чекпойнтами HuBERT.

    Чекпойнт HuBERT сериализует экземпляр ``Dictionary`` вместе с моделью.
    Для корректной десериализации достаточно иметь класс с тем же именем.
    Реализация хранит состояние в обычных атрибутах и предоставляет
    минимальный интерфейс, который могут ожидать вызывающие стороны.
    """

    def __init__(
        self,
        symbols: Optional[Iterable[str]] = None,
        counts: Optional[Iterable[int]] = None,
        indices: Optional[Dict[str, int]] = None,
    ) -> None:
        self.symbols: List[str] = list(symbols or [])
        self.count: List[int] = list(counts or [])
        self.indices: Dict[str, int] = dict(indices or {})

        # Часто встречающиеся специальные токены. Значения будут заполнены
        # из state_dict, если они присутствуют. Эти атрибуты нужны, потому что
        # на них могут ссылаться другие части кода fairseq/HuBERT.
        self.pad_word = "<pad>"
        self.eos_word = "</s>"
        self.unk_word = "<unk>"
        self.bos_word = "<s>"

    def __len__(self) -> int:  # pragma: no cover - тривиально
        return len(self.symbols)

    def __getitem__(self, idx: int) -> str:  # pragma: no cover - тривиально
        return self.symbols[idx]

    def __getstate__(self) -> Dict[str, object]:
        """Возвращает сериализуемое состояние."""

        return self.__dict__

    def __setstate__(self, state: Dict[str, object]) -> None:
        """Восстанавливает состояние из сериализованного вида."""

        self.__dict__.update(state)

    # Методы ниже включены из соображений совместимости. Если какой-либо код
    # попытается вызвать их, он получит предсказуемый результат вместо ошибки.

    def add_symbol(self, sym: str, count: int = 1) -> int:  # pragma: no cover
        idx = self.indices.get(sym)
        if idx is None:
            idx = len(self.symbols)
            self.symbols.append(sym)
        if len(self.count) <= idx:
            self.count.extend([0] * (idx + 1 - len(self.count)))
        self.count[idx] += count
        self.indices[sym] = idx
        return idx

    def index(self, sym: str) -> int:  # pragma: no cover - совместимость
        return self.indices[sym]


__all__ = ["Dictionary"]
