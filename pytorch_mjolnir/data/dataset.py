"""doc
# pytorch_mjolnir.data.dataset

> A generic implementation for a dataset based on parsers and file providers.
"""
from typing import Any, Dict, Iterator
from torch.utils.data import IterableDataset as _IterableDataset
from torch.utils.data import Dataset as _Dataset

from .parser import IParser
from .file_provider import FileProviderSequence, FileProviderIterable
from .data_promise import DataPromise


class IIterableDataset(_IterableDataset):
    """
    Interface for an iterable dataset.

    Has iter and next.
    """
    def __next__(self) -> Any:
        raise NotImplementedError("Must be implemented by subclass.")

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError("Must be implemented by subclass.")


class ISequenceDataset(_Dataset):
    """
    Interface for a sequence dataset.

    Has len, getitem, iter and next.
    """
    def __next__(self) -> Any:
        raise NotImplementedError("Must be implemented by subclass.")

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError("Must be implemented by subclass.")

    def __getitem__(self, index) -> Any:
        raise NotImplementedError("Must be implemented by subclass.")

    def __len__(self) -> int:
        raise NotImplementedError("Must be implemented by subclass.")


class _CommonDataset:
    def __init__(self, file_provider_iterable: FileProviderIterable, parser: IParser) -> None:
        super().__init__()
        self._file_provider = file_provider_iterable
        self._fp_iterator = None
        self._parser = parser
        self.transformers = []

    def _process(self, sample: Dict[str, DataPromise]) -> Any:
        sample = self._parser(sample)
        for transformer in self.transformers:
            sample = transformer(sample)
        return sample
    
    def __next__(self) -> Any:
        if self._fp_iterator is None:
            raise RuntimeError("You must first call iter(...) before you can use next(...).")
        sample = self._fp_iterator.__next__()
        return self._process(sample)

    def __iter__(self) -> Iterator[Any]:
        self._fp_iterator = self._file_provider.__iter__()
        return self

    def __len__(self) -> int:
        return len(self._file_provider)


class IterableDataset(_CommonDataset, IIterableDataset):
    def __init__(self, file_provider_iterable: FileProviderIterable, parser: IParser) -> None:
        """
        An implementation of the IIterableDataset using fileprovider and parser.

        :param file_provider_iterable: The iterable file provider providing samples to the parser.
        :param parser: The parser converting samples into a usable format.
        """
        super().__init__(file_provider_iterable, parser)


class SequenceDataset(_CommonDataset, ISequenceDataset):
    def __init__(self, file_provider_sequence: FileProviderSequence, parser: IParser) -> None:
        """
        An implementation of the ISequenceDataset using fileprovider and parser.

        :param file_provider_sequence: The sequence file provider providing samples to the parser.
        :param parser: The parser converting samples into a usable format.
        """
        super().__init__(file_provider_sequence, parser)
        self._file_provider = file_provider_sequence

    def __getitem__(self, index) -> Any:
        sample = self._file_provider[index]
        return self._process(sample)
