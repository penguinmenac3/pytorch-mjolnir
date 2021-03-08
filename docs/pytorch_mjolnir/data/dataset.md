[Back to Overview](../../README.md)



# pytorch_mjolnir.data.dataset

> A generic implementation for a dataset based on parsers and file providers.


---
---
## *class* **IIterableDataset**(_IterableDataset)

Interface for an iterable dataset.

Has iter and next.


---
---
## *class* **ISequenceDataset**(_Dataset)

Interface for a sequence dataset.

Has len, getitem, iter and next.


---
---
## *class* **IterableDataset**(_CommonDataset, I**IterableDataset**)

An implementation of the IIterableDataset using fileprovider and parser.

* **file_provider_iterable**: The iterable file provider providing samples to the parser.
* **parser**: The parser converting samples into a usable format.


---
---
## *class* **SequenceDataset**(_CommonDataset, I**SequenceDataset**)

An implementation of the ISequenceDataset using fileprovider and parser.

* **file_provider_sequence**: The sequence file provider providing samples to the parser.
* **parser**: The parser converting samples into a usable format.


