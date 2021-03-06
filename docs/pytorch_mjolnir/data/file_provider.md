[Back to Overview](../../README.md)



# pytorch_mjolnir.data.file_provider

> The interfaces for providing files.


---
---
## *class* **FileProviderIterable**(Iterable)

Provides file promises as an iterator.

The next method returns Dict[str, DataPromise] which is a sample.
Also implements iter and can optionally implement len.

A subclass must implement `__next__`.


---
---
## *class* **FileProviderSequence**(FileProviderIterable, Sequence)

Provides file promises as an sequence.

The getitem and next method return Dict[str, DataPromise] which is a sample.
Also implements iter and len.

A subclass must implement `__getitem__` and `__len__`.


