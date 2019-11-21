from typing import (
    Any,
    Dict,
    Iterable,
    Sequence,
    List,
    Optional,
    TypeVar,
    Union,
    Mapping
)

from pandas import Series, DataFrame

# customized types
TimeSeries = Union[Series, DataFrame]
