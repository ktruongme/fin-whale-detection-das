"""Custom DAS class that inherits from np.ndarray. This class is used to store
DAS data and associated attributes. The attributes can be accessed in master
attribute 'meta'. For example, if an attribute is named 'attr_name', it can
be accessed using 'instance_name.meta.attr_name'. To see all the attributes,
use method 'instance_name.meta.print()'. The metadata attributes can be updated
using 'instance_name.meta.update(attr_name=new_value)'.
"""

from typing import Any

import numpy as np

from ..filters.filter import DASFilter
from ..loader.loader import DASLoader
from ..plotting.plotting import DASPlotter
from ..filters.resizer import DASResizer
from ..detection.ht_lines import DASHoughLines
from ..detection.yolo import DASYolo
from ..detection.dbscan import DASDbscan
from ..detection.tm import DASTemplate


class DASMeta:
    """Class to handle metadata with dot notation access."""
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def __getattr__(self, item: str) -> Any:
        """Get metadata attributes."""
        # If attribute doesn't exist, raise an AttributeError
        raise AttributeError(f"'DASMeta' object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        """Set metadata attributes."""
        self.__dict__[key] = value

    def print(self) -> None:
        """Print metadata keys."""
        for key, value in self.__dict__.items():
            print(f"{key}: {type(value).__name__}")

    def update(self, **new_meta_attrs: Any) -> None:
        """Update or add new metadata attributes.

        Args:
            new_meta_attrs: Keyword arguments for new or updated metadata
                attributes.
        """
        for key, value in new_meta_attrs.items():
            setattr(self, key, value)


class DASArray(
    np.ndarray,
    DASLoader, DASFilter, DASPlotter,
    DASResizer, DASHoughLines, DASYolo, DASDbscan, DASTemplate
):
    """Custom DAS class that inherits from np.ndarray.

    This class is used to store DAS data and associated attributes.
    """
    def __new__(
        cls,
        input_array: np.ndarray = np.array([]),
        **meta_attrs: Any
    ) -> 'DASArray':
        """Create a new instance of the class.

        Args:
            input_array (np.ndarray): Input array. Defaults to np.array([]).
            **meta_attrs: Keyword arguments for metadata attributes.

        Returns:
            DASArray: New instance of the class.
        """
        # Step 1: Convert input array to ndarray
        obj = np.asarray(input_array).view(cls)

        # Step 2: Assign the metadata dictionary
        obj.meta = DASMeta()
        obj.meta.update(**meta_attrs)

        return obj

    def __array_finalize__(self, obj: np.ndarray) -> None:
        """Finalize the array."""
        if obj is None:
            return

        # Ensure meta is propagated
        self.meta = getattr(obj, 'meta', DASMeta())
