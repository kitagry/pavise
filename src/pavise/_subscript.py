"""Reuse of the classes that DataFrame[Schema] and LazyFrame[Schema] create."""

import functools
import threading
import weakref
from collections.abc import Callable
from typing import Any

ClassGetitem = Callable[[Any, Any], Any]


def once_per_schema(class_getitem: ClassGetitem) -> ClassGetitem:
    """
    Make ``__class_getitem__`` return the same class each time it gets the same schema.

    Without this, ``isinstance(DataFrame[S](df), DataFrame[S])`` is False, since each
    subscript would create another class.
    """
    # Held weakly so that a schema and its class go away once nothing uses the class
    classes: weakref.WeakValueDictionary[Any, Any] = weakref.WeakValueDictionary()
    lock = threading.Lock()

    @functools.wraps(class_getitem)
    def wrapper(cls: Any, schema: Any) -> Any:
        try:
            return classes[schema]
        except KeyError:
            # Two threads subscripting at once must still get one class between them
            with lock:
                typed = classes.get(schema)
                if typed is None:
                    typed = class_getitem(cls, schema)
                    classes[schema] = typed
                return typed
        except TypeError:
            # An unhashable parameter cannot be a key; such a class is not shared
            return class_getitem(cls, schema)

    return wrapper
