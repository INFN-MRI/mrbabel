"""User parameters search utils."""

__all__ = ["get_user_param"]


from typing import Any

import mrd

from ._serialization import deserialize_array


def get_user_param(head: mrd.Header, key: str, default: Any = None):
    """
    Search for a given key in mrd.Header UserParameters field.

    If key is not found, returns ``None``.

    Parameters
    ----------
    head : mrd.Header
        Input mrd.Header object.
    key : str
        Parameter to be found.

    Returns
    -------
    Any
        Value corresponding to key; ``None`` if not found.

    """
    if head.user_parameters is not None:
        if head.user_parameters.user_parameter_string:
            for item in head.user_parameters.user_parameter_string:
                if item.name == key:
                    return item.value
        if head.user_parameters.user_parameter_long:
            for item in head.user_parameters.user_parameter_long:
                if item.name == key:
                    return item.value
        if head.user_parameters.user_parameter_double:
            for item in head.user_parameters.user_parameter_double:
                if item.name == key:
                    return item.value
        if head.user_parameters.user_parameter_base64:
            for item in head.user_parameters.user_parameter_base64:
                if item.name == key:
                    return deserialize_array(item.value)
    return default
