"""Serialization utils."""

__all__ = ["serialize_array", "deserialize_array"]

import numpy as np
import base64
import json

SIGNATURE = "mrbabel_ndarray"


def serialize_array(array: np.ndarray) -> str:
    """
    Serialize NumPy array to Base64.

    Parameters
    ----------
    array : np.ndarray
        Input NumPy array.

    Returns
    -------
    str
        Serialized string.

    """
    array_dict = {
        "signature": SIGNATURE,
        "dtype": str(array.dtype),
        "shape": array.shape,
        "data": base64.b64encode(array.tobytes()).decode("utf-8"),
    }

    return base64.b64encode(json.dumps(array_dict).encode("utf-8")).decode("utf-8")


def deserialize_array(base64_string: str) -> np.ndarray:
    """
    Deserialize NumPy array to Base64.

    Assume the object was serialized using ``mrbabel.utils.serialize_array``.

    If this was not the case, return ``base64.decode``-d object.

    Parameters
    ----------
    base64_string : str
        Input serialized string.

    Returns
    -------
    np.ndarray
        Original NumPy array.

    """
    json_string = base64.b64decode(base64_string)

    try:
        json_string = json_string.decode("utf-8")
        array_dict = json.loads(json_string)

        if array_dict.get("signature", "") != SIGNATURE:
            raise ValueError("Not a NDArray")

        dtype = np.dtype(array_dict["dtype"])
        shape = tuple(array_dict["shape"])
        data = base64.b64decode(array_dict["data"])
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    except Exception:
        try:
            json_string = json_string.decode("utf-8")
            py_dict = json.loads(json_string)
            return py_dict
        except Exception:
            return json_string
