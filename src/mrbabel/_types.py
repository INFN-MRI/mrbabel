"""Simple implementation of AttrDict."""

__all__ = ["AttrDict"]

import copy

from types import SimpleNamespace


class AttrDict(SimpleNamespace):
    """Simple implementation of AttrDict."""

    def asdict(self):  # noqa
        return vars(self)

    def asnamespace(self):  # noqa
        return SimpleNamespace(**vars(self))

    @classmethod
    def fromdict(cls, input):  # noqa
        return cls(**input)

    @classmethod
    def fromnamespace(cls, input):  # noqa
        return cls(**vars(input))

    def __repr__(self):  # noqa
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in vars(self).items())})"

    def __len__(self):  # noqa
        return vars(self).__len__()

    def __getitem__(self, key):  # noqa
        return vars(self)[key]

    def __setitem__(self, key, value):  # noqa
        vars(self)[key] = value

    def __delitem__(self, key):  # noqa
        del vars(self)[key]

    def __iter__(self):  # noqa
        return iter(vars(self))

    def clear(self):  # noqa
        return vars(self).clear()

    def copy(self):  # noqa
        return copy.copy(self)

    @classmethod
    def fromkeys(cls, keys, *value):  # noqa
        if len(value) > 1:
            raise TypeError(f"pop expected at most 2 arguments, got {len(value)}")
        if len(value) == 0:
            value = [None]
        return cls(**dict.fromkeys(keys, *value))

    def get(self, key, default=None):  # noqa
        return vars(self).get(key, default)

    def items(self):  # noqa
        return vars(self).items()

    def keys(self):  # noqa
        return vars(self).keys()

    def setdefault(self, key, *default):  # noqa
        if len(default) > 1:
            raise TypeError(f"pop expected at most 2 arguments, got {len(default)}")
        if len(default) == 0:
            default = [None]
        return vars(self).pop(key, *default)

    def popitem(self):  # noqa
        return vars(self).popitem()

    def setdefault(self, key, *default):  # noqa
        if len(default) > 1:
            raise TypeError(
                f"setdefault expected at most 2 arguments, got {len(default)}"
            )
        return vars(self).setdefault(key, *default)

    def update(self, other):  # noqa
        return AttrDict(**vars(self).update(other))

    def values(self):  # noqa
        return vars(self).values()

    def __or__(self, other):  # noqa
        res = vars(self) | vars(other)
        return AttrDict(**res)

    def __ior__(self, other):  # noqa
        _self = vars(self)
        _self |= vars(other)
        return AttrDict(**_self)
