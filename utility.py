from pathlib import Path
from typing import Union, Any, AnyStr

import cloudpickle as pickle
import jsonpickle as json


def to_json(
        data: Any,
        **kwargs
):
    return json.encode(data, **kwargs)


def from_json(
        data: str,
        **kwargs
):
    return json.decode(data, **kwargs)


def load_json(
        filepath: Union[AnyStr, Path]
):
    filepath = Path(filepath) if type(filepath) != Path else filepath

    with filepath.open(mode='r') as f:
        data = f.read()
        data = from_json(data=data)

    return data


def save_json(
        filepath: Union[AnyStr, Path],
        data: Any,
        **kwargs
):
    filepath = Path(filepath) if type(filepath) != Path else filepath

    with filepath.open(mode='w') as f:
        data = to_json(data, indent=4, **kwargs)
        f.write(data)


def load_pickle(
        filepath: Union[AnyStr, Path]
) -> Any:
    filepath = Path(filepath) if type(filepath) != Path else filepath
    with filepath.open('rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(
        filepath: Union[AnyStr, Path],
        data: Any
):
    filepath = Path(filepath) if type(filepath) != Path else filepath
    with filepath.open('wb') as f:
        pickle.dump(data, f)
