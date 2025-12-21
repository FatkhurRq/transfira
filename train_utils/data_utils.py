import pathlib


def convert_posixpath_to_string(data):
    """Convert posixpath paths to string for dumping to json."""
    if isinstance(data, dict):
        return {key: convert_posixpath_to_string(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [
            (
                convert_posixpath_to_string(item)
                if isinstance(item, pathlib.Path)
                else item
            )
            for item in data
        ]
    elif isinstance(data, pathlib.Path):
        return str(data)
    else:
        return data
