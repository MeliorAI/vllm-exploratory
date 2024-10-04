from contextlib import contextmanager
from pathlib import Path
from time import perf_counter


def collect_files(file_path: Path | str) -> list[Path]:
    """
    Recursively look through directories to retrieve file paths.

    Args:
        file_path: A path given as a Path or string.

    Returns:
        List of file paths and their types.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    paths: list[Path] = []

    if file_path.is_dir():
        for path in file_path.iterdir():
            paths.extend(collect_files(path))
    else:
        paths.append(file_path)

    return paths


@contextmanager
def catchtime():
    """Prints out the time taken to execute the code."""
    tracker = perf_counter()
    yield
    measured_time = perf_counter() - tracker
    print(f"⏳️ Done in: {measured_time:.3f} seconds")
