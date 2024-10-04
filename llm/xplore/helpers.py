from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any
from rich.console import Console
from rich.table import Table


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


def print_table(title: str, columns: list[dict[str, Any]], rows: list[tuple]):
    # Create a console object to print to
    console = Console()

    # Create a table object
    table = Table(title=title, show_lines=True)

    # Add columns (with optional style and alignment)
    for col in columns:
        table.add_column(col.pop("name"), **col)

    # Add rows
    for row in rows:
        table.add_row(*row)

    # Print the table to the console
    console.print(table)


@contextmanager
def catchtime():
    """Prints out the time taken to execute the code."""
    tracker = perf_counter()
    yield
    measured_time = perf_counter() - tracker
    print(f"⏳️ Done in: {measured_time:.3f} seconds")
