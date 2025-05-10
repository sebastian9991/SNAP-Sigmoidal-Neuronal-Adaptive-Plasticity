import os


def get_project_root(levels_up: int = 2) -> str:
    """
    Returns the absolute path to the project root directory.

    Parameters:
        levels_up (int): Number of directory levels to go up from the current file.
                         Adjust based on where this function is called from.

    Returns:
        str: Absolute path to the project root.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = current_dir
    for _ in range(levels_up):
        root_path = os.path.dirname(root_path)
    return root_path
