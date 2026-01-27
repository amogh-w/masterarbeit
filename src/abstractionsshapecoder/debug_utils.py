"""debug_utils.py

Simple debug message utilities that print colored INFO, ERROR, and SUCCESS
messages to the console when debugging is enabled.

This module relies on the `termcolor` library to produce colored output.

Constants
---------
DEBUG : bool
    Global flag to enable or disable debug message output.
    Set to `True` to print messages, `False` to silence them.

Functions
---------
debug_info(*args)
    Prints a yellow [INFO] message.
debug_error(*args)
    Prints a red [ERROR] message.
debug_success(*args)
    Prints a green [SUCCESS] message.
"""

DEBUG = True
"""bool: Global switch to enable/disable debug prints. (Default: True)"""

from termcolor import cprint


def debug_info(*args):
    """Print a yellow [INFO] message if debugging is enabled.

    Parameters
    ----------
    *args
        Variable length argument list, passed to `print()`.
        All arguments are converted to strings and joined with spaces.
    """
    if DEBUG:
        cprint(f"[INFO] {' '.join(map(str, args))}", "yellow")


def debug_error(*args):
    """Print a red [ERROR] message if debugging is enabled.

    Parameters
    ----------
    *args
        Variable length argument list, passed to `print()`.
        All arguments are converted to strings and joined with spaces.
    """
    if DEBUG:
        cprint(f"[ERROR] {' '.join(map(str, args))}", "red")


def debug_success(*args):
    """Print a green [SUCCESS] message if debugging is enabled.

    Parameters
    ----------
    *args
        Variable length argument list, passed to `print()`.
        All arguments are converted to strings and joined with spaces.
    """
    if DEBUG:
        cprint(f"[SUCCESS] {' '.join(map(str, args))}", "green")