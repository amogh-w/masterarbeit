"""
debug_utils.py

Simple debug message utilities that print colored INFO, ERROR, and SUCCESS messages when DEBUG is enabled.
"""

DEBUG = True

from termcolor import cprint


def debug_info(*args):
    if DEBUG:
        cprint(f"[INFO] {' '.join(map(str, args))}", "yellow")


def debug_error(*args):
    if DEBUG:
        cprint(f"[ERROR] {' '.join(map(str, args))}", "red")


def debug_success(*args):
    if DEBUG:
        cprint(f"[SUCCESS] {' '.join(map(str, args))}", "green")
