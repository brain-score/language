"""Utility functions for logging to stdout and tracking parameters using Weights and Biases"""

################################################################
# stuff for type annotations
################################################################
import typing

################################################################
# stuff for logging to W&B (https://wandb.ai)
################################################################


def init_wandb():

    import wandb

    ...


def log_to_wandb(
    obj: typing.Union["langbrainscore.interface.cacheable._Cacheable", typing.Mapping]
):

    import wandb

    ...


################################################################
# stuff for logging to the terminal
################################################################
import textwrap
from datetime import date
from sys import stderr, stdout
from time import time
import shutil
import os

from colorama import Back, Fore, Style, init
from tqdm import tqdm


def verbose() -> bool:
    """returns True if env variable "VERBOSE" is set to 1"""
    return os.environ.get("VERBOSE", None) == "1"


init(autoreset=True)
_START_TIME = time()


def START_TIME():
    return _START_TIME


def log(message, cmap="INFO", type=None, verbosity_check=False, **kwargs):
    """Utility function to log a `message` to stdout

    Args:
        message (typing.Any): an object that supports `__str__()`
        cmap (str, optional): what  colormap to use. "INFO" corresponds to blue,
            "WARN" Defaults to "INFO".
        type (str, optional): Type of message, for user knowledge. If provided, will be used as the
            tag for this output (e.g. "info").  If no value is provided, the same string as `cmap` is
            used as the tag. Defaults to None.
        verbosity_check (bool, optional): Whether to check for a "VERBOSE" environment flag before
            outputting.  If false, always output text regardless of verbosity setting. Defaults to False.
    """
    if verbosity_check and not verbose():
        return

    _message = str(message)

    class T:
        HEADER = "\033[95m"
        OKBLUE = "\033[94m"
        OKCYAN = "\033[96m"
        OKGREEN = "\033[92m"
        WARNING = "\033[93m"
        FAIL = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"

    if cmap == "INFO":
        c = T.OKBLUE
    elif cmap == "WARN":
        c = T.BOLD + T.WARNING
    elif cmap == "ERR":
        c = "\n" + T.BOLD + T.FAIL
    else:
        c = T.OKBLUE

    timestamp = f"{time() - START_TIME():.2f}s"
    lines = textwrap.wrap(
        _message + T.ENDC,
        width=shutil.get_terminal_size((120, 24))[0] - 1,
        initial_indent=c + "%" * 3 + f" [{type or cmap.lower()} @ {timestamp}] ",
        subsequent_indent=". " * 6 + "",
    )
    tqdm.write("\n".join(lines), file=stderr)
    # print(*lines, sep='\n', file=stderr)
