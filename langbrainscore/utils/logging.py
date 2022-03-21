
################################################################
# stuff for logging to W&B (https://wandb.ai) 
################################################################
import wandb


################################################################
# stuff for logging to the terminal 
################################################################
import textwrap
from datetime import date
from sys import stderr, stdout
from time import time

from colorama import Back, Fore, Style, init
from tqdm import tqdm


init(autoreset=True)
_START_TIME = time()


def START_TIME():
    return _START_TIME


def log(message, cmap="INFO", type=None, **kwargs):
    # if kwargs is not None:
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
        c = T.OKGREEN
    elif cmap == "WARN":
        c = T.BOLD + T.WARNING
    elif cmap == "ERR":
        c = "\n" + T.BOLD + T.FAIL
    else:
        c = T.OKBLUE

    timestamp = f"{time() - START_TIME():.2f}s"
    lines = textwrap.wrap(
        _message + T.ENDC,
        width=120,
        initial_indent=c + "%" * 4 + f" [{type or cmap} @ {timestamp}] ",
        subsequent_indent=".  " * 7 + "",
    )
    tqdm.write("\n".join(lines), file=stderr)
    # print(*lines, sep='\n', file=stderr)
