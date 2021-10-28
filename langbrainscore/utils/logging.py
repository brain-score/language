
import textwrap
from datetime import date
from time import time
from tqdm import tqdm
from sys import stderr, stdout

from colorama import init, Fore, Back, Style

init(autoreset=True)
_START_TIME = time()
def START_TIME(): return _START_TIME

def log(message, type='INFO', **kwargs):
    # if kwargs is not None:

    class T:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    if type == 'INFO':
        c = T.OKCYAN
    elif type == 'WARN':
        c = T.BOLD + T.WARNING
    elif type == 'ERR':
        c = '\n' + T.BOLD + T.FAIL
    else:
        c = T.OKBLUE

    timestamp = f'{time() - START_TIME():.2f}s'
    lines = textwrap.wrap(message+T.ENDC,
                          width=120,
                          initial_indent=c + '%'*4 + f' [{type} @ {timestamp}] ',
                          subsequent_indent='.'*20+' ')
    tqdm.write('\n'.join(lines), file=stderr)
    # print(*lines, sep='\n', file=stderr)
