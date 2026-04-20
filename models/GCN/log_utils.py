"""Colored logging + tqdm-compatible handler shared by training/inference."""
import logging
import sys

from tqdm import tqdm


class _TqdmHandler(logging.StreamHandler):
    """Route log records through tqdm.write so they don't shred active bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


class _ColorFormatter(logging.Formatter):
    GREY = "\x1b[38;5;245m"
    GREEN = "\x1b[32;1m"
    YELLOW = "\x1b[33;1m"
    RED = "\x1b[31;1m"
    BOLD_RED = "\x1b[41;1m"
    CYAN = "\x1b[36;1m"
    BLUE = "\x1b[34;1m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record):
        ts = self.formatTime(record, "%H:%M:%S")
        level = record.levelname
        msg = record.getMessage()
        if not self.use_color:
            return f"[{ts}] {level:<8} {msg}"
        color = self.LEVEL_COLORS.get(record.levelno, "")
        return (f"{self.GREY}[{ts}]{self.RESET} "
                f"{color}{level:<8}{self.RESET} {msg}")


def get_logger(name: str = "gcn", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = _TqdmHandler()
    handler.setFormatter(_ColorFormatter(use_color=sys.stderr.isatty()))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")
_ColorFormatter.LEVEL_COLORS[SUCCESS] = _ColorFormatter.GREEN


def success(self, msg, *args, **kwargs):
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, msg, args, **kwargs)


logging.Logger.success = success
