import logging
import sys
from pathlib import Path
from datetime import datetime, timezone


def setup_logger(
    name: str,
    log_dir: Path | None = None,
    level: int = logging.INFO,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    # stdout handler (always)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # optional file handler
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name}.log"

        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
