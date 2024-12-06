"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella

This module contains:

- the definition of the class Logger;
- the implementation of a timer;
- the implementation of a loading animation.

"""
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator
import logging

from rich.console import Console
from rich.logging import RichHandler


class Logger:
    """
    A class of objects that can log information to stdout with the desired
    verbosity.

    """
    def __init__(
        self,
        name: str,
        verbosity: int = 0,
    ):
        """
        Contructor of the class.

        Parameters
        ----------
        name : str
            The name to call the logger with.
        verbosity : int, default=0
            The level of verbosity. 0 sets the logging level to WARNING, 1 to
            INFO and 2 to DEBUG.

        """
        self.name = name
        self.verbosity = verbosity

        loglevel = 30 - 10*verbosity

        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self.logger.setLevel(loglevel)

        self.formatter = logging.Formatter(
            fmt='%(message)s',
            datefmt='[%H:%M:%S]',
        )

        self.handler = RichHandler(markup=True, rich_tracebacks=True)
        self.handler.setFormatter(self.formatter)

        if self.logger.handlers:
            self.logger.handlers.clear()  # avoid accumulating handlers

        self.logger.addHandler(self.handler)

    def get_logger(
        self,
    ):
        """
        Get from the Logger object with a given name, the attributes previously
        used to set the corresponding logger, in order to get the same logger.

        """
        handler = self.handler
        handler.setFormatter(self.formatter)

        self.logger = logging.getLogger(self.name)

        return self


log = Logger("mylog").get_logger()


@contextmanager
def Loading(
    message: str,
) -> Iterator[None]:
    """
    Implement loading animation.

    Parameters
    ----------
    message : str
        The text to print during the animation.

    Returns
    -------
    Iterator[None]

    """
    console = Console()
    try:
        with console.status(f"[bold green]{message}..."):
            yield
    finally:
        console.log(f"[bold green]{message}... Done")


@contextmanager
def Timer(
    description: str,
) -> Iterator[None]:
    """
    Implement timer.

    Parameters
    ----------
    description : str
        The text to print.

    Returns
    -------
    Iterator[None]

    """
    start = datetime.now()
    try:
        yield
    finally:
        end = datetime.now()
        timedelta = end-start
        message = (
            f"{description}, [green]started[/green]: {start},"
            f" [red]ended[/red]: {end}, [cyan]elapsed[/cyan]: {timedelta}"
        )
        log.logger.info(message)
