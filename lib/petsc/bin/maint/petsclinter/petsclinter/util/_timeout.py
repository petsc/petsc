#!/usr/bin/env python3
"""
# Created: Tue Nov 29 18:02:22 2022 (-0500)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import os
import errno
import signal
import functools

from .._typing import *
from .._error  import TimeoutError

if TYPE_CHECKING:
  from types import FrameType

_P = ParamSpec('_P')
_T = TypeVar('_T')

def timeout(seconds: int = 10, error_message: Optional[str] = None) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
  r"""Decorator to run the decorated function for `seconds` seconds.

  Parameters
  ----------
  seconds : 10, optional
    the number of seconds to wait before timing out
  error_message : optional
    the error message to attach to the timeout

  Returns
  -------
  decorator : callable
    the wrapped function

  Raises
  ------
  TimeoutError
    If the function exceeds `seconds` seconds of execution time
  """
  if error_message is None:
    error_message = os.strerror(errno.ETIME)

  def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
    def timeout_handler(signum: int, frame: Optional[FrameType]) -> NoReturn:
      raise TimeoutError(error_message)

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) ->_T:
      signal.signal(signal.SIGALRM, timeout_handler)
      signal.alarm(seconds)
      result = None
      try:
        result = func(*args, **kwargs)
      finally:
        signal.alarm(0)
      return result

    return wrapper
  return decorator
