#!/usr/bin/env python3
"""
# Created: Tue Nov 29 18:02:22 2022 (-0500)
# @author: Jacob Faibussowitsch
"""
import os
import errno
import signal
import functools

from .._error import BaseError

class TimeoutError(BaseError):
  """
  An error to indicate some operation timed out
  """
  pass

def timeout(seconds=10, error_message=None):
  """
  Decorator to run the decorated function for SECONDS seconds. If the function exceeds this time,
  raises a TimeoutError.
  """
  if error_message is None:
    error_message = os.strerror(errno.ETIME)

  def decorator(func):
    def timeout_handler(signum, frame):
      raise TimeoutError(error_message)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
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
