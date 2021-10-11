#!/usr/bin/env python3
"""
# Created: Mon Jun 20 19:29:45 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
class BaseError(Exception):
  """
  Base class exception for all exceptions defined in PetscLinter
  """
  pass

class ParsingError(BaseError):
  """
  Mostly to just have a custom "something went wrong when trying to perform a check" to except
  for rather than using a built-in type. These are errors that are meant to be caught and logged
  rather than stopping execution alltogether.

  This should make it so that actual errors aren't hidden.
  """
  pass
