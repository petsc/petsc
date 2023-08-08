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

class KnownUnhandleableCursorError(ParsingError):
  """
  For whatever reason (perhaps because its macro stringization hell) PETSC_HASH_MAP
  and PetscKernel_XXX absolutely __brick__ the AST. The resultant cursors have no
  children, no name, no tokens, and a completely incorrect SourceLocation. They are for
  all intents and purposes uncheckable :)
  """
  pass

class ClassidNotRegisteredError(BaseError):
  """
  An error to indicate that a particular object has not been registered in the classid map
  """
  pass

class TimeoutError(BaseError):
  r"""An error to indicate some operation timed out"""
  pass

class ClobberTestOutputError(BaseError):
  r"""An error to indicate you are about to clobbber your test code output with patches"""
  pass
