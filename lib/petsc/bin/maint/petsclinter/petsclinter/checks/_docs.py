#!/usr/bin/env python3
"""
# Created: Mon Jun 20 18:53:35 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import petsclinter as pl

from ..classes.docs._doc_str import PetscDocString

from ..classes._cursor import Cursor

"""Specific 'driver' function to test a particular docstring archetype"""
def check_petsc_function_docstring(linter, cursor):
  cursor = Cursor.cast(cursor)
  try:
    docstring = PetscDocString(linter, cursor).parse()
  except pl.ParsingError as pe:
    return # error already logged with linter

  for section in docstring.sections:
    section.check(linter, cursor, docstring)
  return

def check_petsc_enum_docstring(linter, cursor):
  cursor = Cursor.cast(cursor)
  try:
    docstring = PetscDocString(linter, cursor).parse()
  except pl.ParsingError:
    return # error already logged with linter

  for section in docstring.sections:
    section.check(linter, cursor, docstring)
  return
