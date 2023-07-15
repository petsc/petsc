#!/usr/bin/env python3
"""
# Created: Mon Jun 20 18:53:35 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from ..classes._diag import DiagnosticManager

from ..classes.docs._doc_str import PetscDocString

"""Specific 'driver' function to test a particular docstring archetype"""
def check_petsc_function_docstring(linter, cursor):
  docstring = PetscDocString(linter, cursor)

  with DiagnosticManager.push_from(docstring.get_pragmas()):
    docstring.parse()
    for section in docstring.sections:
      section.check(linter, cursor, docstring)
  return

def check_petsc_enum_docstring(linter, cursor):
  docstring = PetscDocString(linter, cursor)

  with DiagnosticManager.push_from(docstring.get_pragmas()):
    docstring.parse()
    for section in docstring.sections:
      section.check(linter, cursor, docstring)
  return
