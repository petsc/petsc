#!/usr/bin/env python3
"""
# Created: Mon Jun 20 18:53:35 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from .._typing import *

from ..classes._diag import DiagnosticManager

from ..classes.docs._doc_str     import PetscDocString, SectionManager
from ..classes.docs._doc_section import (
  DefaultSection, FunctionSynopsis, EnumSynopsis, FunctionParameterList, OptionDatabaseKeys, Level,
  Notes, FortranNotes, DeveloperNotes, SourceCode, References, SeeAlso
)

def _do_docstring_check(DocStringType: type[PetscDocStringImpl], linter: Linter, cursor: Cursor) -> None:
  r"""Do the actual docstring checking

  Parameters
  ----------
  DocStringType :
    the type of the docstring to instantiate
  linter :
    the linter instance
  cursor :
    the cursor instance to lint
  """
  docstring = DocStringType(linter, cursor)

  with DiagnosticManager.push_from(docstring.get_pragmas()):
    for section in docstring.parse().sections:
      section.check(linter, cursor, docstring)
  return

class PetscFunctionDocString(PetscDocString):
  sections = SectionManager(
    FunctionSynopsis(),
    FunctionParameterList(),
    OptionDatabaseKeys(),
    Notes(),
    SourceCode(),
    DeveloperNotes(),
    References(),
    FortranNotes(),
    Level(),
    SeeAlso(),
    DefaultSection()
  )

"""Specific 'driver' function to test a particular docstring archetype"""
def check_petsc_function_docstring(linter: Linter, cursor: Cursor) -> None:
  r"""Check a PETSc function docstring

  Parameters
  ----------
  linter :
    the linter to check the docstring with
  cursor :
    the cursor representing the function declaration
  """
  return _do_docstring_check(PetscFunctionDocString, linter, cursor)

class PetscEnumDocString(PetscDocString):
  sections = SectionManager(
    EnumSynopsis(),
    OptionDatabaseKeys(),
    Notes(),
    SourceCode(),
    DeveloperNotes(),
    References(),
    FortranNotes(),
    Level(),
    SeeAlso(),
    DefaultSection()
  )

def check_petsc_enum_docstring(linter: Linter, cursor: Cursor) -> None:
  r"""Check a PETSc enum docstring

  Parameters
  ----------
  linter :
    the linter to check the docstring with
  cursor :
    the cursor representing the enum declaration
  """
  return _do_docstring_check(PetscEnumDocString, linter, cursor)
