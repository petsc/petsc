#!/usr/bin/env python3
"""
# Created: Mon Jun 20 17:45:39 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

from .._typing import *

import enum
import ctypes
import clang.cindex as clx # type: ignore[import]

class CXTranslationUnit(enum.IntFlag):
  """
  clang.cindex.TranslationUnit does not have all latest flags

  see: https://clang.llvm.org/doxygen/group__CINDEX__TRANSLATION__UNIT.html
  """
  NONE                                 = 0x0
  DetailedPreprocessingRecord          = 0x01
  Incomplete                           = 0x02
  PrecompiledPreamble                  = 0x04
  CacheCompletionResults               = 0x08
  ForSerialization                     = 0x10
  SkipFunctionBodies                   = 0x40
  IncludeBriefCommentsInCodeCompletion = 0x80
  CreatePreambleOnFirstParse           = 0x100
  KeepGoing                            = 0x200
  SingleFileParse                      = 0x400
  LimitSkipFunctionBodiesToPreamble    = 0x800
  IncludeAttributedTypes               = 0x1000
  VisitImplicitAttributes              = 0x2000
  IgnoreNonErrorsFromIncludedFiles     = 0x4000
  RetainExcludedConditionalBlocks      = 0x8000

# clang options used for parsing files
base_clang_options = (
  CXTranslationUnit.PrecompiledPreamble |
  #CXTranslationUnit.DetailedPreprocessingRecord |
  CXTranslationUnit.SkipFunctionBodies          |
  CXTranslationUnit.LimitSkipFunctionBodiesToPreamble
)

# clang options for creating the precompiled megaheader
base_pch_clang_options = (
  CXTranslationUnit.CreatePreambleOnFirstParse |
  CXTranslationUnit.Incomplete                 |
  CXTranslationUnit.ForSerialization           |
  CXTranslationUnit.KeepGoing
)

# Cursors that may be attached to function-like usage
clx_func_call_cursor_kinds = {
  clx.CursorKind.FUNCTION_DECL,
  clx.CursorKind.CALL_EXPR
}

# Cursors that may be attached to mathematical operations or types
clx_math_cursor_kinds = {
  clx.CursorKind.INTEGER_LITERAL,
  clx.CursorKind.UNARY_OPERATOR,
  clx.CursorKind.BINARY_OPERATOR
}

# Cursors that contain base literal types
clx_literal_cursor_kinds = {
  clx.CursorKind.INTEGER_LITERAL,
  clx.CursorKind.STRING_LITERAL
}

# Cursors that may be attached to casting
clx_cast_cursor_kinds = {
  clx.CursorKind.CSTYLE_CAST_EXPR,
  clx.CursorKind.CXX_STATIC_CAST_EXPR,
  clx.CursorKind.CXX_DYNAMIC_CAST_EXPR,
  clx.CursorKind.CXX_REINTERPRET_CAST_EXPR,
  clx.CursorKind.CXX_CONST_CAST_EXPR,
  clx.CursorKind.CXX_FUNCTIONAL_CAST_EXPR
}

# Cursors that may be attached when types are converted
clx_conversion_cursor_kinds = clx_cast_cursor_kinds | {clx.CursorKind.UNEXPOSED_EXPR}

clx_var_token_kinds = {clx.TokenKind.IDENTIFIER}

clx_function_type_kinds = {clx.TypeKind.FUNCTIONPROTO, clx.TypeKind.FUNCTIONNOPROTO}

# General Array types, note this doesn't contain the pointer type since that is usually handled
# differently
clx_array_type_kinds = {
  clx.TypeKind.INCOMPLETEARRAY,
  clx.TypeKind.CONSTANTARRAY,
  clx.TypeKind.VARIABLEARRAY
}

clx_pointer_type_kinds = clx_array_type_kinds | {clx.TypeKind.POINTER}

# Specific types
clx_enum_type_kinds   = {clx.TypeKind.ENUM}
# because PetscBool is an enum...
clx_bool_type_kinds   = clx_enum_type_kinds | {clx.TypeKind.BOOL}
clx_char_type_kinds   = {clx.TypeKind.CHAR_S, clx.TypeKind.UCHAR}
clx_mpiint_type_kinds = {clx.TypeKind.INT}
clx_int_type_kinds    = clx_enum_type_kinds | clx_mpiint_type_kinds | {
  clx.TypeKind.USHORT,
  clx.TypeKind.SHORT,
  clx.TypeKind.UINT,
  clx.TypeKind.LONG,
  clx.TypeKind.ULONG,
  clx.TypeKind.LONGLONG,
  clx.TypeKind.ULONGLONG
}

clx_real_type_kinds = {
  clx.TypeKind.FLOAT,
  clx.TypeKind.DOUBLE,
  clx.TypeKind.LONGDOUBLE,
  clx.TypeKind.FLOAT128
}

clx_scalar_type_kinds = clx_real_type_kinds | {clx.TypeKind.COMPLEX}

_T = TypeVar('_T')
_U = TypeVar('_U', covariant=True)

class CTypesCallable(Protocol[_T, _U]):
  # work around a bug in mypy:
  # error: "CTypesCallable[_T, _U]" has no attribute "__name__"
  __name__: str

  @property
  def argtypes(self) -> Sequence[type[_T]]: ...

  def __call__(*args: _T) -> _U: ...

class ClangFunction(Generic[_T, _U]):
  r"""A wrapper to enable safely calling a clang function from python.

  Automatically check the return-type (if it is some kind of int) and raises a RuntimeError if an
  error is detected
  """
  __slots__ = ('_function',)

  _function: CTypesCallable[_T, _U]

  def __init__(self, function: CTypesCallable[_T, _U]) -> None:
    r"""Construct a `ClangFunction`

    Parameters
    ----------
    function : callable
      the underlying clang function
    """
    self._function = function
    return

  def __getattr__(self, attr: str) -> Any:
    return getattr(self._function, attr)

  # Unfortunately, this type-hint does not really do what we want yet... it says that
  # every entry in *args must be of type _T. This is both good and bad:
  #
  # The good news is that if len(*args) == 1, or all arguments are indeed the same type,
  # then this __call__() will be properly type checked.
  #
  # The bad new is if *args does take multiple different argument types, then _T
  # will be deduced to Any in get_clang_function(), and this call will be completely
  # unchecked. At least the type checkers won't through spurious warnings though...
  def __call__(self, *args, check: bool = True) -> _U:
    r"""Invoke the clang function

    Parameters
    ----------
    *args :
      arguments to pass to the clang function
    check : optional
      if the return type is ctype.c_uint, check that it is 0

    Returns
    -------
    ret :
      the return value of the clang function

    Raises
    ------
    ValueError
      if the clang function is called with the wrong number of Arguments
    TypeError
      if the clang function was called with the wrong argument types
    RuntimeError
      if the clang function returned a nonzero exit code
    """
    if len(args) != len(self._function.argtypes):
      mess = f'Trying to call {self._function.__name__}(). Wrong number of arguments for function, expected {len(self._function.argtypes)} got {len(args)}'
      raise ValueError(mess)
    for i, (arg, expected) in enumerate(zip(args, self._function.argtypes)):
      if type(arg) != expected:
        mess = f'Trying to call {self._function.__name__}(). Argument type for argument #{i} does not match. Expected {expected}, got {type(arg)}'
        raise TypeError(mess)
    ret = self._function(*args)
    if check and isinstance(ret, int) and ret != 0:
      raise RuntimeError(f'{self._function.__name__}() returned nonzero exit code {ret}')
    return ret

@overload
def get_clang_function(name: str, arg_types: Sequence[type[_T]]) -> ClangFunction[_T, ctypes.c_uint]:
  ...

@overload
def get_clang_function(name: str, arg_types: Sequence[type[_T]], ret_type: type[_U]) -> ClangFunction[_T, _U]:
  ...

def get_clang_function(name: str, arg_types: Sequence[type[_T]], ret_type: Optional[type[_U]] = None) -> ClangFunction[_T, _U]:
  r"""Get (or register) the clang function RET_TYPE (NAME *)(ARG_TYPES...)

  A useful helper routine to reduce verbiage when retrieving a clang function which may or may not
  already be exposed by clang.cindex

  Parameters
  ----------
  name :
    the name of the clang function
  arg_types :
    the argument types of the clang function
  ret_type : optional
    the return type of the clang function, or ctypes.c_uint if None

  Returns
  -------
  clang_func :
    the callable clang function
  """
  if ret_type is None:
    # cast needed, otherwise
    #
    # error: Incompatible types in assignment (expression has type "Type[c_uint]",
    # variable has type "Optional[Type[_U]]")
    ret_type = TYPE_CAST(type[_U], ctypes.c_uint)

  clxlib = clx.conf.lib
  try:
    func = getattr(clxlib, name)
    if (func.argtypes is None) and (func.errcheck is None):
      # if this hasn't been registered before these will be none
      raise AttributeError
  except AttributeError:
    # have to do the book-keeping ourselves since it may not be properly hooked up
    clx.register_function(clxlib, (name, arg_types, ret_type), False)
    func = getattr(clxlib, name)
  return ClangFunction(TYPE_CAST(CTypesCallable[_T, _U], func))
