#!/usr/bin/env python3
"""
# Created: Mon Jun 20 17:45:39 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import enum
import ctypes
import clang.cindex as clx

class CXTranslationUnit(enum.IntFlag):
  """
  clang.cindex.TranslationUnit does not have all latest flags

  see: https://clang.llvm.org/doxygen/group__CINDEX__TRANSLATION__UNIT.html#gab1e4965c1ebe8e41d71e90203a723fe9
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
  #CXTranslationUnit.PrecompiledPreamble |
  #CXTranslationUnit.DetailedPreprocessingRecord |
  CXTranslationUnit.SkipFunctionBodies          |
  CXTranslationUnit.LimitSkipFunctionBodiesToPreamble
)

# clang options for creating the precompiled megaheader
base_pch_clang_options = (
  #CXTranslationUnit.CreatePreambleOnFirstParse |
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

class CtypesEnum(enum.IntEnum):
  """
  A ctypes-compatible IntEnum superclass
  """
  @classmethod
  def from_param(cls, obj):
    return int(obj)

class CXChildVisitResult(CtypesEnum):
  # see
  # https://clang.llvm.org/doxygen/group__CINDEX__CURSOR__TRAVERSAL.html#ga99a9058656e696b622fbefaf5207d715
  # Terminates the cursor traversal.
  Break    = enum.auto()
  # Continues the cursor traversal with the next sibling of the cursor just visited,
  # without visiting its children.
  Continue = enum.auto()
  # Recursively traverse the children of this cursor, using the same visitor and client
  # data.
  Recurse  = enum.auto()

CXCursorAndRangeVisitorCallBackProto = ctypes.CFUNCTYPE(
  ctypes.c_uint, ctypes.py_object, clx.Cursor, clx.SourceRange
)

class PetscCXCursorAndRangeVisitor(ctypes.Structure):
  # see https://clang.llvm.org/doxygen/structCXCursorAndRangeVisitor.html
  #
  # typedef struct CXCursorAndRangeVisitor {
  #   void *context;
  #   enum CXVisitorResult (*visit)(void *context, CXCursor, CXSourceRange);
  # } CXCursorAndRangeVisitor;
  #
  # Note this is not a  strictly accurate recreation, as this struct expects a
  # (void *) but since C lets anything be a (void *) we can pass in a (PyObject *)
  _fields_ = [
    ('context', ctypes.py_object),
    ('visit',   CXCursorAndRangeVisitorCallBackProto)
  ]

def make_cxcursor_and_range_callback(cursor, found_cursors=None, parsing_error_handler=None):
  import petsclinter as pl
  from ..classes._cursor import Cursor

  if found_cursors is None:
    found_cursors = []

  if parsing_error_handler is None:
    parsing_error_handler = lambda exc: None

  def visitor(ctx, cursor, src_range):
    # The "cursor" returned here is actually just a CXCursor, not the real
    # clx.Cursor that we lead python to believe in our function prototype. Luckily we
    # have all we need to remake the python object from scratch
    cursor = clx.Cursor.from_location(ctx.translation_unit, src_range.start)
    try:
      found_cursors.append(Cursor(cursor))
    except pl.ParsingError as pe:
      parsing_error_handler(pe)
    except Exception:

      import traceback

      string = "Full error full error message below:"
      pl.sync_print('='*30, "CXCursorAndRangeVisitor Error", '='*30)
      pl.sync_print("It is possible that this is a false positive! E.g. some 'unexpected number of tokens' errors are due to macro instantiation locations being misattributed.\n", string, "\n", "-" * len(string), "\n", traceback.format_exc(), sep="")
      pl.sync_print('='*30, "CXCursorAndRangeVisitor End Error", '='*26)
    return CXChildVisitResult.Continue # continue, recursively

  cx_callback = PetscCXCursorAndRangeVisitor(
    # (PyObject *)cursor;
    ctypes.py_object(cursor),
    # (enum CXVisitorResult(*)(void *, CXCursor, CXSourceRange))visitor;
    CXCursorAndRangeVisitorCallBackProto(visitor)
  )
  return cx_callback, found_cursors

class ClangFunction:
  """
  A wrapper to enable safely calling a clang function from python. Automatically check the return-type
  (if it is some kind of int) and raises a RuntimeError if an error is detected
  """
  __slots__ = ('_function',)

  def __init__(self, function):
    self._function = function
    return

  def __getattr__(self, attr):
    return getattr(self._function, attr)

  def __call__(self, *args):
    if len(args) != len(self._function.argtypes):
      mess = f'Trying to call {self._function.__name__}(). Wrong number of arguments for function, expected {len(self._function.argtypes)} got {len(args)}'
      raise RuntimeError(mess)
    for i, (arg, expected) in enumerate(zip(args, self._function.argtypes)):
      if type(arg) != expected:
        mess = f'Trying to call {self._function.__name__}(). Argument type for argument #{i} does not match. Expected {expected}, got {type(arg)}'
        raise RuntimeError(mess)
    ret = self._function(*args)
    if isinstance(ret, int) and ret != 0:
      raise RuntimeError(f'{self._function.__name__}() returned nonzero exit code {ret}')
    return ret

def get_clang_function(name, arg_types, ret_type=None):
  """
  Get (or register) the clang function RET_TYPE (NAME *)(ARG_TYPES...)
  """
  if ret_type is None:
    ret_type = ctypes.c_uint

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
  return ClangFunction(func)
