#!/usr/bin/env python3
"""
# Created: Mon Jun 20 18:42:44 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import enum
import ctypes
import clang.cindex as clx # type: ignore[import]

from .._typing import *

from ._src_pos    import SourceRange, SourceLocation
from ._attr_cache import AttributeCache
from ._path       import Path
from .            import _util

from .._error import KnownUnhandleableCursorError, ParsingError

from ..util._clang import (
  get_clang_function,
  clx_math_cursor_kinds, clx_cast_cursor_kinds, clx_pointer_type_kinds, clx_literal_cursor_kinds,
  clx_var_token_kinds, clx_function_type_kinds
)

class CtypesEnum(enum.IntEnum):
  """
  A ctypes-compatible IntEnum superclass
  """
  @classmethod
  def from_param(cls, obj: SupportsInt) -> int:
    return int(obj)

@enum.unique
class CXChildVisitResult(CtypesEnum):
  # see
  # https://clang.llvm.org/doxygen/group__CINDEX__CURSOR__TRAVERSAL.html#ga99a9058656e696b622fbefaf5207d715
  # Terminates the cursor traversal.
  Break    = 0
  # Continues the cursor traversal with the next sibling of the cursor just visited,
  # without visiting its children.
  Continue = 1
  # Recursively traverse the children of this cursor, using the same visitor and client
  # data.
  Recurse  = 2

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

def make_cxcursor_and_range_callback(cursor: CursorLike, parsing_error_handler: Optional[Callable[[ParsingError], None]] = None) -> tuple[PetscCXCursorAndRangeVisitor, list[Cursor]]:
  r"""Make a clang cxcursor and range callback functor

  Parameters
  ----------
  cursor : cursor_like
    the cursor to create the callback visitor for
  found_cursors : array_like, optional
    an array or list to append found cursors to, None to create a new list
  parsing_error_handler : callable, optional
    an error handler to handle petsclinter.ParsingError exceptions, which takes the exception object
    as a single parameter

  Returns
  -------
  cx_callback, found_cursors : callable, array_like
    the callback and found_cursors list
  """
  if parsing_error_handler is None:
    parsing_error_handler = lambda exc: None

  found_cursors = []
  def visitor(ctx: Any, raw_clx_cursor: clx.Cursor, src_range: clx.SourceRange) -> CXChildVisitResult:
    # The "raw_clx_cursor" returned here is actually a raw clang 'CXCursor' c-struct, not
    # the a clx.Cursor that we lead python to believe in our function prototype. Luckily
    # we have all we need to remake the python object from scratch
    try:
      clx_cursor = clx.Cursor.from_location(ctx.translation_unit, src_range.start)
      found_cursors.append(Cursor(clx_cursor))
    except ParsingError as pe:
      assert callable(parsing_error_handler)
      parsing_error_handler(pe)
    except Exception:
      import petsclinter as pl
      import traceback

      string = 'Full error full error message below:'
      pl.sync_print('=' * 30, 'CXCursorAndRangeVisitor Error', '=' * 30)
      pl.sync_print("It is possible that this is a false positive! E.g. some 'unexpected number of tokens' errors are due to macro instantiation locations being misattributed.\n", string, '\n', '-' * len(string), '\n', traceback.format_exc(), sep='')
      pl.sync_print('=' * 30, 'CXCursorAndRangeVisitor End Error', '=' * 26)
    return CXChildVisitResult.Continue # continue, recursively

  cx_callback = PetscCXCursorAndRangeVisitor(
    # (PyObject *)cursor;
    ctypes.py_object(cursor),
    # (enum CXVisitorResult(*)(void *, CXCursor, CXSourceRange))visitor;
    CXCursorAndRangeVisitorCallBackProto(visitor)
  )
  return cx_callback, found_cursors

class Cursor(AttributeCache):
  """
  A utility wrapper around clang.cindex.Cursor that makes retrieving certain useful properties
  (such as demangled names) from a cursor easier.
  Also provides a host of utility functions that get and (optionally format) the source code
  around a particular cursor. As it is a wrapper any operation done on a clang Cursor may be
  performed directly on a Cursor (although this object does not pass the isinstance() check).

  See __getattr__ below for more info.
  """
  __slots__ = '__cursor', 'extent', 'name', 'typename', 'derivedtypename', 'argidx'

  __cursor: clx.Cursor
  extent: SourceRange
  name: str
  typename: str
  derivedtypename: str
  argidx: int

  def __init__(self, cursor: CursorLike, idx: int = -12345) -> None:
    r"""Construct a `Cursor`

    Parameters
    ----------
    cursor :
      the cursor to construct this cursor from, can be a `clang.cindex.Cursor` or another `Cursor`
    id : optional
      the index into the parent functions arguments for this cursor, if applicable

    Raises
    ------
    ValueError
      if `cursor` is not a `Cursor` or a `clang.cindex.Cursor`
    """
    if isinstance(cursor, Cursor):
      super().__init__(cursor._cache)
      self.__cursor        = cursor.clang_cursor()
      self.extent          = cursor.extent
      self.name            = cursor.name
      self.typename        = cursor.typename
      self.derivedtypename = cursor.derivedtypename
      self.argidx          = cursor.argidx if idx == -12345 else idx
    elif isinstance(cursor, clx.Cursor):
      super().__init__()
      self.__cursor        = cursor
      self.extent          = SourceRange.cast(cursor.extent)
      self.name            = self.get_name_from_cursor(cursor)
      self.typename        = self.get_typename_from_cursor(cursor)
      self.derivedtypename = self.get_derived_typename_from_cursor(cursor)
      self.argidx          = idx
    else:
      raise ValueError(type(cursor))
    return

  def __getattr__(self, attr: str) -> Any:
    """
    Allows us to essentialy fake being a clang cursor, if __getattribute__ fails
    (i.e. the value wasn't found in self), then we try the cursor. So we can do things
    like self.translation_unit, but keep all of our variables out of the cursors
    namespace
    """
    return getattr(self.__cursor, attr)

  def __str__(self) -> str:
    return f'{self.get_formatted_location_string()}\n{self.get_formatted_blurb()}'

  def __hash__(self) -> int:
    return hash(self.__cursor.hash)

  @classmethod
  def _unhandleable_cursor(cls, cursor: CursorLike) -> NoReturn:
    r"""Given a `cursor`, try to construct as useful an error message as possible from it before
    self destructing

    Parameters
    ----------
    cursor :
      the cursor to construct the message from

    Raises
    ------
    KnownUnhandleableCursorError
      if the cursor is known not to be handleable
    RuntimeError
      this is raised in all other cases

    Notes
    -----
    This function is 'noreturn'
    """
    # For whatever reason (perhaps because its macro stringization hell) PETSC_HASH_MAP
    # and PetscKernel_XXX absolutely __brick__ the AST. The resultant cursors have no
    # children, no name, no tokens, and a completely incorrect SourceLocation.
    # They are for all intents and purposes uncheckable :)
    srcstr = cls.get_raw_source_from_cursor(cursor)
    errstr = cls.error_view_from_cursor(cursor)
    if 'PETSC_HASH' in srcstr:
      if '_MAP' in srcstr:
        raise KnownUnhandleableCursorError(f'Encountered unparsable PETSC_HASH_MAP for cursor {errstr}')
      if '_SET' in srcstr:
        raise KnownUnhandleableCursorError(f'Encountered unparsable PETSC_HASH_SET for cursor {errstr}')
      raise KnownUnhandleableCursorError(f'Unhandled unparsable PETSC_HASH_XXX for cursor {errstr}')
    if 'PetscKernel_' in srcstr:
      raise KnownUnhandleableCursorError(f'Encountered unparsable PetscKernel_XXX for cursor {errstr}')
    if ('PetscOptions' in srcstr) or ('PetscObjectOptions' in srcstr):
      raise KnownUnhandleableCursorError(f'Encountered unparsable Petsc[Object]OptionsBegin for cursor {errstr}')
    try:
      cursor_view = '\n'.join(_util.view_cursor_full(cursor, max_depth=10))
    except Exception as exc:
      cursor_view = f'ERROR GENERATING CURSOR VIEW\n{str(exc)}'
    raise RuntimeError(
      f'Could not determine useful name for cursor {errstr}\nxxx {"-" * 80} xxx\n{cursor_view}'
    )

  @classmethod
  def cast(cls, cursor: CursorLike) -> Cursor:
    r"""like numpy.asanyarray but for `Cursor`s

    Parameters
    ----------
    cursor :
      the cursor object to cast

    Returns
    -------
    cursor :
      either a newly constructed `Cursor` or `cursor` unchanged
    """
    return cursor if isinstance(cursor, cls) else cls(cursor)

  @classmethod
  def error_view_from_cursor(cls, cursor: CursorLike) -> str:
    r"""Get error handling information from a cursor

    Parameters
    ----------
    cursor :
      the cursor to extract information from

    Returns
    -------
    ret :
      a hopefully useful string to pass to an exception

    Notes
    -----
    Something has gone wrong, and we try to extract as much information from the cursor as
    possible for the exception. Nothing is guaranteed to be useful here.
    """
    # Does not yet raise exception so we can call it here
    loc_str  = cls.get_formatted_location_string_from_cursor(cursor)
    typename = cls.get_typename_from_cursor(cursor)
    src_str  = cls.get_formatted_source_from_cursor(cursor, num_context=2)
    return f"'{cursor.displayname}' of kind '{cursor.kind}' of type '{typename}' at {loc_str}:\n{src_str}"

  @classmethod
  def get_name_from_cursor(cls, cursor: CursorLike) -> str:
    r"""Try to convert **&(PetscObject)obj[i]+73 to obj

    Parameters
    ----------
    cursor :
      the cursor

    Returns
    -------
    name :
      the sanitized name of `cursor`
    """
    if isinstance(cursor, cls):
      return cursor.name

    def cls_get_name_from_cursor_safe_call(cursor: CursorLike) -> str:
      try:
        return cls.get_name_from_cursor(cursor)
      except (RuntimeError, ParsingError):
        return ''

    name = ''
    if cursor.spelling:
      name = cursor.spelling
    elif cursor.kind in clx_math_cursor_kinds:
      if cursor.kind == clx.CursorKind.BINARY_OPERATOR:
        # we arbitrarily use the first token here since we assume that it is the important
        # one.
        operands = list(cursor.get_children())
        # its certainly funky when a binary operation doesn't have a binary system of
        # operands
        assert len(operands) == 2, f'Found {len(operands)} operands for binary operator when only expecting 2 for cursor {cls.error_view_from_cursor(cursor)}'
        for name in map(cls_get_name_from_cursor_safe_call, operands):
          if name:
            break
      else:
        # just a plain old number or unary operator
        name = ''.join(t.spelling for t in cursor.get_tokens())
    elif cursor.kind in clx_cast_cursor_kinds:
      # Need to extract the castee from the caster
      castee = [c for c in cursor.get_children() if c.kind == clx.CursorKind.UNEXPOSED_EXPR]
      # If we don't have 1 symbol left then we're in trouble, as we probably didn't
      # pick the right cursors above
      assert len(castee) == 1, f'Cannot determine castee from the caster for cursor {cls.error_view_from_cursor(cursor)}'
      # Easer to do some mild recursion to figure out the naming for us than duplicate
      # the code. Perhaps this should have some sort of recursion check
      name = cls_get_name_from_cursor_safe_call(castee[0])
    elif (cursor.type.get_canonical().kind == clx.TypeKind.POINTER) or (cursor.kind == clx.CursorKind.UNEXPOSED_EXPR):
      if clx.CursorKind.ARRAY_SUBSCRIPT_EXPR in {c.kind for c in cursor.get_children()}:
        # in the form of obj[i], so we try and weed out the iterator variable
        pointees = [
          c for c in cursor.walk_preorder() if c.type.get_canonical().kind in clx_pointer_type_kinds
        ]
      elif cursor.type.get_pointee().kind == clx.TypeKind.CHAR_S:
        # For some reason preprocessor macros that contain strings don't propagate
        # their spelling up to the primary cursor, so we need to plumb through
        # the various sub-cursors to find it.
        pointees = [c for c in cursor.walk_preorder() if c.kind in clx_literal_cursor_kinds]
      else:
        pointees = []
      pointees = list({p.spelling: p for p in pointees}.values())
      if len(pointees) > 1:
        # sometimes array subscripts can creep in
        subscript_operator_kinds = clx_math_cursor_kinds | {clx.CursorKind.ARRAY_SUBSCRIPT_EXPR}
        pointees                 = [c for c in pointees if c.kind not in subscript_operator_kinds]
      if len(pointees) == 1:
        name = cls_get_name_from_cursor_safe_call(pointees[0])
    elif cursor.kind == clx.CursorKind.ENUM_DECL:
      # have a
      # typedef enum { ... } Foo;
      # so the "name" of the cursor is actually the name of the type itself
      name = cursor.type.get_canonical().spelling
    elif cursor.kind == clx.CursorKind.PAREN_EXPR:
      possible_names = {n for n in map(cls_get_name_from_cursor_safe_call, cursor.get_children()) if n}
      try:
        name = possible_names.pop()
      except KeyError:
        # *** KeyError: 'pop from an empty set'
        pass
    elif cursor.kind == clx.CursorKind.COMPOUND_STMT:
      # we have a cursor pointing to a '{'. clang treats these cursors a bit weirdly, they
      # essentially encompass _all_ of the statements between the brackets, but curiously
      # do not
      name = SourceLocation(cursor.extent.start, cursor.translation_unit).raw().strip()

    if not name:
      if cursor.kind  == clx.CursorKind.PARM_DECL:
        # have a parameter declaration, these are allowed to be unnamed!
        return TYPE_CAST(str, cursor.spelling)
      # Catchall last attempt, we become the very thing we swore to destroy and parse the
      # tokens ourselves
      token_list = [t for t in cursor.get_tokens() if t.kind in clx_var_token_kinds]
      # Remove iterator variables
      token_list = [t for t in token_list if t.cursor.kind not in clx_math_cursor_kinds]
      # removes all cursors that have duplicate spelling
      token_list = list({t.spelling: t for t in token_list}.values())
      if len(token_list) != 1:
        cls._unhandleable_cursor(cursor)
      name = token_list[0].spelling
      assert name, f'Cannot determine name of symbol from cursor {cls.error_view_from_cursor(cursor)}'
    return name

  @classmethod
  def get_raw_name_from_cursor(cls, cursor: CursorLike) -> str:
    r"""If get_name_from_cursor tries to convert **&(PetscObject)obj[i]+73 to obj then this function
    tries to extract **&(PetscObject)obj[i]+73 in the cleanest way possible

    Parameters
    ----------
    cursor :
      the cursor

    Returns
    -------
    name :
      the un-sanitized name of `cursor`
    """
    def get_name() -> str:
      name = ''.join(t.spelling for t in cursor.get_tokens())
      if not name:
        try:
          # now we try for the formatted name
          name = cls.get_name_from_cursor(cursor)
        except ParsingError:
          # noreturn
          cls._unhandleable_cursor(cursor)
      return name

    if isinstance(cursor, cls):
      return cursor._get_cached('name', get_name)
    return get_name()

  @classmethod
  def get_typename_from_cursor(cls, cursor: CursorLike) -> str:
    r"""Try to get the most canonical type from a cursor so DM -> _p_DM *

    Parameters
    ----------
    cursor :
      the cursor

    Returns
    -------
    name :
      the canonical type of `cursor`
    """
    if isinstance(cursor, cls):
      return cursor.typename
    canon: str = cursor.type.get_canonical().spelling
    return canon if canon else cls.get_derived_typename_from_cursor(cursor)

  @staticmethod
  def get_derived_typename_from_cursor(cursor: CursorLike) -> str:
    r"""Get the least canonical type form a cursor so DM -> DM

    Parameters
    ----------
    cursor :
      the cursor

    Returns
    -------
    name :
      the least canonical type of `cursor`
    """
    return TYPE_CAST(str, cursor.type.spelling)

  @classmethod
  def has_internal_linkage_from_cursor(cls, cursor: CursorLike) -> tuple[bool, str, Optional[clx.Cursor]]:
    r"""Determine whether `cursor` has internal linkage

    Parameters
    ----------
    cursor :
      the cursor to check

    Returns
    -------
    is_internal :
      True if `cursor` has internal linkage, False otherwise
    internal_attr_src :
      the raw text of the internal linkage attribute
    internal_cursor :
      the cursor corresponding to the internal linkage designation
    """
    def check() -> tuple[bool, str, Optional[clx.Cursor]]:
      if cursor.linkage == clx.LinkageKind.INTERNAL:
        # is a static function or variable
        return True, cursor.storage_class.name, cursor.get_definition()

      hidden_visibility = {'hidden', 'protected'}
      for child in cursor.get_children():
        if child.kind.is_attribute() and child.spelling in hidden_visibility:
          # is PETSC_INTERN
          return True, SourceRange(child.extent).raw(tight=True), child
      return False, '', None

    if isinstance(cursor, cls):
      return cursor._get_cached('internal_linkage', check)
    return check()

  def has_internal_linkage(self) -> tuple[bool, str, Optional[clx.Cursor]]:
    r"""See `Cursor.has_internal_linkage_from_cursor()`"""
    return self.has_internal_linkage_from_cursor(self)

  @staticmethod
  def get_raw_source_from_cursor(cursor: CursorLike, **kwargs) -> str:
    r"""Get the raw source from a `cursor`

    Parameters
    ----------
    cursor :
      the cursor to get from
    **kwargs : dict
      additional keyword arguments to `petsclinter.classes._util.get_raw_source_from_cursor()`

    Returns
    -------
    src :
      the raw source
    """
    return _util.get_raw_source_from_cursor(cursor, **kwargs)

  def raw(self, **kwargs) -> str:
    r"""See `Cursor.get_raw_source_from_cursor()`"""
    return self.get_raw_source_from_cursor(self, **kwargs)

  @classmethod
  def get_formatted_source_from_cursor(cls, cursor: CursorLike, **kwargs) -> str:
    r"""Get the formatted source from a `cursor`

    Parameters
    ----------
    cursor :
      the cursor to get from
    **kwargs : dict
      additional keyword arguments to `petsclinter.classes._util.get_formatted_source_from_cursor()`

    Returns
    -------
    src :
      the formatted source
    """
    # __extent_final attribute set in getIncludedFileFromCursor() since the translation
    # unit is wrong!
    extent = cursor.extent
    if cursor.kind == clx.CursorKind.FUNCTION_DECL:
      if not isinstance(cursor, cls) or not cursor._cache.get('__extent_final'):
        begin  = extent.start
        # -1 gives you EOL
        fnline = SourceLocation.from_position(cursor.translation_unit, begin.line, -1)
        extent = SourceRange.from_locations(begin, fnline)
    return _util.get_formatted_source_from_source_range(extent, **kwargs)

  def formatted(self, **kwargs) -> str:
    r"""See `Cursor.get_formatted_source_from_cursor()`"""
    return self.get_formatted_source_from_cursor(self, **kwargs)

  def view(self, **kwargs) -> None:
    r"""View a `Cursor`

    Parameters
    ----------
    **kwargs :
      keyword arguments to pass to `Cursor.formatted()`
    """
    import petsclinter as pl

    kwargs.setdefault('num_context', 5)
    pl.sync_print(self.formatted(**kwargs))
    return

  @classmethod
  def get_formatted_location_string_from_cursor(cls, cursor: CursorLike) -> str:
    r"""Return the file:func:line for `cursor`

    Parameters
    ----------
    cursor :
      the cursor to get it from

    Returns
    locstr :
      the location string
    """
    loc = cursor.location
    if isinstance(loc, SourceLocation):
      return str(loc)
    return f'{cls.get_file_from_cursor(cursor)}:{loc.line}:{loc.column}'

  def get_formatted_location_string(self) -> str:
    r"""See `Cursor.get_formatted_location_string_from_cursor()`"""
    return self.get_formatted_location_string_from_cursor(self)

  @classmethod
  def get_formatted_blurb_from_cursor(cls, cursor: CursorLike, **kwargs) -> str:
    r"""Get a formatted blurb for `cursor` suitable for viewing

    Parameters
    ----------
    cursor :
      the cursor
    **kwargs :
      additional keyword arguments to pass to `Cursor.formatted()`

    Returns
    -------
    blurb :
      the formatted blurb
    """
    kwargs.setdefault('num_context', 2)
    cursor   = cls.cast(cursor)
    aka_mess = '' if cursor.typename == cursor.derivedtypename else f' (a.k.a. \'{cursor.typename}\')'
    return f'\'{cursor.name}\' of type \'{cursor.derivedtypename}\'{aka_mess}\n{cursor.formatted(**kwargs)}'

  def get_formatted_blurb(self, **kwargs) -> str:
    r"""See `Cursor.get_formatted_blurb_from_cursor()`"""
    return self.get_formatted_blurb_from_cursor(self, **kwargs)

  @staticmethod
  def view_ast_from_cursor(cursor: CursorLike) -> None:
    r"""View the AST for a cursor

    Parameters
    ----------
    cursor :
      the cursor to view

    Notes
    -----
    Shows a lot of useful information, but is unsuitable for showing the user. Essentially a developer
    debug tool
    """
    import petsclinter as pl

    pl.sync_print('\n'.join(_util.view_ast_from_cursor(cursor)))
    return

  def view_ast(self) -> None:
    r"""See `Cursor.view_ast_from_cursor()`"""
    self.view_ast_from_cursor(self)
    return

  @classmethod
  def find_cursor_references_from_cursor(cls, cursor: CursorLike) -> list[Cursor]:
    r"""Brute force find and collect all references in a file that pertain to a particular
    cursor.

    Essentially refers to finding every reference to the symbol that the cursor represents, so
    this function is only useful for first-class symbols (i.e. variables, functions)

    Parameters
    ----------
    cursor :
      the cursor to search for references

    Returns
    -------
    found_cursors :
      a list of references to the cursor in the file
    """
    cx_callback, found_cursors = make_cxcursor_and_range_callback(cursor)
    get_clang_function(
      'clang_findReferencesInFile', [clx.Cursor, clx.File, PetscCXCursorAndRangeVisitor]
    )(cls.get_clang_cursor_from_cursor(cursor), cls.get_clang_file_from_cursor(cursor), cx_callback)
    return found_cursors

  def find_cursor_references(self) -> list[Cursor]:
    r"""See `Cursor.find_cursor_references_from_cursor()`"""
    return self.find_cursor_references_from_cursor(self)

  @classmethod
  def get_comment_and_range_from_cursor(cls, cursor: CursorLike) -> tuple[str, clx.SourceRange]:
    r"""Get the docstring comment and its source range from a cursor

    Parameters
    ----------
    cursor :
      the cursor to get it from

    Returns
    -------
    raw_comment :
      the raw comment text
    cursor_range :
      the source range for the comment
    """
    cursor_range = get_clang_function('clang_Cursor_getCommentRange', [clx.Cursor], clx.SourceRange)(
      cls.get_clang_cursor_from_cursor(cursor)
    )
    raw_comment = cursor.raw_comment
    if raw_comment is None:
      raw_comment = ''
    return raw_comment, cursor_range

  def get_comment_and_range(self) -> tuple[str, clx.SourceRange]:
    r"""See `Cursor.get_comment_and_range_from_cursor()`"""
    return self.get_comment_and_range_from_cursor(self)

  @classmethod
  def get_clang_file_from_cursor(cls, cursor: Union[CursorLike, clx.TranslationUnit]) -> clx.File:
    r"""Get the `clang.cindex.File` from a cursor

    Parameters
    ----------
    cursor :
      the cursor

    Returns
    -------
    clx_file :
      the `clang.cindex.File` object

    Raises
    ------
    ValueError
      if `cursor` is not one of `Cursor`, `clang.cindex.Cursor` or a `clang.cindex.TranslationUnit`

    Notes
    -----
    Instantiating one of these files is for whatever reason stupidly expensive, and does not appear to
    be cached by clang at all, so this function serves to cache that
    """
    if isinstance(cursor, cls):
      return TYPE_CAST(clx.File, cursor._get_cached('file', lambda c: c.location.file, cursor))
    if isinstance(cursor, clx.Cursor):
      return cursor.location.file
    if isinstance(cursor, clx.TranslationUnit):
      return clx.File.from_name(cursor, cursor.spelling)
    raise ValueError(type(cursor))

  @classmethod
  def get_file_from_cursor(cls, cursor: Union[CursorLike, clx.TranslationUnit]) -> Path:
    r"""See `Cursor.get_clang_file_from_cursor()`"""
    return Path(str(cls.get_clang_file_from_cursor(cursor)))

  def get_file(self) -> Path:
    r"""See `Cursor.get_file_from_cursor()`"""
    return self.get_file_from_cursor(self)

  @staticmethod
  def is_variadic_function_from_cursor(cursor: CursorLike) -> bool:
    r"""Answers the question 'is this cursor variadic'?

    Parameters
    ----------
    cursor :
      the cursor

    Returns
    -------
    variadic :
      True if `cursor` is a variadic function, False otherwise
    """
    return TYPE_CAST(bool, cursor.type.is_function_variadic())

  def is_variadic_function(self) -> bool:
    r"""See `Cursor.is_variadic_function_from_cursor()`"""
    return self.is_variadic_function_from_cursor(self)

  @classmethod
  def get_declaration_from_cursor(cls, cursor: CursorLike) -> Cursor:
    r"""Get the declaration cursor for a cursor

    Parameters
    ----------
    cursor :
      the cursor

    Returns
    -------
    decl :
      The original declaration of the cursor

    Notes
    -----
    I don't believe this fully works yet
    """
    if cursor.type.kind in clx_function_type_kinds:
      cursor_file = cls.get_clang_file_from_cursor(cursor)
      canon       = cursor.canonical
      if canon.location.file != cursor_file:
        return Cursor.cast(canon)
      for child in canon.get_children():
        if child.kind.is_attribute() and child.location.file != cursor_file:
          if refs := cls.find_cursor_references_from_cursor(child):
            assert len(refs) == 1, 'Don\'t know how to handle >1 ref!'
            return Cursor.cast(refs[0])
    return TYPE_CAST(Cursor, cursor)

  def get_declaration(self) -> Cursor:
    r"""See `Cursor.get_declaration_from_cursor()`"""
    return self.get_declaration_from_cursor(self)

  @classmethod
  def get_clang_cursor_from_cursor(cls, cursor: CursorLike) -> clx.Cursor:
    r"""Given a cursor, return the underlying clang cursor

    Parameters
    ----------
    cursor :
      the cursor

    Returns
    -------
    clang_cursor :
      the `clang.cindex.Cursor`

    Raises
    ------
    ValueError
      if `cursor` is not a `Cursor` or `clang.cindex.Cursor`
    """
    if isinstance(cursor, cls):
      return cursor.__cursor
    if isinstance(cursor, clx.Cursor):
      return cursor
    raise ValueError(type(cursor))

  def clang_cursor(self) -> clx.Cursor:
    r"""See `Cursor.get_clang_cursor_from_cursor()`"""
    return self.get_clang_cursor_from_cursor(self)
