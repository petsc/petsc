#!/usr/bin/env python3
"""
# Created: Mon Jun 20 18:42:44 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import clang.cindex as clx
import petsclinter  as pl

from ._src_pos import SourceRange, SourceLocation
from ._path    import Path
from .         import _util

from ..util._clang import *

class Cursor:
  """
  A utility wrapper around clang.cindex.Cursor that makes retrieving certain useful properties
  (such as demangled names) from a cursor easier.
  Also provides a host of utility functions that get and (optionally format) the source code
  around a particular cursor. As it is a wrapper any operation done on a clang Cursor may be
  performed directly on a Cursor (although this object does not pass the isinstance() check).

  See __getattr__ below for more info.
  """
  __slots__ = '__cursor', 'name', 'typename', 'derivedtypename', 'argidx', '_cache'

  def __init__(self, cursor, idx=-12345):
    if isinstance(cursor, Cursor):
      self.__cursor        = cursor.clang_cursor()
      self.name            = cursor.name
      self.typename        = cursor.typename
      self.derivedtypename = cursor.derivedtypename
      self.argidx          = cursor.argidx if idx == -12345 else idx
      self._cache          = cursor._cache
    elif isinstance(cursor, clx.Cursor):
      self.__cursor        = cursor
      self._cache          = {}
      self.name            = self.get_name_from_cursor(cursor)
      self.typename        = self.get_typename_from_cursor(cursor)
      self.derivedtypename = self.get_derived_typename_from_cursor(cursor)
      self.argidx          = idx
    else:
      raise ValueError(type(cursor))
    return

  def __getattr__(self, attr):
    """
    Allows us to essentialy fake being a clang cursor, if __getattribute__ fails
    (i.e. the value wasn't found in self), then we try the cursor. So we can do things
    like self.translation_unit, but keep all of our variables out of the cursors
    namespace
    """
    return getattr(self.__cursor, attr)

  def __str__(self):
    return f'{self.get_formatted_location_string()}\n{self.get_formatted_blurb()}'

  def __hash__(self):
    return hash(self.__cursor.hash)

  @classmethod
  def _unhandleable_cursor(cls, cursor):
    # For whatever reason (perhaps because its macro stringization hell) PETSC_HASH_MAP
    # and PetscKernel_XXX absolutely __brick__ the AST. The resultant cursors have no
    # children, no name, no tokens, and a completely incorrect SourceLocation.
    # They are for all intents and purposes uncheckable :)
    srcstr = cls.get_raw_source_from_cursor(cursor)
    errstr = cls.error_view_from_cursor(cursor)
    if 'PETSC_HASH' in srcstr:
      if '_MAP' in srcstr:
        raise pl.ParsingError(f'Encountered unparsable PETSC_HASH_MAP for cursor {errstr}')
      if '_SET' in srcstr:
        raise pl.ParsingError(f'Encountered unparsable PETSC_HASH_SET for cursor {errstr}')
      raise RuntimeError(f'Unhandled unparsable PETSC_HASH_XXX for cursor {errstr}')
    if 'PetscKernel_' in srcstr:
      raise pl.ParsingError(f'Encountered unparsable PetscKernel_XXX for cursor {errstr}')
    if ('PetscOptions' in srcstr) or ('PetscObjectOptions' in srcstr):
      raise pl.ParsingError(f'Encountered unparsable Petsc[Object]OptionsBegin for cursor {errstr}')
    try:
      cursor_view = '\n'.join(_util.view_cursor_full(cursor, max_depth=10))
    except Exception as exc:
      cursor_view = f'ERROR GENERATING CURSOR VIEW\n{str(exc)}'
    raise RuntimeError(
      f'Could not determine useful name for cursor {errstr}\nxxx {"-" * 80} xxx\n{cursor_view}'
    )

  def _get_cached(self, attr, func, *args, **kwargs):
    cache = self._cache
    if attr not in cache:
      cache[attr] = func(*args, **kwargs)
    return cache[attr]

  @classmethod
  def cast(cls, cursor):
    """like numpy.asanyarray but for Cursors"""
    clx_cursor = clx.Cursor
    if not isinstance(cursor, (clx_cursor, cls)):
      raise ValueError(type(cursor))
    return cls(cursor) if isinstance(cursor, clx_cursor) else cursor

  @classmethod
  def error_view_from_cursor(cls, cursor):
    """
    Something has gone wrong, and we try to extract as much information from the cursor as
    possible for the exception. Nothing is guaranteed to be useful here.
    """
    # Does not yet raise exception so we can call it here
    loc_str  = cls.get_formatted_location_string_from_cursor(cursor)
    typename = cls.get_typename_from_cursor(cursor)
    src_str  = cls.get_formatted_source_from_cursor(cursor, nboth=2)
    return f"'{cursor.displayname}' of kind '{cursor.kind}' of type '{typename}' at {loc_str}:\n{src_str}"

  @classmethod
  def get_name_from_cursor(cls, cursor):
    """
    Try to convert **&(PetscObject)obj[i]+73 to obj
    """
    if isinstance(cursor, cls):
      return cursor.name

    name = None
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
        name = operands[0].spelling
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
      name = cls.get_name_from_cursor(castee[0])
    elif (cursor.type.get_canonical().kind == clx.TypeKind.POINTER) or (cursor.kind == clx.CursorKind.UNEXPOSED_EXPR):
      if cursor.type.get_pointee().kind == clx.TypeKind.CHAR_S:
        # For some reason preprocessor macros that contain strings don't propagate
        # their spelling up to the primary cursor, so we need to plumb through
        # the various sub-cursors to find it.
        pointees = [c for c in cursor.walk_preorder() if c.kind in clx_literal_cursor_kinds]
      elif clx.CursorKind.ARRAY_SUBSCRIPT_EXPR in {c.kind for c in cursor.get_children()}:
        # in the form of obj[i], so we try and weed out the iterator variable
        pointees = [
          c for c in cursor.walk_preorder() if c.type.get_canonical().kind in clx_array_type_kinds
        ]
        if not pointees:
          # wasn't a pure array, so we try pointer
          pointees = [c for c in cursor.walk_preorder() if c.type.kind == clx.TypeKind.POINTER]
      else:
        pointees = []
      pointees = list({p.spelling: p for p in pointees}.values())
      if len(pointees) > 1:
        # sometimes array subscripts can creep in
        pointees = [c for c in pointees if c.kind not in clx_math_cursor_kinds]
      if len(pointees) == 1:
        name = cls.get_name_from_cursor(pointees[0])
    elif cursor.kind == clx.CursorKind.ENUM_DECL:
      # have a
      # typedef enum { ... } Foo;
      # so the "name" of the cursor is actually the name of the type itself
      name = cursor.type.get_canonical().spelling

    if not name:
      if cursor.kind  == clx.CursorKind.PARM_DECL:
        # have a parameter declaration, these are allowed to be unnamed!
        return cursor.spelling
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
  def get_raw_name_from_cursor(cls, cursor):
    """
    if get_name_from_cursor tries to convert **&(PetscObject)obj[i]+73 to obj then this function
    tries to extract **&(PetscObject)obj[i]+73 in the cleanest way possible
    """
    cls_instance = isinstance(cursor, cls)
    if cls_instance:
      cache_entry = 'name'
      try:
        return cursor._cache[cache_entry]
      except KeyError:
        pass
    name = ''.join(t.spelling for t in cursor.get_tokens())
    if not name:
      try:
        # now we try for the formatted name
        name = cls.get_name_from_cursor(cursor)
      except pl.ParsingError:
        # noreturn
        cls._unhandleable_cursor(cursor)
    if cls_instance:
      cursor._cache[cache_entry] = name
    return name

  @classmethod
  def get_typename_from_cursor(cls, cursor):
    """
    Try to get the most canonical type from a cursor so DM -> _p_DM *
    """
    if isinstance(cursor, cls):
      ret = cursor.typename
    else:
      canon = cursor.type.get_canonical().spelling
      ret   = canon if canon else cls.get_derived_typename_from_cursor(cursor)
    return ret

  @staticmethod
  def get_derived_typename_from_cursor(cursor):
    """
    Get the least canonical type form a cursor so DM -> DM
    """
    return cursor.type.spelling

  @classmethod
  def has_internal_linkage_from_cursor(cls, cursor):
    def check():
      if cursor.linkage == clx.LinkageKind.INTERNAL:
        # is a static function or variable
        return True, cursor.storage_class.name, cursor.get_definition()

      hidden_visibility = {'hidden', 'protected'}
      for child in cursor.get_children():
        if child.kind == clx.CursorKind.VISIBILITY_ATTR and child.spelling in hidden_visibility:
          # is PETSC_INTERN
          return True, SourceRange(child.extent).raw(tight=True), child
      return False, None, None

    if isinstance(cursor, cls):
      return cursor._get_cached('internal_linkage', check)
    return check()

  def has_internal_linkage(self):
    return self.has_internal_linkage_from_cursor(self)

  @staticmethod
  def get_raw_source_from_cursor(cursor, nbefore=0, nafter=0, nboth=0, **kwargs):
    return _util.get_raw_source_from_cursor(
      cursor, num_before_context=nbefore, num_after_context=nafter, num_context=nboth, **kwargs
    )

  def raw(self, **kwargs):
    return self.get_raw_source_from_cursor(self, **kwargs)

  @classmethod
  def get_formatted_source_from_cursor(cls, cursor, nbefore=0, nafter=0, nboth=0, **kwargs):
    # __extent_final attribute set in getIncludedFileFromCursor() since the translation
    # unit is wrong!
    extent = cursor.extent
    if cursor.kind == clx.CursorKind.FUNCTION_DECL:
      if not isinstance(cursor, cls) or not cursor._cache.get('__extent_final'):
        begin  = extent.start
        # -1 gives you EOL
        fnline = SourceLocation.from_position(cursor.translation_unit, begin.line, -1)
        extent = SourceRange.from_locations(begin, fnline)
    return _util.get_formatted_source_from_source_range(
      extent, num_before_context=nbefore, num_after_context=nafter, num_context=nboth, **kwargs
    )

  def formatted(self, **kwargs):
    return self.get_formatted_source_from_cursor(self, **kwargs)

  def view(self, **kwargs):
    kwargs.setdefault('nboth', 5)
    return pl.sync_print(self.formatted(**kwargs))

  @classmethod
  def get_formatted_location_string_from_cursor(cls, cursor):
    loc = cursor.location
    if isinstance(loc, SourceLocation):
      return str(loc)
    return f'{cls.get_file_from_cursor(cursor)}:{loc.line}:{loc.column}'

  def get_formatted_location_string(self):
    return self.get_formatted_location_string_from_cursor(self)

  @classmethod
  def get_formatted_blurb_from_cursor(cls, cursor, **kwargs):
    kwargs.setdefault('nboth', 2)
    cursor   = cls.cast(cursor)
    aka_mess = '' if cursor.typename == cursor.derivedtypename else f' (a.k.a. \'{cursor.typename}\')'
    return f'\'{cursor.name}\' of type \'{cursor.derivedtypename}\'{aka_mess}\n{cursor.formatted(**kwargs)}\n'

  def get_formatted_blurb(self, **kwargs):
    return self.get_formatted_blurb_from_cursor(self, **kwargs)

  @staticmethod
  def view_ast_from_cursor(cursor):
    return pl.sync_print('\n'.join(_util.view_ast_from_cursor(cursor)))

  def view_ast(self):
    return self.view_ast_from_cursor(self)

  @classmethod
  def find_cursor_references_from_cursor(cls, cursor):
    """
    Brute force find and collect all references in a file that pertain to a particular
    cursor. Essentially refers to finding every reference to the symbol that the cursor
    represents, so this function is only useful for first-class symbols (i.e. variables,
    functions)
    """
    cx_callback, found_cursors = make_cxcursor_and_range_callback(cursor)
    get_clang_function(
      'clang_findReferencesInFile', [clx.Cursor, clx.File, PetscCXCursorAndRangeVisitor]
    )(cursor.clang_cursor(), cls.get_clang_file_from_cursor(cursor), cx_callback)
    return found_cursors

  def find_cursor_references(self):
    return self.find_cursor_references_from_cursor(self)

  @classmethod
  def get_comment_and_range_from_cursor(cls, cursor):
    if isinstance(cursor, cls):
      clang_cursor = cursor.clang_cursor()
    elif isinstance(cursor, clx.Cursor):
      clang_cursor = cursor
    else:
      raise ValueError(type(cursor))
    cursor_range = get_clang_function('clang_Cursor_getCommentRange', [clx.Cursor], clx.SourceRange)(
      clang_cursor
    )
    return cursor.raw_comment, cursor_range

  def get_comment_and_range(self):
    return self.get_comment_and_range_from_cursor(self)

  @classmethod
  def get_clang_file_from_cursor(cls, cursor):
    if isinstance(cursor, cls):
      return cursor._get_cached('file', lambda c: c.location.file, cursor)
    if isinstance(cursor, clx.Cursor):
      return cursor.location.file
    if isinstance(cursor, clx.TranslationUnit):
      return clx.File.from_name(cursor, cursor.spelling)
    raise ValueError(type(cursor))

  @classmethod
  def get_file_from_cursor(cls, cursor):
    return Path(str(cls.get_clang_file_from_cursor(cursor)))

  def get_file(self):
    return self.get_file_from_cursor(self)

  @classmethod
  def is_variadic_function_from_cursor(cls, cursor):
    def check():
      if cursor.kind == clx.CursorKind.FUNCTION_DECL:
        return cursor.displayname.split(',')[-1].replace(')', '').split()[0] == '...'
      return False

    if isinstance(cursor, cls):
      return cursor._get_cached('variadic_function', check)
    return check()

  def is_variadic_function(self):
    return self.is_variadic_function_from_cursor(self)

  @classmethod
  def get_declaration_from_cursor(cls, cursor):
    tu                         = cursor.translation_unit
    cx_callback, found_cursors = make_cxcursor_and_range_callback(cursor)
    get_clang_function(
      'clang_findIncludesInFile', [clx.TranslationUnit, clx.File, PetscCXCursorAndRangeVisitor]
    )(tu, cursor.location.file, cx_callback)
    if not found_cursors:
      # do this the hard way, manually search the entire TU for the declaration
      usr = cursor.get_usr()
      for child in tu.cursor.walk_preorder():
        # don't add ourselves to the list
        if child.get_usr() == usr and child.location != cursor.location:
          child = cls.cast(child)
          child._cache['__extent_final'] = True
          found_cursors.append(child)
    return found_cursors

  def get_declaration(self):
    return self.get_declaration_from_cursor(self)

  def clang_cursor(self):
    """
    return the internal clang cursor
    """
    return self.__cursor
