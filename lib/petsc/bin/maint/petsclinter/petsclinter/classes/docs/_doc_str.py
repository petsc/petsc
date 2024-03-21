#!/usr/bin/env python3
"""
# Created: Mon Jun 20 18:58:57 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import re
import enum
import difflib
import textwrap
import collections
import clang.cindex as clx # type: ignore[import]

from ..._typing     import *
from ...__version__ import py_version_lt
from ..._error      import BaseError, KnownUnhandleableCursorError

from .. import _util

from .._diag    import DiagnosticManager, Diagnostic
from .._src_pos import SourceRange, SourceLocation
from .._patch   import Patch

from ._doc_section_base import DocBase, SectionBase
from ._doc_section      import (
  DefaultSection, Synopsis, FunctionParameterList, OptionDatabaseKeys, Level, Notes, FortranNotes,
  DeveloperNotes, SourceCode, References, SeeAlso
)

@enum.unique
class Verdict(enum.IntEnum):
  r"""An enum describing whether a particular line is deemed a sowing heading or not."""
  IS_HEADING_BUT_PROBABLY_SHOULDNT_BE = -1
  NOT_HEADING                         = 0
  IS_HEADING                          = 1
  MAYBE_HEADING                       = 2

@enum.unique
class DocStringType(enum.Enum):
  UNKNOWN  = 0
  FUNCTION = enum.auto()
  TYPEDEF  = enum.auto()
  ENUM     = enum.auto()

@enum.unique
class DocStringTypeModifier(enum.Flag):
  NONE     = 0
  MACRO    = enum.auto()
  FLOATING = enum.auto()
  C_FUNC   = enum.auto()

@enum.unique
class MatchReason(enum.IntEnum):
  r"""An enum that describes the reason a header was matched"""
  NOT_FOUND = 0
  NAME      = enum.auto()
  KEYWORD   = enum.auto()
  SUBWORD   = enum.auto()

  def direct_match(self) -> bool:
    return self == MatchReason.NAME

  def __bool__(self) -> bool:
    return self != MatchReason.NOT_FOUND

  def __str__(self) -> str:
    return self.name.casefold()

# expressions that usually end in an unescpaped colon causing the resulting sentence to be
# considered a title
_suspicious_patterns = set(
  map(
    str.casefold,
    (
      r'follows', r'following.*', r'example', r'instance', r'one\sof.*', r'available.*include',
      r'supports.*approaches.*', r'see.*user.*manual', r'y\.\ssaad,\siterative\smethods.*philadelphia',
      r'default', r'in\s.*\scase.*', r'use\sthe.*', r'for\s+example', r'note\s+(also|that)?',
      r'example[,;-]\s', r'.*etc\.', r'references\s+(to|a|so)\s+',
      r'(the|an|the|a)\s+options\s+database\s+(for|to)?'
    )
  )
)
_suspicious_colon_regex = re.compile(r'|'.join(f'{expr}:$' for expr in _suspicious_patterns))
_suspicious_plain_regex = re.compile(r'|'.join(_suspicious_patterns - {'example'}), flags=re.MULTILINE)
del _suspicious_patterns

_pragma_regex = re.compile(r'.*PetscClangLinter\s+pragma\s+(\w+):\s*(.*)')

# Regex to match /* */ patterns
_c_comment_regex = re.compile(r'\/\*(\*(?!\/)|[^*])*\*\/')

_T_co = TypeVar('_T_co', covariant=True)

class SectionNotFoundError(BaseError):
  r"""Exception thrown when a section is searched for, not found, and strict mode was enabled"""
  pass

class GuessHeadingFailError(BaseError):
  r"""Exception thrown then sections fails to guess the appropriate heading for a line"""
  pass

class SectionManager:
  __slots__ = '_verbose', '_sections', '_findcache', '_cachekey'

  _verbose: int
  _sections: dict[str, SectionBase]
  _cachekey: tuple[str, ...]
  _findcache: dict[tuple[str, ...], dict[str, str]]

  def __init__(self, *args: SectionImpl, verbose: int = 0) -> None:
    r"""Construct a `SectionManager` object

    Parameters
    ----------
    *args :
      a set of unique sections to register with the section manager
    verbose : optional
      whether to print verbose output

    Raises
    ------
    ValueError
      if the set of sections to register is not unique
    """
    self._verbose   = verbose
    self._sections  = {section.name : section for section in args}
    self._cachekey  = tuple(self._sections.keys())
    self._findcache = {self._cachekey : {}}
    if len(self._cachekey) != len(args):
      raise ValueError('Have sections with conflicting names!')
    return

  def __getattr__(self, attr: str) -> SectionBase:
    r"""Allows looking up a section via its name, i.e. 'self.fortran_notes'"""
    sections = self._sections
    try:
      return sections[attr]
    except KeyError as ke:
      replaced_attr = attr.replace('_', ' ').casefold()
      try:
        return sections[replaced_attr]
      except KeyError:
        pass
    raise AttributeError(attr)

  def __iter__(self) -> Iterator[SectionBase]:
    yield from self._sections.values()

  def __contains__(self, section: SectionImpl) -> bool:
    return self.registered(section)

  def _print(self, *args, verbosity = 1, **kwargs) -> None:
    r"""Print, but only if verbosity if high enough

    Parameters
    ----------
    *args :
      positional arguments to `petsclinter.sync_print`
    verbosity :
      the minimum verbosity at which to print
    **kwargs :
      keyword arguments to `petsclinter.sync_print`
    """
    if self._verbose >= verbosity:
      import petsclinter as pl

      pl.sync_print(*args, **kwargs)
    return

  def set_verbose(self, verbose: int) -> int:
    r"""Sets verbosity level

    Parameters
    ----------
    verbose :
      the new verbosity level

    Returns
    -------
    verbose :
      the old verbosity level
    """
    old_verbose   = self._verbose
    self._verbose = verbose
    return old_verbose

  def find(self, heading: str, cache_result: bool = True, strict: bool = False) -> SectionBase:
    r"""Given a heading, find the section which best matches it

    Parameters
    ----------
    heading :
      the heading to search for
    cache_result : optional
      should the result of the lookup be cached?
    strict : optional
      is not finding the section considered an error?

    Returns
    -------
    section :
      the section

    Raises
    ------
    SectionNotFoundError
      if `strict` is True and a section could not be matched
    """
    lohead   = heading.casefold()
    sections = self._sections
    cache    = self._findcache[self._cachekey]
    try:
      return sections[cache[lohead]]
    except KeyError:
      pass

    section_names = sections.keys()
    found_reason  = MatchReason.NOT_FOUND
    matched       = self.UNKNOWN_SECTION.name
    try:
      matched = difflib.get_close_matches(heading, section_names, n=1)[0]
    except IndexError:
      pass
    else:
      found_reason = MatchReason.NAME

    if found_reason == MatchReason.NOT_FOUND:
      keywords = [(kw, section.name) for section in self for kw in section.keywords]
      kw_only  = [k for k, _ in keywords]
      try:
        matched = difflib.get_close_matches(heading, kw_only, n=1)[0]
      except IndexError:
        pass
      else:
        found_reason = MatchReason.KEYWORD

    if found_reason == MatchReason.NOT_FOUND:
      # try if we can find a sub-word
      # if heading splits into more than 3 params, then chances are its being mislabeled
      # as a heading anyways
      for head in heading.split(maxsplit=3):
        try:
          # higher cutoff, we have to be pretty sure of a match when using subwords,
          # because it's a lot easier for false positives
          matched = difflib.get_close_matches(head, kw_only, n=1, cutoff=0.8)[0]
        except IndexError:
          continue
        else:
          found_reason = MatchReason.SUBWORD
          break

    max_match_len = max(map(len, section_names))
    if found_reason == MatchReason.NOT_FOUND:
      self._print(
        80 * '*',
        f'UNHANDLED POSSIBLE HEADING! (strict = {strict}, cached = {cache_result})',
        heading,
        80 * '*',
        verbosity=2,
        sep='\n'
      )
      if strict:
        raise SectionNotFoundError(heading)
      # when in doubt, it's probably notes
      self._print(
        '*********** DEFAULTED TO {:{}} FROM {} FOR {}'.format(
          f'{matched} (strict = {strict})', max_match_len, found_reason, heading
        ),
        verbosity=2
      )
    else:
      if not found_reason.direct_match():
        # found via keyword or subword
        matched = next(filter(lambda item: item[0] == matched, keywords))[1]
      self._print(
        f'**** CLOSEST MATCH FOUND {matched:{max_match_len}} FROM {found_reason} FOR {heading}',
        verbosity=2
      )

    if cache_result:
      cache[lohead] = matched
    return sections[matched]

  def registered(self, section: SectionImpl) -> bool:
    r"""Determine whether a section has already been registered with the `SectionManager`

    Parameters
    ----------
    section :
      the section to check for

    Returns
    -------
    reg :
      True if `section` has been registered, False otherwise

    Raises
    ------
    NotImplementedError
      if `section` is not derived from `SectionBase`
    """
    if not isinstance(section, SectionBase):
      raise NotImplementedError(type(section))
    return section.name in self._sections

  def gen_titles(self) -> Generator[str, None, None]:
    r"""Return a generator over all registered titles

    Parameters
    ----------
    get_sections : optional
      retrieve the sections as well

    Returns
    -------
    gen :
      the generator
    """
    return (attr for section in self for attr in section.titles)

  def is_heading(self, line: str, prev_line: str) -> Verdict:
    r"""Determine whether `line` contains a valid heading

    Parameters
    ----------
    line :
      the current line to be checked
    prev_line :
      the previous line

    Returns
    -------
    verdict :
      whether the line is a heading
    """
    def handle_header_with_colon(text: str) -> Verdict:
      if text.endswith(r'\:'):
        return Verdict.NOT_HEADING

      textlo = text.casefold()
      if any(map(textlo.startswith, (t.casefold() + ':' for t in self.gen_titles()))):
        return Verdict.IS_HEADING

      if text.endswith(':'):
        if any(map(text.__contains__, (' - ', '=', '(', ')', '%', '$', '@', '#', '!', '^', '&', '+'))):
          return Verdict.IS_HEADING_BUT_PROBABLY_SHOULDNT_BE

        if _suspicious_colon_regex.search(textlo) is None:
          return Verdict.IS_HEADING
        return Verdict.IS_HEADING_BUT_PROBABLY_SHOULDNT_BE

      try:
        _, _, section = self.fuzzy_find_section(text, cache_result=False, strict=True)
      except GuessHeadingFailError:
        return Verdict.NOT_HEADING
      return Verdict.NOT_HEADING if isinstance(section, DefaultSection) else Verdict.IS_HEADING

    def handle_header_without_colon(line: str, prev_line: str) -> Verdict:
      linelo  = line.casefold()
      results = list(filter(linelo.startswith, map(str.casefold, self.gen_titles())))
      if not results:
        return Verdict.NOT_HEADING
      if _suspicious_plain_regex.search(' '.join((prev_line.casefold(), linelo))):
        # suspicious regex detected, err on the side of caution and say this line is not a
        # heading
        return Verdict.NOT_HEADING
      # not suspicious, still not 100% though
      return Verdict.MAYBE_HEADING

    prev_line = prev_line.strip()
    line      = line.strip()
    if not line or line.startswith(('+', '. ', '-', '$', '.vb', '.ve')):
      return Verdict.NOT_HEADING
    if ':' in line:
      return handle_header_with_colon(line)
    return handle_header_without_colon(line, prev_line)

  def fuzzy_find_section(self, line: str, strict: bool = False, **kwargs) -> tuple[str, str, SectionBase]:
    r"""Try to fuzzy guess what section a heading belongs to.

    Parameters
    ----------
    line :
      the line
    strict : optional
      whether to be strict about matching
    **kwargs :
      additional keywords arguments to `SectionManager.find()`

    Returns
    -------
    attempt :
      the attempt which was successful
    match_title :
      the matched title of the guessed section
    section :
      the matched section

    Raises
    ------
    GuessHeadingFailError
      if header guessing failed

    Notes
    -----
    This needs to be combined with self.find() somehow...
    """
    if strp := line.split(':', maxsplit=1)[0].strip():
      for attempt in (strp, strp.split(maxsplit=1)[0].strip(), strp.title()):
        section = self.find(attempt, **kwargs)
        titles  = section.titles
        if len(titles) > 1:
          titles = tuple(difflib.get_close_matches(attempt, titles, n=1))

        if titles:
          if strict and isinstance(section, DefaultSection):
            break
          return attempt, titles[0], section

    raise GuessHeadingFailError(f'Could not guess heading for:\n{line}')

@DiagnosticManager.register(
  ('internal-linkage','Verify that symbols with internal linkage don\'t have docstrings'),
  ('sowing-chars','Verify that sowing begin and end indicators match the symbol type'),
  ('symbol-spacing','Verify that dosctrings occur immediately above that which they describe'),
  ('indentation','Verify that docstring text is correctly indented'),
  ('section-spacing','Verify that there section headers are separated by at least 1 empty line'),
  ('section-header-maybe-header','Check for lines that seem like they are supposed to be headers'),
  ('section-header-fishy-header','Check for headers that seem like they should not be headers'),
)
class PetscDocString(DocBase):
  """
  Container to encapsulate a sowing docstring and retrieve various objects for it.
  Essentially a Cursor for comments.
  """

  # to pacify type checkers...
  diags: DiagnosticMap

  Type     = DocStringType
  Modifier = DocStringTypeModifier
  sections = SectionManager(
    Synopsis(),
    FunctionParameterList(),
    OptionDatabaseKeys(),
    Notes(),
    SourceCode(),
    DeveloperNotes(),
    References(),
    FortranNotes(),
    Level(),
    SeeAlso(),
    DefaultSection(),
  )
  sowing_types       = {'@', 'S', 'E', 'M'}
  clx_to_sowing_type = {
    clx.TypeKind.FUNCTIONPROTO : ('@', 'functions', Type.FUNCTION),
    clx.TypeKind.ENUM          : ('E', 'enums',     Type.ENUM),
  }
  __slots__ = '_linter', 'cursor', 'raw', 'extent', 'indent', 'type', 'type_mod', '_attr'

  _linter: Linter
  cursor: Cursor
  raw: str
  extent: SourceRange
  indent: int
  type: DocStringType
  type_mod: DocStringTypeModifier
  _attr: dict[str, Any]

  def __init__(self, linter: Linter, cursor: Cursor, indent: int = 2) -> None:
    r"""Construct a `PetscDocString

    Parameters
    ----------
    linter :
      a `Linter` instance
    cursor :
      the cursor to which this docstring belongs
    indent : optional
      the number of line indents for normal lines
    """
    self.sections.set_verbose(linter.verbose)
    self._linter          = linter
    self.cursor           = cursor
    self.raw, self.extent = self._get_sanitized_comment_and_range_from_cursor(self.cursor)
    self.indent           = indent
    self.type             = self.Type.UNKNOWN
    self.type_mod         = self.Modifier.NONE
    self._attr            = self._default_attributes()
    return

  @staticmethod
  def _default_attributes() -> dict[str, Any]:
    return dict()

  @classmethod
  def _is_valid_docstring(cls, cursor: Cursor, raw: str, doc_extent: SourceRange) -> bool:
    r"""Determine whether docstring in `raw` (of `cursor`) is a valid sowing docstring worth checking.

    Parameters
    ----------
    cursor :
      the cursor to which the docstring belongs
    raw :
      the raw text of the docstring
    doc_extent :
      the source range for the docstring itself

    Returns
    -------
    ret :
      True if the docstring is valid, False otherwise
    """
    if not raw or not isinstance(raw, str):
      return False

    # if cursor.extent is *before* the doc_extent we have the following situation:
    #
    # extern PetscErrorCode MatMult_SeqFFTW(Mat, Vec, Vec);
    # ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^ cursor.extent
    # ...
    # /*@                     <
    #   MatMult_SeqFFTW - ... < doc_extent
    # */                      <
    # PetscErrorCode MatMult_SeqFFTW(Mat A, Vec x, Vec y)
    #
    # We can safely "ignore" this cursor and mark it as not a docstring since we will hit
    # the main cursor (i.e. the function definition) later on.
    if cursor.extent < doc_extent:
      return False

    # if we find sowing chars, its probably a docstring
    raw = raw.strip()
    if any(raw.startswith(f'/*{char}') for char in cls.sowing_types):
      return True

    # if we find at least 2 titles, likely this is a docstring, unless it ends in one of
    # the internal suffixes or has internal linkage
    rawlo      = raw.casefold()
    have_title = sum(f'{title}:' in rawlo for title in map(str.casefold, cls.sections.gen_titles()))
    if have_title < 2:
      return False

    # intentionally private symbols don't have docstrings. This is not technically correct
    # since people might still create a docstring for them, but it's very tricky to guess
    # what people meant
    if cursor.name.casefold().endswith(('_private', '_internal')):
      return False

    # likewise symbols with explicitly internal linkage are not considered to have a docstring
    has_internal_linkage, _, _ = cursor.has_internal_linkage()
    return not has_internal_linkage

  @classmethod
  def _get_sanitized_comment_and_range_from_cursor(cls, cursor: Cursor) -> tuple[str, SourceRange]:
    r"""Get the raw docstring text and its source range from a cursor

    Parameters
    ----------
    cursor :
      the cursor

    Returns
    -------
    raw :
      the raw docstring text
    range :
      the source range for `raw`

    Raises
    ------
    KnownUnhandleableCursorError
      if the cursor is not a valid docstring
    """
    raw, clx_extent = cursor.get_comment_and_range()
    extent          = SourceRange.cast(clx_extent, tu=cursor.translation_unit)

    if not cls._is_valid_docstring(cursor, raw, extent):
      raise KnownUnhandleableCursorError('Not a docstring')

    last_match = None
    for re_match in _c_comment_regex.finditer(raw):
      last_match = re_match

    assert last_match is not None
    if start := last_match.start():
      # this handles the following case:
      #
      # /* a dummy comment that is attributed to the symbol */
      # /*
      #   the real docstring comment, note no empty line between this and the previous!
      #   // also handles internal comments
      #   /* of both kinds */
      # */
      # <the symbol>
      assert start > 0
      extent = extent.resized(lbegin=raw.count('\n', 0, start), cbegin=None, cend=None)
      raw    = raw[start:]
    return raw, extent

  def get_pragmas(self) -> dict[str, set[re.Pattern[str]]]:
    r"""Retrieve a dict of pragmas for a particular docstring

    Returns
    -------
    pragmas :
      the pragmas

    Notes
    -----
    `pragmas` is in the form:

    {command_name : set(regex_patterns)}
    """
    def str_remove_prefix(string: str, prefix: str) -> str:
      if py_version_lt(3, 9):
        ret = string[len(prefix):] if string.startswith(prefix) else string
      else:
        # the type checkers do not grok the py_version_lt version guard:
        # error: "str" has no attribute "removeprefix"
        ret = string.removeprefix(prefix) # type: ignore[attr-defined]
      return ret

    start       = self.extent.start
    flag_prefix = DiagnosticManager.flagprefix
    pragmas: collections.defaultdict[str, set[re.Pattern[str]]] = collections.defaultdict(set)
    for line in reversed(_util.read_file_lines_cached(start.file.name, 'r')[:start.line - 1]):
      line = line.rstrip()
      if line.endswith(('}', ';', ')', '>', '"')):
        break
      if re_match := _pragma_regex.match(line):
        pragmas[re_match.group(1)].update(
          map(
            re.compile,
            filter(None, map(str.strip, str_remove_prefix(re_match.group(2), flag_prefix).split(',')))
          )
        )
    return dict(pragmas)

  def make_source_location(self, lineno: int, col: int) -> SourceLocation:
    r"""Make a `SourceLocation`

    Parameters
    ----------
    lineno :
      the line number of the location
    col :
      the column number of the location

    Returns
    -------
    loc :
      the `SourceLocation`

    Notes
    -----
    This is a convenience routine for attaching the docstrings' cursors' translation unit to the
    source location
    """
    return SourceLocation.from_position(self.cursor.translation_unit, lineno, col)

  def make_source_range(self, token: str, string: str, lineno: int, offset: int = 0) -> SourceRange:
    r"""Make a `SourceRange` from a token

    Parameters
    ----------
    token :
      the substring of `string` to make the `SourceRange` for
    string :
      the string to search for `token` in
    lineno :
      the line number of the range
    offset : optional
      the offset into `string` from which to search

    Returns
    -------
    rng :
      the `SourceRange`

    Notes
    -----
    Like `PetscDocString.make_source_location()` this is a convenience routine for properly attaching
    the translation unit to the `SourceRange`. Note though that this is only produces single-lined
    `SourceRange`s.
    """
    col_begin = string.index(token, offset) + 1
    col_end   = col_begin + len(token)
    return SourceRange.from_positions(self.cursor.translation_unit, lineno, col_begin, lineno, col_end)

  def make_diagnostic(self, kind: DiagnosticKind, diag_flag: str, msg: str, src_range: Optional[Union[SourceRange, Cursor]], patch: Optional[Patch] = None, **kwargs) -> Diagnostic:
    r"""Construct a `Diagnostic`

    Parameters
    ----------
    kind :
      the class of `Diagnostic` to create
    diag_flag :
      the command-line flag controlling the diagnostic
    msg :
      the description message for the diagnostic, e.g. the error emssage
    src_range : optional
      the source range to attribute to the diagnostic, if None, the extent for the entire docstring is
      used
    patch : optional
      the patch to fix the diagnostic

    Returns
    -------
    diag :
      the constructed `Diagnotic`
    """
    if src_range is None:
      src_range = self.extent
    else:
      src_range = SourceRange.cast(src_range)
    return Diagnostic.from_source_range(kind, diag_flag, msg, src_range, patch=patch, **kwargs)

  def add_diagnostic_from_source_range(self, kind: DiagnosticKind, diag_flag: str, msg: str, src_range: SourceRangeLike, **kwargs) -> None:
    r"""Log an error from a given source range

    Parameters
    ----------
    kind :
      the kind of `Diagnostic` to add
    diag_flag :
      the diagnostic flag to control the error
    msg :
      the diagnostic message describing the problem in detail
    src_range :
      the `SourceRange` which shows the error in the source
    **kwargs :
      any additional keyword arguments to `PetscDocString.make_diagnostic()`
    """
    return self.add_diagnostic(self.make_diagnostic(kind, diag_flag, msg, src_range, **kwargs))

  def add_diagnostic(self, diagnostic: Diagnostic, cursor: Optional[Cursor] = None) -> None:
    r"""Log an error from a fully-formed diagnostic

    Parameters
    ----------
    diagnostic :
      the diagnostic describing the error
    cursor : optional
      the cursor to attach the error to, if None, the docstrings cursor is used
    """
    return self._linter.add_diagnostic_from_cursor(
      self.cursor if cursor is None else cursor, diagnostic
    )

  def reset(self) -> None:
    r"""Reset any internal state for the `PetscDocString`

    Notes
    -----
    This probably doesn't fully work.
    """
    for section in self.sections:
      section.clear()
    self._attr = self._default_attributes()
    return

  def guess_heading(self, line: str, **kwargs) -> tuple[str, str, SectionBase]:
    r"""A shorthand for `SectionManager.fuzzy_find_section()`"""
    return self.sections.fuzzy_find_section(line, **kwargs)

  def _check_floating(self) -> None:
    r"""Check that the docstring isn't a floating docstring, i.e. for a mansection or particular type

    Raises
    ------
    KnownUnhandleableCursorError
      if the docstring is 'floating', i.e. has 'M' in it
    """
    for line in filter(None, map(str.lstrip, self.raw.splitlines())):
      if not line.startswith(('/*', '//')):
        lsplit = line.split()
        try:
          is_floating = lsplit[0].isupper() and lsplit[1] in {'-', '='}
        except IndexError:
          # the lsplit[1] indexing failed, if it is a macro docstring, it is likely
          # floating
          is_floating = self.Modifier.MACRO in self.type_mod
        if is_floating:
           # don't really know how to handle this for now
          self.type_mod |= self.Modifier.FLOATING
          raise KnownUnhandleableCursorError(
            'DON\'T KNOW HOW TO PROPERLY HANDLE FLOATING DOCSTRINGS'
          )
        break
    return

  def _check_valid_cursor_linkage(self) -> bool:
    r"""Check that a cursor has external linkage, there is no point producing a manpage for function
    that is impossible to call.

    Returns
    -------
    ret :
      True if the cursor has external linkage (and therefore should be checked), False if the cursor
      has internal linkage (and is therefore pointless to check)
    """
    cursor = self.cursor
    # TODO, this should probably also check that the header the cursor is defined in is public
    has_internal_linkage, linked_cursor_name, linkage_cursor = cursor.has_internal_linkage()
    # sometimes a static function has the class description above it, for example
    # VECSEQCUDA sits above a private cuda impls function
    pointless = has_internal_linkage and not (
      cursor.location.file.name.endswith(('.h', '.hpp', '.inc')) or
      self.Modifier.FLOATING in self.type_mod
    )
    if pointless:
      assert linkage_cursor is not None
      begin_sowing_range = self._attr['sowing_char_range']
      linkage_extent     = SourceRange.cast(linkage_cursor.extent)
      diag               = self.make_diagnostic(
        Diagnostic.Kind.ERROR, self.diags.internal_linkage,
        'A sowing docstring for a symbol with internal linkage is pointless', self.extent,
        highlight=False
      ).add_note(
        Diagnostic.make_message_from_formattable(
          f'\'{cursor.displayname}\' is declared \'{linked_cursor_name}\' here', crange=linkage_extent
        ),
        location=linkage_extent.start
      ).add_note(
        'If this docstring is meant as developer-only documentation, remove the sowing chars from the docstring declaration. The linter will then ignore this docstring.'
      ).add_note(
        Diagnostic.make_message_from_formattable(
          'Sowing chars declared here', crange=begin_sowing_range
        ),
        location=begin_sowing_range.start
      )
      self.add_diagnostic(diag)
    return not pointless

  def _check_valid_sowing_chars(self) -> None:
    r"""Check that the sowing prefix and postfix match the expected and are symmetric

    Raises
    ------
    KnownUnhandleableCursorError
      if start of the comment line is invalid
    RuntimeError
      if the start comment contains an unknown sowing char
    """
    sowing_type, lay_type, self.type = self.clx_to_sowing_type[self.cursor.type.kind]
    # check the beginning
    splitlines            = self.raw.splitlines()
    line                  = splitlines[0]
    begin_sowing_range    = self.make_source_range(line, line, self.extent.start.line)
    diag_name             = self.diags.sowing_chars
    possible_sowing_chars = line.split('/*')[1].split()
    try:
      begin_sowing = possible_sowing_chars[0]
    except IndexError:
      begin_sowing = sowing_type
      mess         = f'Invalid comment begin line, does not contain sowing identifier. Expected \'/*{sowing_type}\' for {lay_type}'
      self.add_diagnostic_from_source_range(Diagnostic.Kind.ERROR, diag_name, mess, begin_sowing_range)
    else:
      assert isinstance(begin_sowing, str), f'begin_sowing is not a string: {begin_sowing}'
      if begin_sowing[0] not in self.sowing_types:
        diagnosed = False
        if line[line.find(begin_sowing) - 1].isspace():
          # There is a space between the "sowing char" and the character before
          # it. Therefore it is likely just regular text. Sometimes people make internal
          # sowing-like docstrings just to keep things consistent, for example:
          #
          #        v--- identified as begin_sowing
          # /*     KSPSolve_LCD - This routine actually applies the left conjugate
          # ...
          #
          # we should ignore it, and stop processing this docstring altogether since it is
          # not an actual docstring.
          raise KnownUnhandleableCursorError
        if begin_sowing[0] == 'C':
          # sometimes people mix up the order, or forget to add the right letter for the
          # type, for example:
          #
          #   v--- begin_sowing, should be @C
          # /*C
          #   MatElimininateZeroes
          #
          if len(begin_sowing) == 1:
            # they forgot the correct identifier
            sub_mess  = f'It appears you forgot to prepend \'{sowing_type}\''
            expected  = f'{sowing_type}{begin_sowing}'
            diagnosed = True
            # making a new source range instead of using begin_sowing_range is
            # deliberate. The line may still contain other garbage, i.e.:
            #
            # /*C FooBarBaz - asdasdasdasd
            #   ^~~~~~~~~~~~~~~~~~~~~~~~~^ begin_sowing_range
            #
            # which we do not want to overwrite with 'expected'. In order for the patch to
            # be maximally stable we also don't want to have the replacement contain the
            # (possibly) trailing stuff, so we make our new range just encompass 'C'.
            patch = Patch(
              self.make_source_range(begin_sowing, line, begin_sowing_range.start.line), expected
            )
          elif any(c in self.sowing_types for c in begin_sowing):
            # wrong order
            sub_mess  = 'Did you put it in the wrong order'
            expected  = f'{sowing_type}{begin_sowing.replace(sowing_type, "")}'
            diagnosed = True
            patch     = None
          if diagnosed:
            self.add_diagnostic_from_source_range(
              Diagnostic.Kind.ERROR, diag_name,
              f'Invalid docstring identifier, contains unexpected char sequence \'{begin_sowing}\', expected \'/*{expected}\'. {sub_mess}?',
              begin_sowing_range,
              patch=patch
            )
        if not diagnosed:
          raise RuntimeError(f'Unknown sowing char {begin_sowing[0]} not in sowing types {self.sowing_types} found in {line}')
      begin_sowing_range = self.make_source_range(begin_sowing, line, begin_sowing_range.start.line)

    self._attr['sowing_char_range'] = begin_sowing_range

    if 'M' in begin_sowing:
      self.type_mod |= self.Modifier.MACRO
    if 'C' in begin_sowing:
      self.type_mod |= self.Modifier.C_FUNC

    # check that nothing else is on the comment begin line
    lsplit = line.strip().split(maxsplit=1)
    if len(lsplit) != 1:
      rest    = lsplit[1]
      restloc = self.make_source_range(rest, line, self.extent.start.line)
      mess    = 'Invalid comment begin line, must only contain \'/*\' and sowing identifier'
      self.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, diag_name, mess, restloc,
        patch=Patch(restloc, '\n' + (' '*self.indent) + rest)
      )
    return

  def _check_valid_docstring_spacing(self) -> None:
    r"""Check that the docstring itself is flush against the thing it describes.

    Notes
    -----
    Checks that the docstring looks like
    ```
    /*
      PetscFooBar - ...
    */
    PetscErrorCode PetscFooBar(...)
    ```
    not
    ```
    /*
      PetscFooBar - ...
    */

    PetscErrorCode PetscFooBar(...)
    ```
    """
    if self.Modifier.FLOATING in self.type_mod:
      return # floating docstring sections need not be checked for this

    end_line     = self.extent.end.line + 1
    cursor_start = self.cursor.extent.start
    if end_line != cursor_start.line:
      # there is at least 1 (probably empty) line between the comment end and whatever it
      # is describing
      diag = self.diags.symbol_spacing
      mess = 'Invalid line-spacing between docstring and the symbol it describes. The docstring must appear immediately above its target'
      eloc = self.make_source_range('', '', end_line)
      floc = SourceRange.from_locations(self.make_source_location(end_line, 1), cursor_start)
      self.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, diag, mess, eloc, highlight=False, patch=Patch(floc, '')
      )
    return

  def _check_valid_indentation(self, lineno: int, line: str, left_stripped: str) -> None:
    r"""If the line is regular (not empty, or a parameter list), check that line is indented correctly

    Parameters
    ----------
    lineno :
      the line number of the line
    line :
      the line itself
    left_stripped :
      the line that has been left-stripped
    """
    if linelen := len(line):
      indent       = linelen - len(left_stripped)
      expected_ind = 0 if line.startswith(('.', '+', '-', '$')) else self.indent
      if indent != expected_ind:
        diag = self.diags.indentation
        loc  = self.make_source_range(' ' * indent, line, lineno)
        mess = f'Invalid indentation ({indent}), all regular (non-empty, non-parameter, non-seealso) text must be indented to {self.indent} columns'
        self.add_diagnostic_from_source_range(
          Diagnostic.Kind.ERROR, diag, mess, loc, patch=Patch(loc, ' ' * expected_ind)
        )
    return

  def _check_valid_section_spacing(self, prevline: str, lineno: int) -> None:
    r"""Check that sections have at least 1 empty line between them

    Parameters
    ----------
    prevline :
      the previous line
    lineno :
      the current line number

    Notes
    -----
    Checks that sections are formatted like
    ```
    Notes:
    asdadsadasdads

    Example Usage:
    asdasdasd
    ```
    not
    ```
    Notes:
    asdasdasd
    Example Usage:
    asdadasd
    ```
    """
    if prevline and not prevline.isspace():
      loc = self.make_source_range('', '', lineno)
      self.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, self.diags.section_spacing,
        'Missing empty line between sections, must have one before this section',
        loc, highlight=False, patch=Patch(loc, '\n')
      )
    return

  def _check_section_header_typo(self, verdict: Verdict, line: str, lineno: int) -> Verdict:
    r"""Check that a section header that looks like a section header is actually one

    Parameters
    ----------
    verdict :
      the current header verdict of the line
    line :
      the line
    lineno :
      the line number

    Returns
    -------
    verdict :
      the new verdict (if changed)
    """
    if verdict == Verdict.MAYBE_HEADING:
      try:
        name, match_title, _ = self.guess_heading(line, strict=True)
      except GuessHeadingFailError as ghfe:
        # Not being able to guess the heading here is OK since we only *think* it's a
        # heading
        self.sections._print(ghfe)
        return Verdict.NOT_HEADING
      if ':' in line:
        mess = f'Line seems to be a section header but doesn\'t directly end with \':\', did you mean \'{match_title}\'?'
      else:
        mess = f'Line seems to be a section header but missing \':\', did you mean \'{match_title}:\'?'
      self.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, self.diags.section_header_maybe_header, mess,
        self.make_source_range(name, line, lineno)
      )
    return verdict

  def _check_section_header_that_probably_should_not_be_one(self, verdict: Verdict, line: str, stripped: str, lineno: int) -> Verdict:
    r"""Check that a section header that ends with ':' is not really a header

    Parameters
    ----------
    verdict :
      the current heading verdict
    line :
      the line
    stripped :
      `line` but stripped
    lineno :
      the line number

    Returns
    -------
    verdict :
      the update verdict
    """
    if verdict < 0:
      try:
        _, _, section_guess = self.guess_heading(line, cache_result=False)
      except GuessHeadingFailError as ghfe:
        # Not being able to guess the heading here is OK since we aren't sure this isn't a
        # heading after all
        self.sections._print(ghfe)
        verdict = Verdict.NOT_HEADING
      else:
        assert isinstance(section_guess, SectionBase)
        if isinstance(section_guess, DefaultSection):
          # we could not find a suitable section for it
          assert not line.endswith(r'\:')
          eloc = self.make_source_range(':', line, lineno, offset=line.rfind(':'))
          mess = f'Sowing treats all lines ending with \':\' as header, are you sure \'{textwrap.shorten(stripped, width=35)}\' qualifies? Use \'\:\' to escape the colon if not'
          self.add_diagnostic_from_source_range(
            Diagnostic.Kind.ERROR, self.diags.section_header_fishy_header, mess, eloc
          )
    return verdict

  def parse(self) -> PetscDocString:
    r"""Parse a docstring

    Returns
    -------
    docstring :
      the `PetscDocString` instance

    Raises
    ------
    KnownUnhandleableCursorError
      if the cursor has internal linkage and should not have its docstring checked
    """
    self.reset()
    self._check_valid_sowing_chars()
    self._check_floating()
    if not self._check_valid_cursor_linkage():
      # no point in continuing analysis, the docstring should not exist!
      raise KnownUnhandleableCursorError()
    self._check_valid_docstring_spacing()

    section      = self.sections.synopsis
    check_indent = section.check_indent_allowed()
    # if True we are in a verbatim block. We should not try to detect any kind of
    # headers until we reach the end of the verbatim block
    in_verbatim = 0
    prev_line   = ''

    raw_data: list[tuple[SourceRange, str, Verdict]] = []
    for lineno, line in enumerate(self.raw.splitlines(), start=self.extent.start.line):
      left_stripped = line.lstrip()
      stripped      = left_stripped.rstrip()
      if stripped.startswith('/*') or stripped.endswith('*/'):
        continue

      # TODO remove this, the current active section should be deciding what to do here instead
      # we shouldn't be checking indentation in verbatim blocks
      if stripped.startswith('.vb'):
        check_indent = False
        in_verbatim  = 1
      elif stripped.startswith('.ve'):
        check_indent = True # note we don't need to check indentation of line with .ve
        in_verbatim  = 0
      elif stripped.startswith('$'):
        # inline verbatim don't modify check flag but dont check indentation either
        in_verbatim = 2
      elif check_indent:
        self._check_valid_indentation(lineno, line, left_stripped)

      if in_verbatim == 0:
        heading_verdict = self.sections.is_heading(stripped, prev_line)
        heading_verdict = self._check_section_header_typo(heading_verdict, line, lineno)
        if heading_verdict > 0:
          # we may switch headings, we should check indentation
          if not check_indent:
            self._check_valid_indentation(lineno, line, left_stripped)
          self._check_valid_section_spacing(prev_line, lineno)
          new_section = self.sections.find(stripped.split(':', maxsplit=1)[0].strip().casefold())
          if new_section != section:
            raw_data     = section.consume(raw_data)
            section      = new_section
            check_indent = section.check_indent_allowed()
        else:
          heading_verdict = self._check_section_header_that_probably_should_not_be_one(
            heading_verdict, line, stripped, lineno
          )
      else:
        # verbatim blocks are never headings
        heading_verdict = Verdict.NOT_HEADING

      raw_data.append((self.make_source_range(line, line, lineno), line, heading_verdict))
      if in_verbatim == 2:
        # reset the dollar verbatim
        in_verbatim = 0
      prev_line = stripped

    section.consume(raw_data)
    for sec in self.sections:
      sec.setup(self)
    return self

del DocStringType
del DocStringTypeModifier
