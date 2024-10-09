#!/usr/bin/env python3
"""
# Created: Sun Nov 20 12:27:36 2022 (-0500)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import re
import enum
import difflib
import textwrap
import itertools
import collections

from ..._typing import *
from ..._error  import ParsingError

from .._diag    import DiagnosticManager, Diagnostic
from .._src_pos import SourceRange
from .._patch   import Patch

"""
==========================================================================================
Base Classes
==========================================================================================
"""
class DescribableItem:
  __slots__ = 'text', 'prefix', 'arg', 'description', 'sep', 'expected_sep'

  text: str
  prefix: str
  arg: str
  description: str
  sep: str
  expected_sep: str

  def __init__(self, raw: str, prefixes: Optional[Sequence[str]] = None, expected_sep: str = '-') -> None:
    r"""Construct a `DescribableItem`

    Parameters
    ----------
    raw :
      the raw line
    prefixes : optional
      the set of possible item prefixes
    expected_sep : optional
      the expected separator char between the arg and description
    """
    if prefixes is None:
      prefixes = tuple()
    text = raw.strip()
    sep  = expected_sep

    prefix, arg, descr = self.split_param(text, prefixes, sep)
    if not descr:
      found = False
      for sep in (',', '='):
        _, arg, descr = self.split_param(text, prefixes, sep)
        if descr:
          found = True
          break
      if not found:
        sep = ' '
        if prefix:
          arg = text.split(prefix, maxsplit=1)[1].strip()
        else:
          arg, *rest = text.split(maxsplit=1)
          if isinstance(rest, (list, tuple)):
            descr = rest[0] if len(rest) else ''
          assert isinstance(descr, str)
    self.text         = raw
    self.prefix       = prefix
    self.arg          = arg
    self.description  = descr
    self.sep          = sep
    self.expected_sep = expected_sep
    return

  @staticmethod
  def split_param(text: str, prefixes: Sequence[str], sep: str) -> tuple[str, str, str]:
    r"""Retrieve groups '([\.+-$])\s*([A-z,-]+) - (.*)'

    Parameters
    ----------
    text :
      the raw text line
    prefixes :
      the set of possible line prefixes to look for, empty set for no prexies
    sep :
      the separator char between the argument and its description

    Returns
    -------
    prefix :
      the detected prefix
    arg :
      the detected argument
    descr :
      the detected deescription

    Notes
    -----
    Any one of the returned values may be the empty string, which indicates that value was not detected.
    """
    stripped = text.strip()
    if not prefixes:
      prefix = ''
      rest   = stripped
    else:
      try:
        prefix = next(filter(stripped.startswith, prefixes))
      except StopIteration:
        prefix = ''
        rest   = stripped
      else:
        rest = stripped.split(prefix, maxsplit=1)[1].strip()
      assert len(prefix) >= 1
      assert rest
    arg, part_sep, descr = rest.partition(sep.join((' ', ' ')))
    if not part_sep:
      if rest.endswith(sep):
        arg = rest[:-1]
      elif sep + ' ' in rest:
        arg, _, descr = rest.partition(sep + ' ')
      # if we hit neither then there is no '-' in text, possible case of '[prefix] foo'?
    return prefix, arg.strip(), descr.lstrip()

  def arglen(self) -> int:
    r"""Return the argument length

    Returns
    -------
    alen :
      the length l such that self.text[:l] returns all text up until the end of the arg name
    """
    arg = self.arg
    return self.text.find(arg) + len(arg)

  def check(self, docstring: PetscDocStringImpl, section: SectionImpl, loc: SourceRange) -> None:
    r"""Check a `DescribableItem` for errors

    Parameters
    ----------
    docstring :
      the owning `PetscDocString` instance
    section :
      the owning section instance
    loc :
      the source range of the argument
    """
    name = section.transform(section.name)
    if self.sep != self.expected_sep:
      diag  = section.diags.wrong_description_separator
      mess  = f"{name} seems to be missing a description separator; I suspect you may be using '{self.sep}' as a separator instead of '{self.expected_sep}'. Expected '{self.arg} {self.expected_sep} {self.description}'"
    elif not self.description:
      diag = section.diags.missing_description
      mess = f"{name} missing a description. Expected '{self.arg} {self.expected_sep} a very useful description'"
    else:
      return # ok?
    docstring.add_diagnostic_from_source_range(Diagnostic.Kind.ERROR, diag, mess, loc)
    return

class DocBase:
  __slots__: tuple[str, ...] = tuple()

  @classmethod
  def __diagnostic_prefix__(cls, *flags: str) -> collections.deque[str]:
    return cls.diagnostic_flag('-'.join(flags))

  @classmethod
  def diagnostic_flag(cls, text: Union[str, collections.deque[str]], *, prefix: str = 'doc') -> collections.deque[str]:
    r"""Construct the diagnostic flag components

    Parameters
    ----------
    text :
      the base flag or collections.deque of flags
    prefix : optional
      the flag prefix

    Returns
    -------
    A collection.deque of flag components
    """
    if isinstance(text, str):
      ret = collections.deque((prefix, text))
      if prefix != 'doc':
        ret.appendleft('doc')
      return ret
    assert isinstance(text, collections.deque)
    if not text[0].startswith(prefix):
      text.appendleft(prefix)
    return text

@DiagnosticManager.register(
  ('section-header-missing', 'Verify that required sections exist in the docstring'),
  ('section-header-unique', 'Verify that appropriate sections are unique per docstring'),
  ('section-barren', 'Verify there are no sections containing a title and nothing else'),
  ('section-header-solitary', 'Verify that qualifying section headers are alone on their line'),
  ('section-header-spelling', 'Verify section headers are correctly spelled'),
  ('section-header-unknown', 'Verify that section header is known'),
)
class SectionBase(DocBase):
  """
  Container for a single section of the docstring
  """
  __slots__ = (
    'name', 'required', 'titles', 'keywords', 'raw', 'extent', '_lines', 'items', 'seen_headers',
    'solitary'
  )

  name: str
  required: bool
  titles: tuple[str, ...]
  keywords: tuple[str, ...]
  extent: SourceRange
  _lines: list[tuple[SourceRange, str, Verdict]]
  items: Any
  seen_headers: dict[str, list[SourceRange]]
  solitary: bool

  # to pacify type checkers...
  diags: DiagnosticMap

  LineInspector: TypeAlias = collections.abc.Callable[['PetscDocStringImpl', SourceRange, str, 'Verdict'], None]

  def __init__(self, name: str, required: bool = False, keywords: Optional[tuple[str, ...]] = None, titles: Optional[tuple[str, ...]] = None, solitary: bool = True) -> None:
    r"""Construct a `SectionBase`

    Parameters
    ----------
    name :
      the name of this section
    required : optional
      is this section required in the docstring
    keywords : optional
      keywords to help match an unknown title to a section
    titles :
      header-titles, i.e. "Input Parameter", or "Level", must be spelled correctly
    solitary : optional
      should the heading be alone on the line it sits on

    Notes
    -----
    In addition it has the following additional members
    raw :
      the raw text in the section
    extent :
      the SourceRange for the whole section
    _lines :
      a tuple of each line of text and its SourceRange in the section
    items :
      a container of extracted tokens of interest, e.g. the level value, options parameters,
      function parameters, etc
    """
    assert isinstance(name, str)
    titlename = name.title()
    if titles is None:
      titles = (titlename,)
    else:
      titles = tuple(titles)
    if keywords is None:
      keywords = (titlename,)
    else:
      keywords = tuple(keywords)

    self.name     = name
    self.required = required
    self.titles   = titles
    self.keywords = tuple(set(keywords + self.titles))
    self.solitary = solitary
    self.clear()
    return

  def __str__(self) -> str:
    return '\n'.join([
      f'Type:   {type(self)}',
      f'Name:   {self.name}',
      f'Extent: {self.extent}'
    ])

  def __bool__(self) -> bool:
    return bool(self.lines())

  def clear(self) -> None:
    r"""Clear a `SectionBase`

    Notes
    -----
    Resets the section to its default state
    """
    self.raw          = ''
    self.extent       = None # type: ignore[assignment]
    self._lines       = []
    self.items        = None
    self.seen_headers = {}
    return

  def lines(self, headings_only: bool = False) -> list[tuple[SourceRange, str, Verdict]]:
    r"""Retrieve the lines for this section

    Parameters
    ----------
    headings_only : optional
      retrieve only lines which are definitely headings

    Returns
    -------
    lines :
      the iterable of lines
    """
    if headings_only:
      return [(loc, line, verdict) for loc, line, verdict in self._lines if verdict > 0]
    return self._lines

  def consume(self, data: Collection[tuple[SourceRange, str, Verdict]]) -> list[tuple[SourceRange, str, Verdict]]:
    r"""Consume raw data and add it to the section

    Parameters
    ----------
    data :
      the container of raw data to consume

    Returns
    -------
    data :
      the consumed (and now empty) container
    """
    if data:
      self.lines().extend(data)
      self.raw    = '\n'.join(s for _, s, _ in self.lines())
      self.extent = SourceRange.from_locations(self.lines()[0][0].start, self.lines()[-1][0].end)
    return []

  def _do_setup(self, docstring: PetscDocStringImpl, inspect_line: LineInspector[PetscDocStringImpl]) -> None:
    r"""Do the actual setting up

    Parameters
    ----------
    docstring :
      the `PetscDocString` instance to use to log any errors
    inspect_line
      a callback to inspect each line

    Notes
    -----
    This is intended to be called by derived classes that wish to set a custom line inspector
    """
    seen = collections.defaultdict(list)
    for loc, line, verdict in self.lines():
      if verdict > 0:
        possible_header = line.split(':' if ':' in line else None, maxsplit=1)[0].strip()
        seen[possible_header.casefold()].append(
          docstring.make_source_range(possible_header, line, loc.start.line)
        )
      # let each section type determine if this line is useful
      inspect_line(docstring, loc, line, verdict)

    self.seen_headers = dict(seen)
    return

  def setup(self, docstring: PetscDocStringImpl) -> None:
    r"""Set up a section

    Parameters
    ----------
    docstring :
      the `PetscDocString` instance to use to log any errors

    Notes
    -----
    This routine is used to populate `self.items` and any other metadata before checking. As a rule,
    subclasses should do minimal error handling or checking here, gathering only the necessary
    statistics and data.
    """
    self._do_setup(docstring, lambda ds, loc, line, verdict: None)
    return

  def barren(self) -> bool:
    r"""Is this section empty?

    Returns
    -------
    ret :
      True if the sectino is empty, False otherwise
    """
    lines = self.lines()
    return not self.items and sum(not line.strip() for _, line, _ in lines) == len(lines) - 1

  @staticmethod
  def transform(text: str) -> str:
    r"""Transform a text into the expected title form

    Parameters
    ----------
    text :
      the string to transform

    Returns
    -------
    text :
      the transformed string

    Notes
    -----
    This is used for the equality check:
    ```
    if self.transform(text) in self.titles:
      # text could be a title if transformed
    else:
      # text needs further work
    ```
    """
    return text.title()

  def check_indent_allowed(self) -> bool:
    r"""Whether this section should check for indentation

    Returns
    -------
    ret :
      True if the linter should check indentation, False otherwise

    Notes
    -----
    This is used to disable indentation checking in e.g. source code blocks, but the implementation
    is very incomplete and likely needs a lot more work...
    """
    return True

  def _check_required_section_found(self, docstring: PetscDocStringImpl) -> None:
    r"""Check a required section does in fact exist

    Parameters
    ----------
    docstring :
      the `PetscDocString` owning the section
    """
    if not self and self.required:
      diag = self.diags.section_header_missing
      mess = f'Required section \'{self.titles[0]}\' not found'
      docstring.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, diag, mess, docstring.extent, highlight=False
      )
    return

  def _check_section_is_not_barren(self, docstring: PetscDocStringImpl) -> None:
    r"""Check that a section isn't just a solitary header out on its own

    Parameters
    ----------
    docstring :
      the `PetscDocString` owning the section
    """
    if self and self.barren():
      diag      = self.diags.section_barren
      highlight = len(self.lines()) == 1
      mess      = 'Section appears to be empty; while I\'m all for a good mystery, you should probably elaborate here'
      docstring.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, diag, mess, self.extent, highlight=highlight
      )
    return

  def _check_section_header_spelling(self, linter: Linter, docstring: PetscDocStringImpl, headings: Optional[Sequence[tuple[SourceRange, str, Verdict]]] = None, transform: Optional[Callable[[str], str]] = None) -> None:
    r"""Check that a section header is correctly spelled and formatted.

    Parameters
    ----------
    linter :
      the `Linter` instance to log any errors with
    docstring :
      the `PetscDocString` that owns this section
    headings : optional
      a set of heading lines
    transform : optional
      the text transformation function to transform a line into a heading

    Notes
    -----
    Sections may be found through fuzzy matching so this check asserts that a particular heading is
    actually correct.
    """
    if headings is None:
      headings = self.lines(headings_only=True)

    if transform is None:
      transform = self.transform

    diag = self.diags.section_header_spelling
    for loc, text, verdict in headings:
      before, sep, _ = text.partition(':')
      if not sep:
        # missing colon, but if we are at this point then we are pretty sure it is a
        # header, so we assume the first word is the header
        before, _, _ = docstring.guess_heading(text)

      heading = before.strip()
      if any(t in heading for t in self.titles):
        continue

      heading_loc = docstring.make_source_range(heading, text, loc.start.line)
      correct     = transform(heading)
      if heading != correct and any(t in correct for t in self.titles):
        docstring.add_diagnostic_from_source_range(
          Diagnostic.Kind.ERROR, diag,
          f'Invalid header spelling. Expected \'{correct}\' found \'{heading}\'',
          heading_loc, patch=Patch(heading_loc, correct)
        )
        continue

      try:
        matchname = difflib.get_close_matches(correct, self.titles, n=1)[0]
      except IndexError:
        warn_diag = docstring.make_diagnostic(
          Diagnostic.Kind.WARNING, self.diags.section_header_unknown,
          f'Unknown section \'{heading}\'', heading_loc
        )
        prevline = docstring.extent.start.line - 1
        loc      = SourceRange.from_positions(
          docstring.cursor.translation_unit, prevline, 1, prevline, -1
        )
        warn_diag.add_note(
          f'If this is indeed a valid heading, you can locally silence this diagnostic by adding \'// PetscClangLinter pragma disable: {DiagnosticManager.make_command_line_flag(warn_diag.flag)}\' on its own line before the docstring'
        ).add_note(
          Diagnostic.make_message_from_formattable(
            'add it here', crange=loc, highlight=False
          ),
          location=loc.start
        )
        docstring.add_diagnostic(warn_diag)
      else:
        docstring.add_diagnostic_from_source_range(
          Diagnostic.Kind.ERROR, diag,
          f'Unknown section header \'{heading}\', assuming you meant \'{matchname}\'',
          heading_loc, patch=Patch(heading_loc, matchname)
        )
    return

  def _check_duplicate_headers(self, docstring: PetscDocStringImpl) -> None:
    r"""Check that a particular heading is not repeated within the docstring

    Parameters
    ----------
    docstring :
      the `PetscDocString` owning the section
    """
    for heading, where in self.seen_headers.items():
      if len(where) <= 1:
        continue

      lasti           = len(where) - 1
      src_list        = []
      nbefore         = 2
      nafter          = 0
      prev_line_begin = 0
      for i, loc in enumerate(where):
        startline = loc.start.line
        if i:
          nbefore = startline - prev_line_begin - 1
          if i == lasti:
            nafter = 2
        src_list.append(loc.formatted(num_before_context=nbefore, num_after_context=nafter, trim=False))
        prev_line_begin = startline
      mess = "Multiple '{}' subheadings. Much like Highlanders, there can only be one:\n{}".format(
        self.transform(self.name), '\n'.join(src_list)
      )
      docstring.add_diagnostic(
        Diagnostic(Diagnostic.Kind.ERROR, self.diags.section_header_unique, mess, self.extent.start)
      )
    return

  def _check_section_header_solitary(self, docstring: PetscDocStringImpl, headings: Optional[Sequence[tuple[SourceRange, str, Verdict]]] = None) -> None:
    r"""Check that a section appears solitarily on its line, i.e. that there is no other text after ':'

    Parameters
    ----------
    docstring :
      the `PetscDocString` owning the section
    headings : optional
      a set of heading lines
    """
    if not self.solitary:
      return

    if headings is None:
      headings = self.lines(headings_only=True)

    for loc, text, verdict in headings:
      _, sep, after = text.partition(':')
      if not sep:
        head, _, _    = docstring.guess_heading(text)
        _, sep, after = text.partition(head)
        assert sep
      if after.strip():
        diag = self.diags.section_header_solitary
        mess = 'Heading must appear alone on a line, any content must be on the next line'
        docstring.add_diagnostic_from_source_range(
          Diagnostic.Kind.ERROR, diag, mess, docstring.make_source_range(after, text, loc.start.line)
        )
      break
    return

  def check(self, linter: Linter, cursor: Cursor, docstring: PetscDocStringImpl) -> None:
    r"""Perform a set of base checks for this instance

    Parameters
    ----------
    linter :
      the `Linter` instance to log any errors with
    cursor :
      the cursor to which the docstring this section belongs to
    docstring :
      the docstring to which this section belongs
    """
    self._check_required_section_found(docstring)
    self._check_section_header_spelling(linter, docstring)
    self._check_section_is_not_barren(docstring)
    self._check_duplicate_headers(docstring)
    self._check_section_header_solitary(docstring)
    return

@DiagnosticManager.register(
  ('matching-symbol-name','Verify that description matches the symbol name'),
  ('missing-description','Verify that a synopsis has a description'),
  ('wrong-description-separator','Verify that synopsis uses the right description separator'),
  ('verbose-description','Verify that synopsis descriptions don\'t drone on and on'),
  ('macro-explicit-synopsis-missing','Verify that macro docstrings have an explicit synopsis section'),
  ('macro-explicit-synopsis-valid-header','Verify that macro docstrings with explicit synopses have the right header include')
)
class Synopsis(SectionBase):
  _header_include_finder = re.compile(r'\s*#\s*include\s*[<"](.*)[>"]')
  _sowing_include_finder = re.compile(
    _header_include_finder.pattern + r'\s*/\*\s*I\s*(["<].*[>"])\s*I\s*\*/.*'
  )

  NameItemType: TypeAlias  = Tuple[Optional[SourceRange], str]
  BlurbItemType: TypeAlias = List[Tuple[SourceRange, str]]
  ItemsType                = TypedDict(
    'ItemsType',
    {
      'name'  : NameItemType,
      'blurb' : BlurbItemType,
    }
  )
  items: ItemsType

  diags: DiagnosticMap # satisfy type checkers

  class Inspector:
    __slots__ = 'cursor_name', 'lo_name', 'found_description', 'found_synopsis', 'capturing', 'items'

    class CaptureKind(enum.Enum):
      NONE        = enum.auto()
      DESCRIPTION = enum.auto()
      SYNOPSIS    = enum.auto()

    cursor_name: str
    lo_name: str
    found_description: bool
    found_synopsis: bool
    capturing: CaptureKind
    items: Synopsis.ItemsType

    def __init__(self, cursor: Cursor) -> None:
      self.cursor_name       = cursor.name
      self.lo_name           = self.cursor_name.casefold()
      self.found_description = False
      self.found_synopsis    = False
      self.capturing         = self.CaptureKind.NONE
      self.items             = {
        'name'  : (None, ''),
        'blurb' : []
      }
      return

    def __call__(self, ds: PetscDocStringImpl, loc: SourceRange, line: str, verdict: Verdict) -> None:
      r"""Look for the '<NAME> - description' block in a synopsis"""
      if self.found_description:
        return

      startline = loc.start.line
      if self.capturing == self.CaptureKind.NONE:
        pre, dash, rest = line.partition('-')
        if dash:
          rest = rest.strip()
        elif self.lo_name in line.casefold():
          pre  = self.cursor_name
          rest = line.split(self.cursor_name, maxsplit=1)[1].strip()
        else:
          return
        item = pre.strip()
        self.items['name'] = (ds.make_source_range(item, line, startline), item)
        self.items['blurb'].append((ds.make_source_range(rest, line, startline), rest))
        self.capturing = self.CaptureKind.DESCRIPTION # now capture the rest of the blurb
      else:
        assert self.capturing == self.CaptureKind.DESCRIPTION, 'Mixing blurb and synopsis capture?'
        if item := line.strip():
          self.items['blurb'].append((ds.make_source_range(item, line, startline), item))
        else:
          self.capturing         = self.CaptureKind.NONE
          self.found_description = True
      return

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('synopsis', *flags)

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct a `Synopsis`

    Parameters
    ----------
    *args :
      additional positional arguments to `SectionBase.__init__()`
    **kwargs :
      additional keyword arguments to `SectionBase.__init__()`
    """
    kwargs.setdefault('name', 'synopsis')
    kwargs.setdefault('required', True)
    kwargs.setdefault('keywords', ('Synopsis', 'Not Collective'))
    super().__init__(*args, **kwargs)
    return

  def barren(self) -> bool:
    return False # synoposis is never barren

  def _check_symbol_matches_synopsis_name(self: SynopsisImpl, docstring: PetscDocStringImpl, cursor: Cursor, loc: SourceRange, symbol: str) -> None:
    r"""Ensure that the name of the symbol matches that of the name in the custom synopsis (if provided)

    Parameters
    ----------
    docstring :
      the `PetscDocString` this section belongs to
    cursor :
      the cursor this docstring belongs to
    loc :
      the source range for symbol
    symbol :
      the name of the symbol in the docstring description

    Notes
    -----
    Checks:

    /*@
      FooBar - ....
      ^^^^^^------------------x-- Checks that these match
      ...             ________|
    @*/            vvvvvv
    PetscErrorCode FooBar(...)
    """
    if symbol != cursor.name:
      if len(difflib.get_close_matches(symbol, [cursor.name], n=1)):
        mess  = f"Docstring name '{symbol}' does not match symbol. Assuming you meant '{cursor.name}'"
        patch = Patch(loc, cursor.name)
      else:
        mess  = f"Docstring name '{symbol}' does not match symbol name '{cursor.name}'"
        patch = None
      docstring.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, self.diags.matching_symbol_name, mess, loc, patch=patch
      )
    return

  def _check_synopsis_description_separator(self: SynopsisImpl, docstring: PetscDocStringImpl, start_line: int) -> None:
    r"""Ensure that the synopsis uses the proper separator

    Parameters
    ----------
    docstring :
      the docstring this section belongs to
    start_line :
      the line number of the description
    """
    for sloc, sline, _ in self.lines():
      if sloc.start.line == start_line:
        DescribableItem(sline, expected_sep='-').check(docstring, self, sloc)
        break
    return

  def _check_blurb_length(self: SynopsisImpl, docstring: PetscDocStringImpl, cursor: Cursor, blurb_items: Synopsis.BlurbItemType) -> None:
    r"""Ensure the blurb is not too wordy

    Parameters
    ----------
    docstring :
      the docstring this section belongs to
    cursor :
      the cursor this docstring belongs to
    items :
      the synopsis items
    """
    total_blurb = [line for _, line in blurb_items]
    word_count  = sum(len(l.split()) for l in total_blurb)
    char_count  = sum(map(len, total_blurb))

    max_char_count = 250
    max_word_count = 40
    if char_count > max_char_count and word_count > max_word_count:
      mess = f"Synopsis for '{cursor.name}' is too long (must be at most {max_char_count} characters or {max_word_count} words), consider moving it to Notes. If you can't explain it simply, then you don't understand it well enough!"
      docstring.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, self.diags.verbose_description, mess, self.extent, highlight=False
      )
    return

  def _syn_common_checks(self: SynopsisImpl, linter: Linter, cursor: Cursor, docstring: PetscDocStringImpl) -> None:
    r"""Perform the common set of checks for all synopses

    Parameters
    ----------
    linter :
      the `Linter` instance to log any errors with
    cursor :
      the cursor to which the docstring this section belongs to
    docstring :
      the docstring to which this section belongs

    Notes
    -----
    Does not call `super().check()`! Therefore this should be used as the epilogue to your synopsis
    checks, after any potential early returns
    """
    items                 = self.items
    name_loc, symbol_name = items['name']
    assert name_loc is not None # pacify type checkers
    self._check_symbol_matches_synopsis_name(docstring, cursor, name_loc, symbol_name)
    self._check_synopsis_description_separator(docstring, name_loc.start.line)
    self._check_blurb_length(docstring, cursor, items['blurb'])
    return

@DiagnosticManager.register(
  ('alignment', 'Verify that parameter list entries are correctly white-space aligned'),
  ('prefix', 'Verify that parameter list entries begin with the correct prefix'),
  ('missing-description', 'Verify that parameter list entries have a description'),
  ('wrong-description-separator', 'Verify that parameter list entries use the right description separator'),
  ('solitary-parameter', 'Verify that each parameter has its own entry'),
)
class ParameterList(SectionBase):
  __slots__ = ('prefixes', )

  prefixes: Tuple[str, ...]

  ItemsType: TypeAlias = Dict[int, List[Tuple[SourceRange, DescribableItem, int]]]
  items: ItemsType

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('param-list', *flags)

  def __init__(self, *args, prefixes: Optional[tuple[str, ...]] = None, **kwargs) -> None:
    r"""Construct a `ParameterList`

    Parameters
    ----------
    prefixes : optional
      a set of prefixes which the parameter list starts with
    """
    if prefixes is None:
      prefixes = ('+', '.', '-')

    self.prefixes = prefixes
    kwargs.setdefault('name', 'parameters')
    super().__init__(*args, **kwargs)
    return

  def check_indent_allowed(self) -> bool:
    r"""Whether `ParameterList`s should check for indentation

    Returns
    -------
    ret :
      Always True
    """
    return False

  def check_aligned_descriptions(self, ds: PetscDocStringImpl, group: Sequence[tuple[SourceRange, DescribableItem, int]]) -> None:
    r"""Verify that the position of the '-' before the description for each argument is aligned

    Parameters
    ----------
    ds :
      the `PetscDocString` instance which owns this section
    group :
      the item group to check, each entry is a tuple of src_range for the item, the `DescribableItem`
      instance, and the arg len for that item
    """
    align_diag  = self.diags.alignment
    group_args  = [item.arg for _, item, _ in group]
    lens        = list(map(len, group_args))
    max_arg_len = max(lens, default=0)
    longest_arg = group_args[lens.index(max_arg_len)] if lens else 'NO ARGS'

    for loc, item, _ in group:
      pre   = item.prefix
      arg   = item.arg
      descr = item.description
      text  = item.text
      fixed = f'{pre} {arg:{max_arg_len}} - {descr}'
      try:
        diff_index = next(
          i for i, (a1, a2) in enumerate(itertools.zip_longest(text, fixed)) if a1 != a2
        )
      except StopIteration:
        assert text == fixed # equal
        continue

      if diff_index <= text.find(pre):
        mess = f'Prefix \'{pre}\' must be indented to column (1)'
      elif diff_index <= text.find(arg):
        mess = f'Argument \'{arg}\' must be 1 space from prefix \'{pre}\''
      else:
        mess = f'Description \'{textwrap.shorten(descr, width=35)}\' must be aligned to 1 space from longest (valid) argument \'{longest_arg}\''

      eloc = ds.make_source_range(text[diff_index:], text, loc.end.line)
      ds.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, align_diag, mess, eloc, patch=Patch(eloc, fixed[diff_index:])
      )
    return

  def setup(self, ds: PetscDocStringImpl, parameter_list_prefix_check: Optional[Callable[[ParameterList, PetscDocString, ItemsType], ItemsType]] = None) -> None:
    r"""Set up a `ParmeterList`

    Parameters
    ----------
    ds :
      the `PetscDocString` instance for this section
    parameters_list_prefix_check : optional
      a callable to check the prefixes of each item
    """
    groups: collections.defaultdict[
      int,
      list[tuple[SourceRange, DescribableItem, int]]
    ]          = collections.defaultdict(list)
    subheading = 0

    def inspector(ds: PetscDocStringImpl, loc: SourceRange, line: str, verdict: Verdict) -> None:
      if not line or line.isspace():
        return

      if verdict > 0 and len(groups.keys()):
        nonlocal subheading
        subheading += 1
      lstp = line.lstrip()
      # .ve and .vb might trip up the prefix detection since they start with '.'
      if lstp.startswith(self.prefixes) and not lstp.startswith(('.vb', '.ve')):
        item = DescribableItem(line, prefixes=self.prefixes)
        groups[subheading].append((loc, item, item.arglen()))
      return

    super()._do_setup(ds, inspector)
    items = dict(groups)
    if parameter_list_prefix_check is not None:
      assert callable(parameter_list_prefix_check)
      items = parameter_list_prefix_check(self, ds, items)
    self.items = items
    return

  def _check_opt_starts_with(self, docstring: PetscDocStringImpl, item: tuple[SourceRange, DescribableItem, int], entity_name: str, char: str) -> None:
    r"""Check an option starts with the given prefix

    Parameters
    ----------
    docstring :
      the `PetscDocString` that owns this section
    item :
      the `SourceRange`, `DescribableItem`, arg len triple for the line
    entity_name :
      the name of the entity to which the param list belongs, e.g. 'function' or 'enum'
    char :
      the prefix character
    """
    loc, descr_item, _ = item
    pre                = descr_item.prefix
    if pre != char:
      eloc = docstring.make_source_range(pre, descr_item.text, loc.start.line)
      mess = f'{entity_name} parameter list entry must start with \'{char}\''
      docstring.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, self.diags.prefix, mess, eloc, patch=Patch(eloc, char)
      )
    return

  def _check_prefixes(self, docstring: PetscDocStringImpl) -> None:
    r"""Check all prefixes in the section for validity

    Parameters
    ----------
    docstring :
      the `PetscDocString` instance owning this section
    """
    for key, opts in sorted(self.items.items()):
      lopts = len(opts)
      assert lopts >= 1, f'number of options {lopts} < 1, key: {key}, items: {self.items}'

      if lopts == 1:
        # only 1 option, should start with '.'
        self._check_opt_starts_with(docstring, opts[0], 'Solitary', '.')
      else:
        # more than 1, should be '+', then however many '.', then last is '-'
        self._check_opt_starts_with(docstring, opts[0], 'First multi', '+')
        for opt in opts[1:-1]:
          self._check_opt_starts_with(docstring, opt, 'Multi', '.')
        self._check_opt_starts_with(docstring, opts[-1], 'Last multi', '-')
    return

  def check(self, linter: Linter, cursor: Cursor, docstring: PetscDocStringImpl) -> None:
    r"""Perform all checks for this param list

    Parameters
    ----------
    linter :
      the `Linter` instance to log any errors with
    cursor :
      the cursor to which the docstring this section belongs to
    docstring :
      the docstring to which this section belongs
    """
    super().check(linter, cursor, docstring)
    self._check_prefixes(docstring)
    return

class Prose(SectionBase):
  ItemsType: TypeAlias = Dict[int, Tuple[Tuple[SourceRange, str], List[Tuple[SourceRange, str]]]]
  items: ItemsType

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('prose', *flags)

  def setup(self, ds: PetscDocStringImpl) -> None:
    r"""Set up a `Prose`

    Parameters
    ----------
    ds :
      the `PetscDocString` instance for this section

    Raises
    ------
    ParsingError
      if a subheading does not exist yet??
    """
    subheading = 0
    self.items = {}

    def inspector(ds: PetscDocStringImpl, loc: SourceRange, line: str, verdict: Verdict) -> None:
      if verdict > 0:
        head, _, rest = line.partition(':')
        head          = head.strip()
        assert head, f'No heading in PROSE section?\n\n{loc.formatted(num_context=5)}'
        if self.items.keys():
          nonlocal subheading
          subheading += 1
        start_line = loc.start.line
        self.items[subheading] = (
          (ds.make_source_range(head, line, start_line), head),
          [(ds.make_source_range(rest, line, start_line), rest)] if rest else []
        )
      elif line.strip():
        try:
          self.items[subheading][1].append((loc, line))
        except KeyError as ke:
          raise ParsingError from ke
      return

    super()._do_setup(ds, inspector)
    return

class VerbatimBlock(SectionBase):
  ItemsType: TypeAlias = Dict[int, List[int]]
  items: ItemsType

  def setup(self, ds: PetscDocStringImpl) -> None:
    r"""Set up a `VerbatimBlock`

    Parameters
    ----------
    ds :
      the `PetscDocString` instance for this section
    """
    items = {}

    class Inspector:
      __slots__ = 'codeblocks', 'startline'

      codeblocks: int
      startline: int

      def __init__(self, startline: int) -> None:
        self.codeblocks = 0
        self.startline  = startline
        return

      def __call__(self, ds: PetscDocStringImpl, loc: SourceRange, line: str, verdict: Verdict) -> None:
        sub   = self.codeblocks
        lstrp = line.lstrip()
        if lstrp.startswith('.vb'):
          items[sub] = [loc.start.line - self.startline]
        elif lstrp.startswith('.ve'):
          assert len(items[sub]) == 1
          items[sub].append(loc.start.line - self.startline + 1)
          self.codeblocks += 1
        return

    super()._do_setup(ds, Inspector(self.extent.start.line if self else 0))
    self.items = items
    return

@DiagnosticManager.register(
  ('formatting', 'Verify that inline lists are correctly white-space formatted')
)
class InlineList(SectionBase):
  ItemsEntry: TypeAlias = Tuple[Tuple[str, str], List[Tuple[SourceRange, str]]]
  ItemsType: TypeAlias  = Tuple[ItemsEntry, ...]
  items: ItemsType

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct an `InlineList`

    Parameters
    ----------
    *args :
      additional positional parameters to `SectionBase.__init__()`
    **kwargs :
      additional keywords parameters to `SectionBase.__init__()`
    """
    kwargs.setdefault('solitary', False)
    super().__init__(*args, **kwargs)
    return

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('inline-list', *flags)

  def check_indent_allowed(self) -> bool:
    r"""Whether this section should check for indentation

    Returns
    -------
    ret :
      always False
    """
    return False

  def setup(self, ds: PetscDocStringImpl) -> None:
    r"""Set up an `InlineList`

    Parameters
    ----------
    ds :
      the `PetscDocString` instance for this section
    """
    items: list[InlineList.ItemsEntry] = []
    titles                             = set(map(str.casefold, self.titles))

    def inspector(ds: PetscDocStringImpl, loc: SourceRange, line: str, verdict: Verdict) -> None:
      rest = (line.split(':', maxsplit=2)[1] if ':' in line else line).strip()
      if not rest:
        return

      if ':' not in rest:
        # try and see if this is one of the bad-egg lines where the heading is missing
        # the colon
        bad_title = next(filter(lambda t: t.casefold() in titles, rest.split()), None)
        if bad_title:
          # kind of a hack, we just erase the bad heading with whitespace so it isnt
          # picked up below in the item detection
          rest = rest.replace(bad_title, ' ' * len(bad_title))

      start_line = loc.start.line
      offset     = 0
      sub_items  = []
      for sub in filter(bool, map(str.strip, rest.split(','))):
        subloc = ds.make_source_range(sub, line, start_line, offset=offset)
        offset = subloc.end.column - 1
        sub_items.append((subloc, sub))
      if sub_items:
        items.append(((line, rest), sub_items))
      return

    super()._do_setup(ds, inspector)
    self.items = tuple(items)
    return

  def _check_whitespace_formatting(self, docstring: PetscDocStringImpl) -> None:
    r"""Ensure that inline list ensures are on the same line and 1 space away from the title

    Parameters
    ----------
    docstring :
      the `PetscDocString` which owns this section
    """
    format_diag = self.diags.formatting
    base_mess   = f'{self.transform(self.name)} values must be (1) space away from colon not ({{}})'
    for (line, line_after_colon), sub_items in self.items:
      colon_idx = line.find(':')
      if colon_idx < 0:
        continue

      correct_offset = colon_idx + 2
      rest_idx       = line.find(line_after_colon)
      if rest_idx == correct_offset:
        continue

      nspaces = rest_idx - correct_offset
      if rest_idx > correct_offset:
        sub    = ' ' * nspaces
        offset = correct_offset
        fix    = ''
      else:
        sub    = ':'
        offset = colon_idx
        fix    = ': '
      floc = docstring.make_source_range(sub, line, sub_items[0][0].start.line, offset=offset)
      docstring.add_diagnostic_from_source_range(
        Diagnostic.Kind.ERROR, format_diag, base_mess.format(nspaces + 1), floc, patch=Patch(floc, fix)
      )
    return

  def check(self, linter: Linter, cursor: Cursor, docstring: PetscDocStringImpl) -> None:
    r"""Perform all checks for this inline list

    Parameters
    ----------
    linter :
      the `Linter` instance to log any errors with
    cursor :
      the cursor to which the docstring this section belongs to
    docstring :
      the docstring to which this section belongs
    """
    super().check(linter, cursor, docstring)
    self._check_whitespace_formatting(docstring)
    return
