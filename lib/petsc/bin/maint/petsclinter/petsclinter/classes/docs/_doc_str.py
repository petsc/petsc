#!/usr/bin/env python3
"""
# Created: Mon Jun 20 18:58:57 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import re
import enum
import difflib
import textwrap
import collections
import clang.cindex as clx
import petsclinter  as pl

from .._diag    import DiagnosticManager, Diagnostic
from .._linter  import Linter
from .._cursor  import Cursor
from .._src_pos import SourceRange, SourceLocation
from .._patch   import Patch

from ._doc_section_base import DocBase, SectionBase
from ._doc_section      import DefaultSection, Synopsis, FunctionParameterList, OptionDatabaseKeys
from ._doc_section      import Level, Notes, FortranNotes, DeveloperNotes, SourceCode, References
from ._doc_section      import SeeAlso

@enum.unique
class Verdict(enum.IntEnum):
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

# expressions that usually end in an unescpaped colon causing the resulting sentence to be
# considered a title
_suspicious_expression_regex = re.compile(
  r'|'.join(
    f'{expr}:$' for expr in {
      r'follows', r'following.*', r'example', r'instance', r'one of.*', r'available.*include',
      r'supports.*approaches.*', r'see.*user.*manual', r'y. saad, iterative methods.*philadelphia',
      r'default', r'in .* case.*', r'use the.*'
    }
  )
)

class SectionNotFoundError(pl.BaseError):
  """
  Exception thrown when a section is searched for, not found, and strict mode was enabled
  """
  pass

class GuessHeadingFailError(pl.BaseError):
  """
  Exception thrown then sections fails to guess the appropriate heading for a line
  """
  pass

class Sections:
  __slots__ = '_verbose', '_sections', '_findcache', '_cachekey'

  def __init__(self, *args, verbose=False):
    assert len({s.name for s in args}) == len(args)
    self._verbose  = verbose
    self._sections = collections.OrderedDict()
    self.add_sections(*args)
    return

  def __getattr__(self, attr):
    sections = self._sections
    try:
      return sections[attr]
    except KeyError as ke:
      replaced_attr = attr.replace('_', ' ')
      try:
        return sections[replaced_attr]
      except KeyError:
        pass
    raise AttributeError(attr)

  def __iter__(self):
    yield from self._sections.values()

  def __contains__(self, section):
    return self.registered(section)


  def _reset_cache(self):
    self._cachekey  = tuple(self._sections.keys())
    self._findcache = {self._cachekey : {}}
    return

  def _print(self, *args, **kwargs):
    if self._verbose:
      pl.sync_print(*args, **kwargs)
    return

  def set_verbose(self, verbose):
    old_verbose   = self._verbose
    self._verbose = verbose
    return old_verbose

  def find(self, heading, cache_result=True, strict=False):
    lohead   = heading.casefold()
    sections = self._sections
    cache    = self._findcache[self._cachekey]
    try:
      return sections[cache[lohead]]
    except KeyError:
      pass

    section_names = sections.keys()
    matched       = difflib.get_close_matches(heading, section_names, n=1)
    try:
      matched = matched[0]
    except IndexError:
      found_match = 0
    else:
      found_match = 1
      reason      = 'name' # delete me

    if not found_match:
      keywords = [(kw, s.name) for kw, s in self.keywords(sections=True)]
      kw_only  = [k for k, _ in keywords]
      matched  = difflib.get_close_matches(heading, kw_only, n=1)
      try:
        matched = matched[0]
      except IndexError:
        pass
      else:
        found_match = 2
        reason      = 'keyword' # delete me

    if not found_match:
      # try if we can find a sub-word
      # if heading splits into more than 3 params, then chances are its being mislabeled
      # as a heading anyways
      for head in heading.split(maxsplit=3):
        matched = difflib.get_close_matches(head, kw_only, n=1, cutoff=0.8)
        try:
          # higher cutoff, we have to be pretty sure of a match when using subwords,
          # because it's a lot easier for false positives
          matched = matched[0]
        except IndexError:
          continue
        else:
          reason      = 'subword' # delete me
          found_match = 3
          break

    if found_match:
      if found_match > 1:
        # found via keyword or subword
        matched = next(filter(lambda item: item[0] == matched, keywords))[1]
      self._print('**** CLOSEST MATCH FOUND {:{}} FROM {:{}} FOR {}'.format(matched,max(map(len,section_names)),reason,len('not found'),heading))
    else:
      self._print(
        80*'*',
        f'UNHANDLED POSSIBLE HEADING! (strict = {strict}, cached = {cache_result})',
        heading,
        80*'*',
        sep='\n'
      )
      if strict:
        raise SectionNotFoundError(heading)
      # this should be handled
      # when in doubt, it's probably notes
      reason  = 'not found'
      matched = 'UNKNOWN'
      maxlen  = max(map(len, section_names))
      string  = '*********** DEFAULTED TO {:{}} FROM {} FOR {}'
      self._print(string.format(f'UNKNOWN (strict = {strict})', maxlen, reason, heading))

    if cache_result:
      cache[lohead] = matched
    return sections[matched]

  def registered(self, section):
    if isinstance(section, SectionBase):
      return section.name in self._sections
    if isinstance(section, str):
      return section in self._sections
    raise NotImplementedError(type(section))

  def add_sections(self, *args):
    for section in args:
      assert not self.registered(section), f'overwriting section {section}'
      self._sections[section.name] = section
    self._reset_cache()
    return

  def __unpack_attr_list(self, attr_name, sections):
    if sections:
      gen = ((attr, section) for section in self for attr in getattr(section, attr_name))
    else:
      gen = (attr for section in self for attr in getattr(section, attr_name))
    return gen

  def titles(self, sections=False):
    return self.__unpack_attr_list('titles', sections)

  def keywords(self, sections=False):
    return self.__unpack_attr_list('keywords', sections)

  def is_heading(self, item):
    if isinstance(item, tuple):
      assert len(item) == 2
      assert isinstance(item[0], SourceRange) and isinstance(item[1], str)
      text = item[1]
    elif isinstance(item, str):
      text = item
    else:
      raise NotImplementedError(type(item))

    def handle_header_with_colon(text):
      if text.endswith('\:'):
        return Verdict.NOT_HEADING

      textlo = text.casefold()
      if any(map(textlo.startswith, (t.casefold() + ':' for t in self.titles()))):
        return Verdict.IS_HEADING

      if text.endswith(':'):
        if any(map(text.__contains__, (' - ', '=', '(', ')', '%', '$', '@', '#', '!', '^', '&', '+'))):
          return Verdict.IS_HEADING_BUT_PROBABLY_SHOULDNT_BE

        if _suspicious_expression_regex.search(textlo) is None:
          return Verdict.IS_HEADING
        return Verdict.IS_HEADING_BUT_PROBABLY_SHOULDNT_BE

      try:
        guessed = self.guess_heading(text, cache_result=False, strict=True)
      except SectionNotFoundError:
        return Verdict.NOT_HEADING
      return Verdict.IS_HEADING if guessed else Verdict.NOT_HEADING

    def handle_header_without_colon(text):
      try:
        next(filter(text.casefold().startswith, map(str.casefold, self.titles())))
      except StopIteration:
        return Verdict.NOT_HEADING
      return Verdict.MAYBE_HEADING


    text = text.strip()
    if not text or text.startswith(('+ ', '. ', '- ', '$', '.vb', '.ve')):
      return Verdict.NOT_HEADING
    if ':' in text:
      return handle_header_with_colon(text)
    return handle_header_without_colon(text)

  def guess_heading(self, line, **kwargs):
    def guess(item):
      titles = self.find(item, **kwargs).titles
      if len(titles) == 1:
        return titles
      return difflib.get_close_matches(item, titles, n=1)

    strp = line.split(':', maxsplit=1)[0].strip()
    if strp:
      attempts = (strp, strp.split(maxsplit=1)[0].strip(), strp.title())

      for attempt, match_found in zip(attempts, map(guess, attempts)):
        if match_found:
          return attempt, match_found[0]
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
  Type     = DocStringType
  Modifier = DocStringTypeModifier
  sections = Sections(
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

  def __init__(self, linter, cursor, indent=2):
    if not isinstance(linter, Linter):
      raise ValueError(type(linter))

    self.sections.set_verbose(linter.verbose)
    self._linter          = linter
    self.cursor           = Cursor.cast(cursor)
    self.raw, self.extent = self._get_sanitized_comment_and_range_from_cursor(self.cursor)
    self.indent           = indent
    self.type             = self.Type.UNKNOWN
    self.type_mod         = self.Modifier.NONE
    self._attr            = self._default_attributes()
    return

  @staticmethod
  def _default_attributes():
    return dict()

  @classmethod
  def register_section(cls, section):
    """
    Register SECTION with the list of docstring sections to check for
    """
    return cls.sections.add_section(section)

  @classmethod
  def _is_valid_docstring(cls, cursor, raw, doc_extent):
    """
    Determine whether docstring in RAW (of cursor CURSOR) is a valid sowing docstring worth checking.
    Returns True if it is, and False otherwise
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
    have_title = sum(f'{title}:' in rawlo for title in map(str.casefold, cls.sections.titles()))
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
  def _get_sanitized_comment_and_range_from_cursor(cls, cursor):
    if not isinstance(cursor, Cursor):
      raise ValueError(type(cursor))

    raw, extent = cursor.get_comment_and_range()
    extent      = SourceRange.cast(extent, tu=cursor.translation_unit)

    if not cls._is_valid_docstring(cursor, raw, extent):
      raise pl.ParsingError('Not a docstring')

    rawlines = raw.splitlines()
    comments = [i for i, line in enumerate(rawlines) if line.lstrip().startswith('/*')]
    if len(comments) > 1:
      # this handles the following case:
      #
      # /* a dummy comment that is attributed to the symbol */
      # /*
      #   the real docstring comment, note no empty line between this and the previous!
      # */
      # <the symbol>
      offset = comments[-1]
      raw    = '\n'.join(rawlines[offset:])
      extent = extent.resized(lbegin=offset, cbegin=None, cend=None)
    return raw, extent

  @classmethod
  def is_heading(cls, *args, **kwargs):
    return cls.sections.is_heading(*args, **kwargs)

  @classmethod
  def _get_is_heading(cls, section):
    return getattr(section, 'is_heading', cls.sections.is_heading)

  @staticmethod
  def make_error_message(message, crange=None, num_context=2, **kwargs):
    if crange is None:
      crange_text = ''
    else:
      crange_text = crange.formatted(num_context=num_context, **kwargs)
    return f'{message}:\n{crange_text}'

  def make_source_location(self, lineno, col):
    return SourceLocation.from_position(self.cursor.translation_unit, lineno, col)

  def make_source_range(self, token, string, lineno, offset=0):
    col_begin = string.index(token, offset) + 1
    col_end   = col_begin + len(token)
    return SourceRange.from_positions(self.cursor.translation_unit, lineno, col_begin, lineno, col_end)

  def make_diagnostic(self, diag_flag, msg, src_range, patch=None, **kwargs):
    if src_range is None:
      src_range = self.extent
    else:
      src_range = SourceRange.cast(src_range)
    return Diagnostic(
      diag_flag, self.make_error_message(msg, crange=src_range, **kwargs), src_range.start, patch=patch
    )

  def add_error_from_diagnostic(self, diagnostic, cursor=None):
    """
    Add an error from a fully-formed diagnostic

    diagnostic - The diagnostic describing the error
    cursor     - The cursor to attach the error to, None for currect cursors
    """
    if cursor is None:
      cursor = self.cursor
    return self._linter.add_error_from_cursor(cursor, diagnostic)

  def add_error_from_source_range(self, diag_flag, msg, src_range, **kwargs):
    """
    Log an error from a given source range

    diag_flag - the diagnostic flag to control the error
    msg       - the diagnostic message describing the problem in detail
    src_range - the SourceRange which shows the error in the source
    kwargs    - any additional keyword arguments to make_diagnostic()
    """
    return self.add_error_from_diagnostic(self.make_diagnostic(diag_flag, msg, src_range, **kwargs))

  def clear(self):
    for section in self.sections:
      section.clear()
    self._attr = self._default_attributes()
    return

  def guess_heading(self, line, **kwargs):
    return self.sections.guess_heading(line, **kwargs)

  def _check_floating(self):
    """
    check that the docstring isn't a floating docstring, i.e. for a mansection or particular type
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
          raise pl.ParsingError('DON\'T KNOW HOW TO PROPERLY HANDLE FLOATING DOCSTRINGS')
        break
    return

  def _check_valid_cursor_linkage(self):
    """
    check that a cursor has external linkage, there is no point producing a manpage for function
    that is impossible to call
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
      begin_sowing_range = self._attr['sowing_char_range']
      diag               = self.make_diagnostic(
        self.diags.internal_linkage,
        f'A sowing docstring for a symbol with internal linkage is pointless {Diagnostic.FLAG_SUBST}!',
        self.extent, highlight=False
      ).add_note(
        f'\'{cursor.displayname}\' is declared \'{linked_cursor_name}\' at {Cursor.cast(linkage_cursor)}',
        location=linkage_cursor.extent.start
      ).add_note(
        'If this docstring is meant as developer-only documentation, remove the sowing chars from the docstring declaration. The linter will then ignore this docstring.'
      ).add_note(
        f'Sowing chars declared here:\n{begin_sowing_range.formatted(num_context=2)}',
        location=begin_sowing_range.start
      )
      self.add_error_from_diagnostic(diag)
    return not pointless

  def _check_valid_sowing_chars(self):
    """
    check that the sowing prefix and postfix match the expected and are symmetric
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
      self.add_error_from_source_range(diag_name, mess, begin_sowing_range)
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
          raise pl.ParsingError
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
            self.add_error_from_source_range(
              diag_name,
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
      self.add_error_from_source_range(
        diag_name, mess, restloc, patch=Patch(restloc, '\n' + (' '*self.indent) + rest)
      )

    # now check the end
    line       = splitlines[-1]
    end_sowing = line.split('*/')[0].split()
    try:
      end_sowing = end_sowing[-1]
    except IndexError:
      pass
    else:
      if sorted(end_sowing) != sorted(begin_sowing) and 0:
        # TODO: REVIEW: should this check exist?
        correct = begin_sowing[::-1]
        endline = self.extent.end.line
        mess    = f'Invalid comment end line, sowing identifier(s) do not match begin identifier(s). Expected \'{correct}*/\' found \'{end_sowing}*/\''
        patch   = Patch(self.make_source_range(line, line, endline), line.replace(end_sowing, correct))
        self.add_error_from_source_range(
          diag_name, mess, self.make_source_range(end_sowing, line, endline), patch=patch
        )
    return

  def _check_valid_docstring_spacing(self):
    """
    Check that the docstring itself is flush against the thing it describes, i.e.:

    /*
      PetscFooBar - ...
    */
    PetscErrorCode PetscFooBar(...)

    not

    /*
      PetscFooBar - ...
    */

    PetscErrorCode PetscFooBar(...)
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
      self.add_error_from_source_range(diag, mess, eloc, highlight=False, patch=Patch(floc, ''))
    return

  def _check_valid_indentation(self, lineno, line, lstripped):
    """
    If the line is regular (not empty, or a parameter list), check that line is indented correctly
    """
    linelen = len(line)
    if linelen:
      indent       = linelen - len(lstripped)
      expected_ind = 0 if line.startswith(('.', '+', '-', '$')) else self.indent
      if indent != expected_ind:
        diag = self.diags.indentation
        loc  = self.make_source_range(' ' * indent, line, lineno)
        mess = f'Invalid indentation ({indent}), all regular (non-empty, non-parameter, non-seealso) text must be indented to {self.indent} columns'
        self.add_error_from_source_range(diag, mess, loc, patch=Patch(loc, ' ' * expected_ind))
    return

  def _check_valid_section_spacing(self, prevline, lineno):
    """
    Check that sections have at least 1 empty line between them, i.e.:

    Notes:
    asdadsadasdads

    Example Usage:
    asdasdasd

    not

    Notes:
    asdasdasd
    Example Usage:
    asdadasd
    """
    if prevline and not prevline.isspace():
      loc = self.make_source_range('', '', lineno)
      self.add_error_from_source_range(
        self.diags.section_spacing,
        'Missing empty line between sections, must have one before this section',
        loc, highlight=False, patch=Patch(loc, '\n')
      )
    return

  def _check_section_header_typo(self, heading, line, lineno):
    """
    Check that a section header that looks like a section header is actually one
    """
    if heading == Verdict.MAYBE_HEADING:
      try:
        name, matched = self.guess_heading(line)
      except GuessHeadingFailError as ghfe:
        # Not being able to guess the heading here is OK since we only *think* it's a
        # heading
        self.sections._print(ghfe)
        return Verdict.NOT_HEADING
      if ':' in line:
        mess = f'Line seems to be a section header but doesn\'t directly end with with \':\', did you mean \'{matched}\'?'
      else:
        mess = f'Line seems to be a section header but missing \':\', did you mean \'{matched}:\'?'
      diag = self.diags.section_header_maybe_header
      self.add_error_from_source_range(diag, mess, self.make_source_range(name, line, lineno))
    return heading

  def _check_section_header_that_probably_should_not_be_one(self, heading, line, lineno):
    """
    check that a section header that ends with ':' is not really a header
    """
    if heading < 0:
      try:
        possible_heading, section_guess = self.guess_heading(line, cache_result=False)
      except GuessHeadingFailError as ghfe:
        # Not being able to guess the heading here is OK since we aren't sure this isn't a
        # heading after all
        self.sections._print(ghfe)
        return Verdict.NOT_HEADING
      if section_guess == '__UNKNOWN_SECTION__':
        assert not line.endswith(r'\:')
        eloc = self.make_source_range(':', line, lineno, offset=line.rfind(':'))
        mess = f'Sowing treats all lines ending with \':\' as header, are you sure \'{textwrap.shorten(line.strip(), width=35)}\' qualifies? Use \'\:\' to escape the colon if not'
        self.add_error_from_source_range(self.diags.section_header_fishy_header, mess, eloc)
    return heading

  def parse(self):
    self.clear()
    self._check_valid_sowing_chars()
    self._check_floating()
    if not self._check_valid_cursor_linkage():
      raise pl.ParsingError # no point in continuing analysis, the docstring should not exist!
    self._check_valid_docstring_spacing()

    raw_data     = []
    heading_data = []
    section      = self.sections.synopsis
    check_indent = section.check_indent_allowed()
    is_heading   = self._get_is_heading(section)
    # if True we are in a verbatim block. We should not try to detect any kind of
    # headers until we reach the end of the verbatim block
    in_verbatim = 0
    for lineno, line in enumerate(self.raw.splitlines(), start=self.extent.start.line):
      stripped = line.strip()
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
        self._check_valid_indentation(lineno, line, stripped)

      if in_verbatim == 0:
        heading_verdict = self._check_section_header_typo(is_heading(stripped), line, lineno)
        if heading_verdict > 0:
          self._check_valid_section_spacing(raw_data[-1][1] if raw_data else None, lineno)
          new_section = self.sections.find(stripped.split(':', maxsplit=1)[0].strip().casefold())
          if new_section != section:
            raw_data     = section.consume(raw_data)
            section      = new_section
            check_indent = section.check_indent_allowed()
            is_heading   = self._get_is_heading(section)
        else:
          heading_verdict = self._check_section_header_that_probably_should_not_be_one(
            heading_verdict, line, lineno
          )
      else:
        # verbatim blocks are never headings
        heading_verdict = Verdict.NOT_HEADING

      raw_data.append((self.make_source_range(line, line, lineno), line, heading_verdict))
      if in_verbatim == 2:
        # reset the dollar verbatim
        in_verbatim = 0

    raw_data = section.consume(raw_data)
    for sec in self.sections:
      sec.setup(self)
    return self

del DocStringType
del DocStringTypeModifier
