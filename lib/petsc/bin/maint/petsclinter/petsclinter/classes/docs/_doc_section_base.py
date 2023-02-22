#!/usr/bin/env python3
"""
# Created: Sun Nov 20 12:27:36 2022 (-0500)
# @author: Jacob Faibussowitsch
"""
import difflib
import textwrap
import itertools
import collections
import petsclinter as pl

from .._diag    import DiagnosticManager, Diagnostic
from .._src_pos import SourceRange
from .._patch   import Patch

"""
==========================================================================================
Base Classes
==========================================================================================
"""
class DescribableItem:
  __slots__ = 'text', 'prefix', 'arg', 'description', 'sep'

  def __init__(self, raw, prefixes=None, sep='-'):
    split_param = self.split_param
    text        = raw.strip()

    prefix, arg, descr = split_param(text, prefixes, sep)
    if not descr:
      found = False
      for sep in (',', '='):
        _, arg, descr = split_param(text, prefixes, sep)
        if descr:
          found = True
          break
      if not found:
        sep = ' '
        if prefix:
          arg = text.split(prefix, maxsplit=1)[1].strip()
        else:
          arg, *descr = text.split(maxsplit=1)
          if isinstance(descr, list):
            descr = descr[0] if len(descr) else ''
          assert isinstance(descr, str)
    self.text        = raw
    self.prefix      = prefix
    self.sep         = sep
    self.arg         = arg
    self.description = descr
    return

  @classmethod
  def cast(cls, other, **kwargs):
    if isinstance(other, cls):
      return other
    return cls(other, **kwargs)

  @staticmethod
  def split_param(text, prefixes, char):
    r"""
    retrieve groups '([\.+-$])\s*([A-z,-]+) - (.*)'
    """
    stripped = text.strip()
    if prefixes is None:
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
    arg, sep, descr = rest.partition(char.join((' ', ' ')))
    if not sep:
      if rest.endswith(char):
        arg = rest[:-1]
      elif char + ' ' in rest:
        arg, _, descr = rest.partition(char + ' ')
      # if we hit neither then there is no '-' in text, possible case of '[prefix] foo'?
    return prefix, arg.strip(), descr.lstrip()

  def arglen(self):
    """
    return a length l such that text[:l] returns all text up until the end of the arg name
    """
    arg = self.arg
    return self.text.find(arg) + len(arg)

  def check(self, docstring, section, loc, expected_sep=None):
    if expected_sep is None:
      expected_sep = self.sep

    name = section.transform(section.name)
    if self.sep != expected_sep:
      diag  = section.diags.wrong_description_separator
      mess  = f"{name} seems to be missing a description separator; I suspect you may be using '{self.sep}' as a separator instead of '{expected_sep}'. Expected '{self.arg} {expected_sep} {self.description}'"
    elif not self.description:
      diag = section.diags.missing_description
      mess = f"{name} missing a description. Expected '{self.arg} {expected_sep} a very useful description'"
    else:
      return # ok?
    docstring.add_error_from_source_range(diag, mess, loc)
    return

class DocBase:
  __slots__ = tuple()

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return cls.diagnostic_flag('-'.join(flags))

  @classmethod
  def diagnostic_flag(cls, text, *, prefix='doc'):
    if isinstance(text, str):
      return collections.deque((prefix, text) if prefix == 'doc' else ('doc', prefix, text))
    if not isinstance(text, collections.deque):
      text = collections.deque(text)
    if not text[0].startswith(prefix):
      text.appendleft(prefix)
    return text

@DiagnosticManager.register(
  ('section-header-missing', 'Verify that required sections exist in the docstring'),
  ('section-header-unique', 'Verify that appropriate sections are unique per docstring'),
  ('section-barren', 'Verify there are no sections containing a title and nothing else'),
  ('section-header-solitary', 'Verify that qualifying section headers are alone on their line'),
  ('section-header-spelling', 'Verify section headers are correctly spelled'),
)
class SectionBase(DocBase):
  """
  Container for a single section of the docstring, has members:

  'name'     - the name of this section
  'required' - is this section required in the docstring
  'titles'   - header-titles, i.e. "Input Parameter", or "Level", must be spelled correctly
  'keywords' - keywords to help match an unknown title to a section
  'raw'      - the raw text in the section
  'extent'   - the SourceRange for the whole section
  'lines'    - a tuple of each line of text and its SourceRange in the section
  'items'    - a tuple of extracted tokens of interest, e.g. the level value, options parameters,
               function parameters, etc.
  """
  __slots__ = (
    'name', 'required', 'titles', 'keywords', 'raw', 'extent', '_lines', 'items', 'seen_headers',
    'solitary'
  )

  def __init__(self, name, required=False, keywords=None, titles=None, solitary=True):
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

  def __str__(self):
    return '\n'.join([
      f'Type:   {type(self)}',
      f'Name:   {self.name}',
      f'Extent: {self.extent}'
    ])

  def __bool__(self):
    return bool(self.lines())

  def clear(self):
    self.raw          = ''
    self.extent       = None
    self._lines       = []
    self.items        = None
    self.seen_headers = {}
    return

  def lines(self, headings_only=False):
    if headings_only:
      return [(loc, line, verdict) for loc, line, verdict in self._lines if verdict > 0]
    return self._lines

  def consume(self, data):
    data = list(data)
    if data:
      self.lines().extend(data)
      self.raw    = '\n'.join(s for _, s, _ in self.lines())
      self.extent = SourceRange.from_locations(self.lines()[0][0].start, self.lines()[-1][0].end)
    return []

  def setup(self, docstring, inspect_line=None):
    inspect = inspect_line is not None
    seen    = collections.defaultdict(list)

    for loc, line, verdict in self.lines():
      if verdict > 0:
        possible_header = line.split(':' if ':' in line else None, maxsplit=1)[0].strip()
        seen[possible_header.casefold()].append(
          docstring.make_source_range(possible_header, line, loc.start.line)
        )
      if inspect:
        # let each section type determine if this line is useful
        inspect_line(loc, line, verdict)

    self.seen_headers = dict(seen)
    return

  def barren(self):
    lines = self.lines()
    return not self.items and sum(not line.strip() for _, line, _ in lines) == len(lines) - 1

  @staticmethod
  def transform(text):
    return text.title()

  @staticmethod
  def check_indent_allowed():
    return True

  def _check_required_section_found(self, docstring):
    if not self and self.required:
      diag = self.diags.section_header_missing
      mess = f'Required section \'{self.titles[0]}\' not found'
      docstring.add_error_from_source_range(diag, mess, docstring.extent, highlight=False)
    return

  def _check_section_is_not_barren(self, docstring):
    """
    check that a section isn't just a solitary header out on its own
    """
    if self and self.barren():
      diag      = self.diags.section_barren
      highlight = len(self.lines()) == 1
      mess      = 'Section appears to be empty; while I\'m all for a good mystery, you should probably elaborate here'
      docstring.add_error_from_source_range(diag, mess, self.extent, highlight=highlight)
    return

  def _check_section_header_spelling(self, linter, docstring, headings=None, transform=None):
    """
    Check that a section header is correctly spelled and formatted. Sections may be found
    through fuzzy matching so this check asserts that a particular heading is actually correct
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
        before, _ = docstring.guess_heading(text)

      heading = before.strip()
      if any(t in heading for t in self.titles):
        continue

      heading_loc = docstring.make_source_range(heading, text, loc.start.line)
      correct     = transform(heading)
      if heading != correct and any(t in correct for t in self.titles):
        docstring.add_error_from_source_range(
          diag, f'Invalid header spelling. Expected \'{correct}\' found \'{heading}\'',
          heading_loc, patch=Patch(heading_loc, correct)
        )
        continue

      try:
        matchname = difflib.get_close_matches(correct, self.titles, n=1)[0]
      except IndexError:
        linter.add_warning_from_cursor(
          docstring.cursor, Diagnostic(diag, f'Unknown section \'{heading}\'', self.extent.start)
        )
      else:
        docstring.add_error_from_source_range(
          diag,
          f'Unknown section header \'{heading}\', assuming you meant \'{matchname}\'',
          heading_loc, patch=Patch(heading_loc, matchname)
        )
    return

  def _check_duplicate_headers(self, docstring):
    """
    Check that a particular heading is not repeated within the docstring
    """
    for heading, where in self.seen_headers.items():
      if len(where) <= 1:
        continue

      lasti    = len(where) - 1
      src_list = []
      nbefore  = 2
      nafter   = 0
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
      docstring.add_error_from_diagnostic(
        Diagnostic(self.diags.section_header_unique, mess, self.extent.start)
      )
    return

  def _check_section_header_solitary(self, docstring, headings=None):
    """
    Check that a section appears solitarily on its line, i.e. that there is no other text after
    ':'
    """
    if not self.solitary:
      return

    if headings is None:
      headings = self.lines(headings_only=True)

    for loc, text, verdict in headings:
      _, sep, after = text.partition(':')
      if not sep:
        head, _       = docstring.guess_heading(text)
        _, sep, after = text.partition(head)
        assert sep
      if after.strip():
        diag = self.diags.section_header_solitary
        mess = 'Heading must appear alone on a line, any content must be on the next line'
        docstring.add_error_from_source_range(
          diag, mess, docstring.make_source_range(after, text, loc.start.line)
        )
      break
    return

  def check(self, linter, cursor, docstring):
    self._check_required_section_found(docstring)
    self._check_section_header_spelling(linter, docstring)
    self._check_section_is_not_barren(docstring)
    self._check_duplicate_headers(docstring)
    self._check_section_header_solitary(docstring)
    return

@DiagnosticManager.register(
  ('alignment', 'Verify that parameter list entries are correctly white-space aligned'),
  ('prefix', 'Verify that parameter list entries begin with the correct prefix'),
  ('missing-description', 'Verify that parameter list entries have a description'),
  ('missing-description-separator', 'Verify that a parameter list entry has a separator before the description'),
  ('wrong-description-separator', 'Verify that parameter list entries use the right description separator'),
  ('solitary-parameter', 'Verify that each parameter has its own entry'),
)
class ParameterList(SectionBase):
  __slots__ = ('prefixes', )

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('param-list', *flags)

  def __init__(self, *args, prefixes=None, **kwargs):
    if prefixes is None:
      prefixes = ('+', '.', '-')

    self.prefixes = prefixes
    kwargs.setdefault('name', 'parameters')
    super().__init__(*args, **kwargs)
    return

  def check_indent_allowed(self):
    return False

  def check_aligned_descriptions(self, ds, group):
    """
    Verify that the position of the '-' before the description for each argument is aligned
    """
    align_diag  = self.diags.alignment
    group_args  = [item.arg for _, item, _ in group]
    lens        = list(map(len, group_args))
    max_arg_len = max(lens) if lens else 0
    assert max_arg_len >= 0, f'Negative maximum argument length {max_arg_len}'
    longest_arg = group_args[lens.index(max_arg_len)] if lens else 'NO ARGS'

    for loc, item, _ in group:
      pre   = item.prefix
      arg   = item.arg
      descr = item.description
      text  = item.text
      fixed = '{} {:{width}} - {}'.format(pre, arg, descr, width=max_arg_len)
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
      ds.add_error_from_source_range(align_diag, mess, eloc, patch=Patch(eloc, fixed[diff_index:]))
    return

  def setup(self, ds, *args, parameter_list_prefix_check=None, **kwargs):
    subheading = 0
    groups     = collections.defaultdict(list)

    def inspector(loc, line, verdict, *args, **kwargs):
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

    super().setup(ds, *args, inspect_line=inspector, **kwargs)
    items = dict(groups)
    if parameter_list_prefix_check is not None:
      items = parameter_list_prefix_check(self, ds, items)
    self.items = items
    return

  def _check_opt_starts_with(self, docstring, item, descr, char):
    loc, descr_item, _ = item
    pre                = descr_item.prefix
    if pre != char:
      eloc = docstring.make_source_range(pre, descr_item.text, loc.start.line)
      mess = f'{descr} parameter list entry must start with \'{char}\''
      docstring.add_error_from_source_range(self.diags.prefix, mess, eloc, patch=Patch(eloc, char))
    return

  def _check_prefixes(self, docstring):
    for key, opts in sorted(self.items.items()):
      lopts = len(opts)
      if lopts < 1:
        raise RuntimeError(f'number of options {lopts} < 1, key: {key}, items: {items}')

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

  def check(self, linter, cursor, docstring):
    super().check(linter, cursor, docstring)
    self._check_prefixes(docstring)
    return

class Prose(SectionBase):
  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('prose', *flags)

  def setup(self, ds, *args, **kwargs):
    subheading = 0
    items      = {}

    def inspector(loc, line, verdict, *args, **kwargs):
      if verdict > 0:
        head, _, rest = line.partition(':')
        head          = head.strip()
        assert head, f'No heading in PROSE section?\n\n{loc.formatted(num_context=5)}'
        if items.keys():
          nonlocal subheading
          subheading += 1
        start_line = loc.start.line
        items[subheading] = (
          (ds.make_source_range(head, line, start_line), head),
          [(ds.make_source_range(rest, line, start_line), rest)] if rest else []
        )
      elif line.strip():
        try:
          items[subheading][1].append((loc, line))
        except KeyError as ke:
          raise pl.ParsingError from ke
      return

    super().setup(ds, *args, inspect_line=inspector, **kwargs)
    self.items = items
    return

class VerbatimBlock(SectionBase):
  def setup(self, ds, *args, **kwargs):
    items = {}

    class Inspector:
      __slots__ = 'codeblocks', 'startline'

      def __init__(self, obj):
        self.codeblocks = 0
        self.startline  = obj.extent.start.line if obj else 0
        return

      def __call__(self, loc, line, *args, **kwargs):
        sub   = self.codeblocks
        lstrp = line.lstrip()
        if lstrp.startswith('.vb'):
          items[sub] = [loc.start.line - self.startline]
        elif lstrp.startswith('.ve'):
          assert len(items[sub]) == 1
          items[sub].append(loc.start.line - self.startline + 1)
          self.codeblocks += 1
        return

    super().setup(ds, *args, **kwargs, inspect_line=Inspector(self))
    self.items = items
    return

@DiagnosticManager.register(
  ('formatting', 'Verify that inline lists are correctly white-space formatted')
)
class InlineList(SectionBase):
  __slots__ = ('special_chars',)

  def __init__(self, *args, special_chars='', **kwargs):
    kwargs.setdefault('solitary', False)
    super().__init__(*args, **kwargs)
    self.special_chars = special_chars
    return

  @classmethod
  def __diagnostic_prefix__(cls, *flags):
    return DiagnosticManager.flag_prefix(super())('inline-list', *flags)

  @classmethod
  def check_indent_allowed(cls):
    return False

  def setup(self, ds, *args, **kwargs):
    items  = []
    titles = set(map(str.casefold, self.titles))

    def inspector(loc, line, *args, **kwargs):
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

    super().setup(ds, *args, inspect_line=inspector, **kwargs)
    self.items = tuple(items)
    return

  def _check_whitespace_formatting(self, docstring):
    """
    Ensure that inline list ensures are on the same line and 1 space away from the title
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
      docstring.add_error_from_source_range(
        format_diag, base_mess.format(nspaces + 1), floc, patch=Patch(floc, fix)
      )
    return

  def check(self, linter, cursor, docstring):
    super().check(linter, cursor, docstring)
    self._check_whitespace_formatting(docstring)
    return
