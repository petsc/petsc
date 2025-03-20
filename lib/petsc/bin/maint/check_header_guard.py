#!/usr/bin/env python3
"""
# Created: Thu Aug 17 14:06:52 2023 (-0500)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import re
import os
import sys
import abc
import difflib
import pathlib
import argparse

from typing          import TypeVar, Union
from collections.abc import Iterable, Sequence

__version__     = (1, 0, 0)
__version_str__ = '.'.join(map(str, __version__))

class Replacer(abc.ABC):
  __slots__ = 'verbose', 'added', 'path'

  verbose: bool
  added: bool
  path: pathlib.Path

  def __init__(self, verbose: bool, path: pathlib.Path) -> None:
    self.verbose = verbose
    self.added   = False
    self.path    = path
    return

  def _strip_empty_lines(self, idx: int, ret: list[str]) -> list[str]:
    r"""Strip empty lines from a list of lines at index `idx`

    Parameters
    ----------
    idx :
      the index to remove at
    ret :
      the list of lines

    Returns
    -------
    ret :
      the lines with empty lines removed at `idx`
    """
    while ret and not ret[idx].strip():
      entry = ret.pop(idx)
      if self.verbose:
        print(f'- {entry}')
    return ret

  @abc.abstractmethod
  def prologue(self, ret: Sequence[str]) -> list[str]:
    r"""Common prologue for replacement, strips any leading blank lines

    Parameters
    ----------
    ret :
      the list of lines for the file

    Returns
    -------
    ret :
      the list of lines with leading blank spaces removed
    """
    return self._strip_empty_lines(0, ret)

  @abc.abstractmethod
  def replace(self, last_line: str, line: str, ret: list[str]) -> list[str]:
    return ret

  @abc.abstractmethod
  def epilogue(self, last_endif: int, ret: list[str]) -> list[str]:
    r"""Common epilogue for replacements, strips any trailing blank lines from the header

    Parameters
    ----------
    last_endif :
      unused
    ret :
      the list of lines

    Returns
    -------
    ret :
      the lines with trailing blank lines pruned
    """
    return self._strip_empty_lines(-1, ret)

class PragmaOnce(Replacer):
  def prologue(self, ret: list[str]) -> list[str]:
    return super().prologue(ret)

  def replace(self, prev_line: str, line: str, ret: list[str]) -> list[str]:
    r"""Replace the selected header-guard line with #pragma once

    Parameters
    ----------
    prev_line :
      the previous line
    line :
      the current line
    ret :
      the list previously seen lines to append to

    Returns
    -------
    ret :
      the list of lines with the new header guard inserted

    Notes
    -----
    This routine is idempotent, i.e. does nothing if it already added it
    """
    ret = super().replace(prev_line, line, ret)
    if self.added:
      return ret

    pragma_once = '#pragma once'
    if line.startswith(pragma_once):
      # nothing to do, just add the pragma once line back in
      ret.append(line)
      return ret

    assert prev_line.startswith('#ifndef')
    # header-guard to pragma once conversion
    if self.verbose:
      print(f'{self.path}:')
      print(f'- {prev_line.lstrip()}')
      print(f'- {line.lstrip()}')
      print(f'+ {pragma_once}')

    ret[-1]    = pragma_once
    self.added = True
    return ret

  def epilogue(self, last_endif: int, ret: list[str]) -> list[str]:
    r"""Final function to call after performing replacement

    Parameters
    ----------
    last_endif :
      the index into `ret` containing the last `#endif` line
    ret :
      the list of lines for the file

    Returns
    -------
    ret :
      If the header guard was replaced with #pragma once, `ret` with the final `#endif` removed,
      otherwise ret unchanged
    """
    if self.added:
      endif_line = ret.pop(last_endif - 1)
      if self.verbose:
        print(f'- {endif_line}')
      # # prune empty lines as a result of deleting the header guard
      # while not ret[-1].strip():
      #   end = ret.pop()
      #   if self.verbose:
      #     print(f'- {end}')
    ret = super().epilogue(last_endif, ret)
    return ret

class VerboseHeaderGuard(Replacer):
  __slots__ = 'new_ifndef', 'new_guard', 'new_endif', 'append_endif'

  new_ifndef: str
  new_guard: str
  new_endif: str
  append_endif: bool

  def __init__(self, *args, **kwargs) -> None:
    r"""Construct a `VerboseHeaderGuard`

    Parameters
    ----------
    *args :
      positional arguments to forward to `Replacer` constructor
    **kwargs :
      keyword arguments to forward to `Replacer` constructor
    """
    super().__init__(*args, **kwargs)
    str_path          = str(self.path).casefold()
    guard_str         = str_path[max(str_path.find('petsc'), 0):]
    guard_str         = ''.join('_' if c in {'/', '.', '-', ' '} else c for c in guard_str)
    self.new_ifndef   = f'#ifndef {guard_str}'
    self.new_guard    = f'#define {guard_str}'
    self.new_endif    = f'#endif // {guard_str}'
    self.append_endif = False
    return

  def prologue(self, ret: list[str]) -> list[str]:
    return super().prologue(ret)

  def replace(self, prev_line: str, line: str, ret: list[str]) -> list[str]:
    r"""Replace the selected header-guard line with a verbose header guard

    Parameters
    ----------
    prev_line :
      the previous line
    line :
      the current line
    ret :
      the list previously seen lines to append to

    Returns
    -------
    ret :
      the list of lines with the new header guard inserted

    Raises
    ------
    ValueError
      if the line to convert is neither a header-gaurd or #pragma once line

    Notes
    -----
    This routine is idempotent, i.e. does nothing if it already added it
    """
    ret = super().replace(prev_line, line, ret)
    if self.added:
      return ret

    self.added = True
    if prev_line == self.new_ifndef and line == self.new_guard:
      # nothing to do, add the line back in
      ret.append(line)
      return ret

    if self.verbose:
      print(f'{self.path}:')

    if prev_line.startswith('#ifndef'):
      # header-guard to header-guard conversion
      if self.verbose:
        print(f'- {prev_line.lstrip()}')
        print(f'- {line.lstrip()}')

      ret[-1] = self.new_ifndef
      ret.append(self.new_guard)
    elif line.startswith('#pragma once'):
      # pragma once to header-guard conversion
      if self.verbose:
        print(f'- {line.lstrip()}')
      self.append_endif = True
      ret.extend([
        self.new_ifndef,
        self.new_guard
      ])
    else:
      raise ValueError(
        f'Line to convert must be either a header-guard or #pragma once, found neither: {line}'
      )

    if self.verbose:
      print(f'+ {self.new_ifndef}')
      print(f'+ {self.new_guard}')
    return ret

  def epilogue(self, last_endif: int, ret: list[str]) -> list[str]:
    r"""Final function to call after replacements

    Parameters
    ----------
    last_endif :
      the index into `ret` containing the last `#endif` line
    ret :
      the list of lines for the file

    Returns
    -------
    ret :
      `ret` either with an append `#endif` (if converting from `#pragma once` to header-guard) or
       unchanged
    """
    if self.append_endif:
      ret.append(self.new_endif)
      if self.verbose:
        print(f'+ {ret[-1]}')
    elif (old := ret[last_endif].lstrip()) != self.new_endif:
      ret[last_endif] = self.new_endif
      if self.verbose:
        print(f'- {old}')
        print(f'+ {ret[last_endif]}')
    ret = super().epilogue(last_endif, ret)
    return ret

_T = TypeVar('_T', bound=Replacer)

def do_replacement(replacer: _T, lines: Iterable[str]) -> list[str]:
  r"""Replace the header guard using the replacement class

  Parameters
  ----------
  replacer :
    an instance of a concrete replacement class
  lines :
    an iterable of lines of the file

  Returns
  -------
  ret :
    the file lines with the replaced header guard, if applicable
  """
  header_re = re.compile(r'#ifndef\s+(.*)')
  define_re = re.compile(r'#define\s+(.*)')

  def is_pragma_once(line: str) -> bool:
    return line.startswith('#pragma once')

  def is_header_guard(prev_line: str, line: str) -> bool:
    d_match = define_re.match(line)
    h_match = header_re.match(prev_line)
    return d_match is not None and h_match is not None and d_match.group(1) == h_match.group(1)

  def is_match(prev_line: str, line: str) -> bool:
    return is_pragma_once(line) or is_header_guard(prev_line, line)

  ret: list[str] = []
  last_endif     = 0

  lines = replacer.prologue(list(lines))
  for i, line in enumerate(lines):
    try:
      prev_line = ret[-1]
    except IndexError:
      prev_line = ''

    if is_match(prev_line, line):
      ret = replacer.replace(prev_line, line, ret)
    else:
      if line.startswith('#endif'):
        last_endif = i
      ret.append(line)

  ret = replacer.epilogue(last_endif, ret)
  return ret

def replace_in_file(path: pathlib.Path, opts: argparse.Namespace, ReplacerCls: type[_T]) -> list[str]:
  r"""Replace the header guards in a file

  Parameters
  ----------
  path :
    the path to check
  opts :
    the options database to use
  replacer :
    the replacement class type to use to make the replacements

  Notes
  -----
  Does nothing if the file isn't a header
  """
  error_diffs: list[str] = []

  if not path.name.endswith(opts.suffixes):
    return error_diffs

  if opts.verbose:
    print('Reading', path)

  lines = path.read_text().splitlines()
  repl  = ReplacerCls(opts.verbose, path)
  ret   = do_replacement(repl, lines)

  if opts.action == 'convert':
    if not opts.dry_run:
      path.write_text('\n'.join(ret) + '\n')
  elif opts.action == 'check':
    if diffs := list(
        difflib.unified_diff(lines, ret, fromfile='actual', tofile='expected', lineterm='')
    ):
      err_bars = '=' * 95
      error_diffs.extend([
        err_bars,
        'ERROR: Malformed header guard!',
        f'ERROR: {path}',
        *diffs,
        err_bars
      ])
  return error_diffs

def main(args: argparse.Namespace) -> int:
  r"""Perform header guard replacement

  Parameters
  ----------
  args :
    the collected configurations arguments

  Returns
  -------
  ret :
    a return-code indicating status, 0 for success and nonzero otherwise

  Raises
  ------
  ValueError
    if `args.kind` is unknown, or `args.action` is unknown
  """
  if args.action not in {'check', 'convert'}:
    raise ValueError(f'Unknown action {args.action}')

  if args.kind == 'verbose_header_guard':
    replacer_cls = VerboseHeaderGuard
  elif args.kind == 'pragma_once':
    replacer_cls = PragmaOnce
  else:
    raise ValueError(f'Unknown replacer kind: {args.kind}')

  args.suffixes     = tuple(args.suffixes)
  exclude_dirs      = set(args.exclude_dirs)
  exclude_files     = set(args.exclude_files)
  errors: list[str] = []
  for path in args.paths:
    path = path.resolve(strict=True)
    if exclude_dirs.intersection(path.parts) or path.name in exclude_files:
      # the path itself is in an excluded path
      continue

    if path.is_file():
      errors.extend(replace_in_file(path, args, replacer_cls))
    else:
      for dirname, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        dirpath = pathlib.Path(dirname)
        for f in files:
          if f not in exclude_files:
            errors.extend(replace_in_file(dirpath / f, args, replacer_cls))

  if errors:
    print(*errors, sep='\n')
    return 1
  return 0

def command_line_main() -> int:
  def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
      return v
    v = v.casefold()
    if v in {'yes', 'true', 't', 'y', '1'}:
      return True
    if v in {'no', 'false', 'f', 'n', '0', ''}:
      return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got \'{v}\'')

  parser = argparse.ArgumentParser(
    'header guard conversion tool',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('paths', nargs='+', type=pathlib.Path, help='paths to check/convert')
  parser.add_argument(
    '--verbose',
    nargs='?', const=True, default=False, metavar='bool', type=str2bool, help='verbose output'
  )
  parser.add_argument(
    '--dry-run',
    nargs='?', const=True, default=False, metavar='bool', type=str2bool,
    help='don\'t actually write results to file, only useful when replacing'
  )
  parser.add_argument(
    '--kind', required=True, choices=('verbose_header_guard', 'pragma_once'),
    help='Determine the kind of header guard to enforce'
  )
  parser.add_argument(
    '--action', required=True, choices=('convert', 'check'),
    help='whether to replace or check the header guards'
  )
  parser.add_argument(
    '--suffixes', nargs='+', default=['.h', '.hpp', '.cuh', '.inl', '.H', '.hh'],
    help='set file suffixes to check, must contain \'.\', e.g. \'.h\''
  )
  parser.add_argument(
    '--exclude-dirs', nargs='+',
    default=[
      'binding', 'finclude', 'ftn-mod', 'ftn-auto', 'contrib', 'perfstubs', 'yaml', 'fsrc',
      'benchmarks', 'valgrind', 'khash', 'mpiuni'
    ],
    help=f'set directory names to exclude, must not contain \'{os.path.sep}\''
  )
  parser.add_argument(
    '--exclude-files', nargs='+', default=['petscversion.h', 'slepcversion.h'],
    help=f'set file names to exclude, must not contain \'{os.path.sep}\''
  )
  parser.add_argument('--version', action='version', version=f'%(prog)s v{__version_str__}')

  args = parser.parse_args()
  ret  = main(args)
  if ret:
    err_bar = 'x' + 93 * '*' + 'x'
    print(err_bar)
    print('run the following to automatically fix your errors:')
    print('')
    print(' '.join('--action=convert' if a.startswith('--action') else a for a in sys.argv))
    print(err_bar)
    # to ensure it prints everything when running in CI
    sys.stdout.flush()
  return ret

if __name__ == '__main__':
  ret = command_line_main()
  sys.exit(ret)
