#!/usr/bin/env python3
"""
# Created: Mon Jun 20 14:36:59 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
from __future__ import annotations

import os
import re
import sys
import traceback
import subprocess
import ctypes.util
import clang.cindex as clx # type: ignore[import]

from .._typing import *

from ..__version__ import py_version_lt

from ._clang import base_pch_clang_options

def traceback_format_exception(exc: ExceptionKind) -> list[str]:
  r"""Format an exception for printing

  Parameters
  ----------
  exc :
    the exception instance

  Returns
  -------
  ret :
    a list of lines which would have been printed had the exception been re-raised
  """
  if py_version_lt(3, 10):
    etype, value, tb = sys.exc_info()
    ret = traceback.format_exception(etype, value, tb, chain=True)
  else:
    # type checkers do not grok that py_version_lt() means sys.version_info < (3, 10, 0)
    ret = traceback.format_exception(exc, chain=True) # type: ignore [call-arg, arg-type]
  return ret

_T = TypeVar('_T')

def subprocess_check_returncode(ret: subprocess.CompletedProcess[_T]) -> subprocess.CompletedProcess[_T]:
  r"""Check the return code of a subprocess return value

  Paramters
  ---------
  ret :
    the return value of `subprocess.run()`

  Returns
  -------
  ret :
    `ret` unchanged

  Raises
  ------
  RuntimeError
    if `ret.returncode` is nonzero
  """
  try:
    ret.check_returncode()
  except subprocess.CalledProcessError as cpe:
    emess = '\n'.join([
      'Subprocess error:',
      'stderr:',
      f'{cpe.stderr}',
      'stdout:',
      f'{cpe.stdout}',
      f'{cpe}'
    ])
    raise RuntimeError(emess) from cpe
  return ret

def subprocess_capture_output(*args, **kwargs) -> subprocess.CompletedProcess[str]:
  r"""Lightweight wrapper over subprocess.run

  turns a subprocess.CalledProcessError into a RuntimeError with more diagnostics

  Parameters
  ----------
  *args :
    arguments to subprocess.run
  **kwargs :
    keyword arguments to subprocess.run

  Returns
  -------
  ret :
    the return value of `subprocess.run()`

  Raises
  ------
  RuntimeError
    if `subprocess.run()` raises a `subprocess.CalledProcessError`, this routine converts it into a
    RuntimeError with the output attached
  """
  old_check       = kwargs.get('check', True)
  kwargs['check'] = False
  ret             = subprocess.run(*args, capture_output=True, universal_newlines=True, **kwargs)
  if old_check:
    ret = subprocess_check_returncode(ret)
  return ret


def initialize_libclang(clang_dir: Optional[StrPathLike] = None, clang_lib: Optional[StrPathLike] = None, compat_check: bool = True) -> tuple[Optional[StrPathLike], Optional[StrPathLike]]:
  r"""Initialize libclang

  Sets the required library file or directory path to initialize libclang

  Parameters
  ----------
  clang_dir : optional
    the directory containing libclang
  clang_lib : optional
    the direct path to libclang
  compat_check : optional
    perform compatibility checks on loading the dynamic library

  Returns
  -------
  clang_dir, clang_lib : path_like
    the resolved paths if loading occurred, otherwise the arguments unchanged

  Raises
  ------
  ValueError
    if both `clang_dir` and `clang_lib` are None

  Notes
  -----
  If both `clang_lib` and `clang_dir` are given, `clang_lib` takes precedence. `clang_dir` is not
  used in this instance.
  """
  clxconf = clx.conf
  if not clxconf.loaded:
    from ..classes._path import Path

    clxconf.set_compatibility_check(compat_check)
    if clang_lib:
      clang_lib = Path(clang_lib).resolve()
      clxconf.set_library_file(str(clang_lib))
    elif clang_dir:
      clang_dir = Path(clang_dir).resolve()
      clxconf.set_library_path(str(clang_dir))
    else:
      raise ValueError('Must supply either clang directory path or clang library path')
  return clang_dir, clang_lib

def try_to_find_libclang_dir() -> Optional[Path]:
  r"""Crudely tries to find libclang directory.

  First using ctypes.util.find_library(), then llvm-config, and then finally checks a few places on
  macos

  Returns
  -------
  llvm_lib_dir : path_like | None
    the path to libclang (i.e. LLVM_DIR/lib) or None if it was not found
  """
  from ..classes._path import Path

  llvm_lib_dir = ctypes.util.find_library('clang')
  if not llvm_lib_dir:
    try:
      llvm_lib_dir = subprocess_capture_output(['llvm-config', '--libdir']).stdout.strip()
    except FileNotFoundError:
      # FileNotFoundError: [Errno 2] No such file or directory: 'llvm-config'
      # try to find llvm_lib_dir by hand
      import platform

      if platform.system().casefold() == 'darwin':
        try:
          xcode_dir = subprocess_capture_output(['xcode-select', '-p']).stdout.strip()
          if xcode_dir == '/Applications/Xcode.app/Contents/Developer': # default Xcode path
            llvm_lib_dir = os.path.join(
              xcode_dir, 'Toolchains', 'XcodeDefault.xctoolchain', 'usr', 'lib'
            )
          elif xcode_dir == '/Library/Developer/CommandLineTools':      # CLT path
            llvm_lib_dir = os.path.join(xcode_dir, 'usr', 'lib')
        except FileNotFoundError:
          # FileNotFoundError: [Errno 2] No such file or directory: 'xcode-select'
          pass
  if not llvm_lib_dir:
    return None
  return Path(llvm_lib_dir).resolve(strict=True)

def get_petsc_extra_includes(petsc_dir: Path, petsc_arch: str) -> list[str]:
  r"""Retrieve the set of compiler flags to include PETSc libs

  Parameters
  ----------
  petsc_dir : path_like
    the value of PETSC_DIR
  petsc_arch : str
    the value of PETSC_ARCH

  Returns
  -------
  ret : list
    a list containing the flags to add to the compiler flags to pick up PETSc headers and configuration
  """
  # keep these separate, since ORDER MATTERS HERE. Imagine that for example the
  # mpiInclude dir has copies of old PETSc headers, you don't want these to come first
  # in the include search path and hence override those found in petsc/include.

  # You might be thinking that seems suspiciously specific, but I was this close to filing
  # a bug report for python believing that cdll.load() was not deterministic...
  petsc_includes = []
  mpi_includes   = []
  raw_cxx_flags  = []
  with open(petsc_dir/petsc_arch/'lib'/'petsc'/'conf'/'petscvariables', 'r') as pv:
    cc_includes_re  = re.compile(r'^PETSC_CC_INCLUDES\s*=')
    mpi_includes_re = re.compile(r'^MPI_INCLUDE\s*=')
    mpi_show_re     = re.compile(r'^MPICC_SHOW\s*=')
    cxx_flags_re    = re.compile(r'^CXX_FLAGS\s*=')

    def split_and_strip(line: str) -> list[str]:
      return line.split('=', maxsplit=1)[1].split()

    for line in pv:
      if cc_includes_re.search(line):
        petsc_includes.extend(split_and_strip(line))
      elif mpi_includes_re.search(line) or mpi_show_re.search(line):
        mpi_includes.extend(split_and_strip(line))
      elif cxx_flags_re.search(line):
        raw_cxx_flags.extend(split_and_strip(line))

  def filter_flags(flags: list[str], keep_prefix: str) -> Iterable[str]:
    return (flag for flag in flags if flag.startswith(keep_prefix))

  std_flags = list(filter_flags(raw_cxx_flags, '-std='))
  cxx_flags = [std_flags[-1]] if std_flags else [] # take only the last one

  include_gen    = filter_flags(petsc_includes + mpi_includes, '-I')
  seen: set[str] = set()
  seen_add       = seen.add
  extra_includes = [flag for flag in include_gen if not flag in seen and not seen_add(flag)]

  return cxx_flags + extra_includes

def get_clang_sys_includes() -> list[str]:
  r"""Get system clangs set of default include search directories.

  Because for some reason these are hardcoded by the compilers and so libclang does not have them.

  Returns
  -------
  ret :
    list of paths to append to compiler flags to pick up sys inclusions (e.g. <ctypes> or <stdlib.h>)
  """
  from ..classes._path import Path

  output = subprocess_capture_output(['clang', '-E', '-x', 'c++', os.devnull, '-v'])
  # goes to stderr because of /dev/null
  includes = output.stderr.split('#include <...> search starts here:\n')[1]
  includes = includes.split('End of search list.', maxsplit=1)[0].replace('(framework directory)', '')
  return [f'-I{Path(i.strip()).resolve()}' for i in includes.splitlines() if i]

def build_compiler_flags(petsc_dir: Path, petsc_arch: str, extra_compiler_flags: Optional[list[str]] = None, verbose: int = 0) -> list[str]:
  r"""Build the baseline set of compiler flags.

  These are passed to all translation unit parse attempts.

  Parameters
  ----------
  petsc_dir : path_like | str
    the value of PETSC_DIR
  petsc_arch : str
    the value of PETSC_ARCH
  extra_compiler_flags : list[str] | None, optional
    extra compiler flags, if None, an empty list is used
  verbose : False, optional
    print verbose output (at level)

  Returns
  -------
  compiler_flags : list[str]
    the full list of compiler flags to pass to the parsers
  """
  if extra_compiler_flags is None:
    extra_compiler_flags = []

  misc_flags = [
    '-DPETSC_CLANG_STATIC_ANALYZER',
    '-xc++',
    '-Wno-empty-body',
    '-Wno-writable-strings',
    '-Wno-array-bounds',
    '-Wno-nullability-completeness',
    '-fparse-all-comments',
    '-g'
  ]
  petsc_includes = get_petsc_extra_includes(petsc_dir, petsc_arch)
  compiler_flags = get_clang_sys_includes() + misc_flags + petsc_includes + extra_compiler_flags
  if verbose > 1:
    import petsclinter as pl

    pl.sync_print('\n'.join(['Compile flags:', *compiler_flags]))
  return compiler_flags

class PrecompiledHeader:
  __slots__ = 'pch', 'verbose'

  pch: PathLike
  verbose: int

  def __init__(self, pch: PathLike, verbose: int) -> None:
    r"""Construct the PrecompiledHeader

    Parameters
    ----------
    pch :
      the path where the precompiled header should be stored
    verbose :
      print verbose information (at level)
    """
    self.pch     = pch
    self.verbose = verbose
    return

  def __enter__(self) -> PrecompiledHeader:
    return self

  def __exit__(self, *args, **kwargs) -> None:
    if self.verbose:
      import petsclinter as pl

      pl.sync_print('Deleting precompiled header', self.pch)
      self.pch.unlink()
    return

  @classmethod
  def from_flags(cls, petsc_dir: Path, compiler_flags: list[str], extra_header_includes: Optional[list[str]] = None, verbose: int = 0, pch_clang_options: Optional[CXTranslationUnit] = None) -> PrecompiledHeader:
    r"""Create a precompiled header from flags.

    This builds the precompiled head from petsc.h, and all of the private headers. This not only saves
    a lot of time, but is critical to finding struct definitions. Header contents are not parsed
    during the actual linting, since this balloons the parsing time as libclang provides no builtin
    auto header-precompilation like the normal compiler does.

    Including petsc.h first should define almost everything we need so no side effects from including
    headers in the wrong order below.

    Parameters
    ----------
    petsc_dir : path_like | str
      the value of PETSC_DIR
    compiler_flags : list[str]
      the list of compiler flags to parse with
    extra_header_includes : list[str], optional
      extra header include directives to add
    verbose : False, optional
      print verbose information
    pch_clang_options : iterable(int)
      clang parsing options to use, if not set, petsclinter.util.base_pch_clang_options are used

    Returns
    -------
    ret : PrecompiledHeader
      the precompiled header object

    Raises
    ------
    clang.cindex.LibclangError
      if `extra_header_includes` is not None, and the compilation results in compiler diagnostics
    """
    import petsclinter as pl

    def verbose_print(*args, **kwargs) -> None:
      if verbose > 1:
        pl.sync_print(*args, **kwargs)
      return

    assert isinstance(petsc_dir, pl.Path)
    if pch_clang_options is None:
      pch_clang_options = base_pch_clang_options

    if extra_header_includes is None:
      extra_header_includes = []

    index              = clx.Index.create()
    precompiled_header = petsc_dir/'include'/'petsc_ast_precompile.pch'
    mega_header_lines  = [
      # Kokkos needs to go first since it mucks with complex
      ('petscvec_kokkos.hpp', '#include <petscvec_kokkos.hpp>'),
      ('petsc.h',             '#include <petsc.h>')
    ]
    private_dir_name = petsc_dir/'include'/'petsc'/'private'
    mega_header_name = 'mega_header.hpp'

    # build a megaheader from every header in private first
    for header in private_dir_name.iterdir():
      if header.suffix in ('.h', '.hpp'):
        header_name = header.name
        mega_header_lines.append((header_name, f'#include <petsc/private/{header_name}>'))

    # loop until we get a completely clean compilation, any problematic headers are discarded
    while True:
      mega_header = '\n'.join(hfi for _,hfi in mega_header_lines)+'\n'  # extra newline for last line
      tu = index.parse(
        mega_header_name,
        args=compiler_flags, unsaved_files=[(mega_header_name, mega_header)], options=pch_clang_options
      )
      diags = {}
      for diag in tu.diagnostics:
        try:
          filename = diag.location.file.name
        except AttributeError:
          # file is None
          continue
        basename, filename = os.path.split(filename)
        if filename not in diags:
          # save the problematic header name as well as its path (a surprise tool that will
          # help us later)
          diags[filename] = (basename,diag)
      for dirname, diag in tuple(diags.values()):
        # the reason this is done twice is because as usual libclang hides
        # everything in children. Suppose you have a primary header A (which might be
        # include/petsc/private/headerA.h), header B and header C. Header B and C are in
        # unknown locations and all we know is that Header A includes B which includes C.
        #
        # Now suppose header C is missing, meaning that Header A needs to be removed.
        # libclang isn't gonna tell you that without some elbow grease since that would be
        # far too easy. Instead it raises the error about header B, so we need to link it
        # back to header A.
        if dirname != private_dir_name:
          # problematic header is NOT in include/petsc/private, so we have a header B on our
          # hands
          for child in diag.children:
            # child of header B here is header A not header C
            try:
              filename = child.location.file.name
            except AttributeError:
              # file is None
              continue
            # filter out our fake header
            if filename != mega_header_name:
              # this will be include/petsc/private, headerA.h
              basename, filename = os.path.split(filename)
              if filename not in diags:
                diags[filename] = (basename, diag)
      if diags:
        diagerrs = '\n'+'\n'.join(str(d) for _, d in diags.values())
        verbose_print('Included header has errors, removing', diagerrs)
        mega_header_lines = [(hdr, hfi) for hdr, hfi in mega_header_lines if hdr not in diags]
      else:
        break
    if extra_header_includes:
      # now include the other headers but this time immediately crash on errors, let the
      # user figure out their own busted header files
      mega_header += '\n'.join(extra_header_includes)
      verbose_print(f'Mega header:\n{mega_header}')
      tu = index.parse(
        mega_header_name,
        args=compiler_flags, unsaved_files=[(mega_header_name, mega_header)], options=pch_clang_options
      )
      if tu.diagnostics:
        pl.sync_print('\n'.join(map(str, tu.diagnostics)))
        raise clx.LibclangError('\n\nWarnings or errors generated when creating the precompiled header. This usually means that the provided libclang setup is faulty. If you used the auto-detection mechanism to find libclang then perhaps try specifying the location directly.')
    else:
      verbose_print(f'Mega header:\n{mega_header}')
    precompiled_header.unlink(missing_ok=True)
    tu.save(precompiled_header)
    compiler_flags.extend(['-include-pch', str(precompiled_header)])
    verbose_print('Saving precompiled header', precompiled_header)
    return cls(precompiled_header, verbose)
