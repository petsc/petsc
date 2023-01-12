#!/usr/bin/env python3
"""
# Created: Mon Jun 20 14:36:59 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import os
import re
import subprocess
import ctypes.util
import clang.cindex as clx
import petsclinter  as pl

from .. import __version__

# synchronized print function, should be used everywhere
sync_print = print

def set_sync_print(print_fn=None):
  if print_fn is None:
    print_fn = print
  global sync_print
  old_sync_print = sync_print
  sync_print     = print_fn
  return old_sync_print

def subprocess_run(*args, **kwargs):
  """
  lightweight wrapper to hoist the ugly version check out of the regular code, turns a
  subprocess.CalledProcessError into a RuntimeError with more diagnostics
  """
  if __version__.py_version_lt(3, 7):
    if kwargs.pop('capture_output', None):
      kwargs.setdefault('stdout', subprocess.PIPE)
      kwargs.setdefault('stderr', subprocess.PIPE)
  try:
    output = subprocess.run(*args, **kwargs)
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
  return output

def initialize_libclang(clang_dir=None, clang_lib=None):
  """
  Set the required library file or directory path to initialize libclang
  """
  clxconf = clx.conf
  if not clxconf.loaded:
    clxconf.set_compatibility_check(True)
    if clang_lib:
      clang_lib = pl.Path(clang_lib).resolve()
      clxconf.set_library_file(str(clang_lib))
    elif clang_dir:
      clang_dir = pl.Path(clang_dir).resolve()
      clxconf.set_library_path(str(clang_dir))
    else:
      raise RuntimeError('Must supply either clang directory path or clang library path')
  return clang_dir, clang_lib

def try_to_find_libclang_dir():
  """
  Crudely tries to find libclang directory first using ctypes.util.find_library(), then llvm-config,
  and then finally checks a few places on macos
  """
  llvm_lib_dir = ctypes.util.find_library('clang')
  if not llvm_lib_dir:
    try:
      output = subprocess_run(
        ['llvm-config', '--libdir'], capture_output=True, universal_newlines=True, check=True
      )
      llvm_lib_dir = output.stdout.strip()
    except FileNotFoundError:
      # FileNotFoundError: [Errno 2] No such file or directory: 'llvm-config'
      # try to find llvm_lib_dir by hand
      import platform

      if platform.system().casefold() == 'darwin':
        try:
          output = subprocess_run(
            ['xcode-select', '-p'], capture_output=True, universal_newlines=True, check=True
          )
          xcode_dir = output.stdout.strip()
          if xcode_dir == '/Applications/Xcode.app/Contents/Developer': # default Xcode path
            llvm_lib_dir = os.path.join(
              xcode_dir, 'Toolchains', 'XcodeDefault.xctoolchain', 'usr', 'lib'
            )
          elif xcode_dir == '/Library/Developer/CommandLineTools':      # CLT path
            llvm_lib_dir = os.path.join(xcode_dir, 'usr', 'lib')
        except FileNotFoundError:
          # FileNotFoundError: [Errno 2] No such file or directory: 'xcode-select'
          pass
  return pl.Path(llvm_lib_dir).resolve() if llvm_lib_dir else llvm_lib_dir

def get_petsc_extra_includes(petsc_dir, petsc_arch):
  # keep these separate, since ORDER MATTERS HERE. Imagine that for example the
  # mpiInclude dir has copies of old petsc headers, you don't want these to come first
  # in the include search path and hence override those found in petsc/include.

  # You might be thinking that seems suspiciously specific, but I was this close to filing
  # a bug report for python believing that cdll.load() was not deterministic...
  petsc_includes = []
  mpi_includes   = []
  cxx_flags      = []
  with open(pl.Path(petsc_dir, petsc_arch, 'lib', 'petsc', 'conf', 'petscvariables'), 'r') as pv:
    cc_includes_re  = re.compile('^PETSC_CC_INCLUDES\s*=')
    mpi_includes_re = re.compile('^MPI_INCLUDE\s*=')
    mpi_show_re     = re.compile('^MPICC_SHOW\s*=')
    cxx_flags_re    = re.compile('^CXX_FLAGS\s*=')
    for line in pv:
      if cc_includes_re.search(line):
        petsc_includes.append(line.split('=', maxsplit=1)[1])
      elif mpi_includes_re.search(line) or mpi_show_re.search(line):
        mpi_includes.append(line.split('=', maxsplit=1)[1])
      elif cxx_flags_re.search(line):
        cxx_flags.append(line.split('=', maxsplit=1)[1])

  cxx_flags      = [flag.strip().split() for flag in cxx_flags if flag]
  cxx_flags      = [flag for flags in cxx_flags for flag in flags if flag.startswith('-std=')]
  cxx_flags      = [cxx_flags[-1]] if cxx_flags else [] # take only the last one
  extra_includes = [flag.strip().split() for flag in petsc_includes + mpi_includes if flag]
  extra_includes = [flag for sublist in extra_includes for flag in sublist if flag.startswith('-I')]
  seen           = set()
  extra_includes = [flag for flag in extra_includes if not flag in seen and not seen.add(flag)]
  return cxx_flags + extra_includes

def get_clang_sys_includes():
  """
  Get system clangs set of default include search directories.

  Because for some reason these are hardcoded by the compilers and so libclang does not have them.
  """
  output = subprocess_run(
    ['clang', '-E', '-x', 'c++', os.devnull, '-v'],
    capture_output=True, check=True, universal_newlines=True
  )
  # goes to stderr because of /dev/null
  includes = output.stderr.split('#include <...> search starts here:\n')[1]
  includes = includes.split('End of search list.', maxsplit=1)[0].replace(
    '(framework directory)', ''
  ).splitlines()
  return [f'-I{pl.Path(i.strip()).resolve()}' for i in includes if i]

def build_compiler_flags(petsc_dir, petsc_arch, extra_compiler_flags=None, verbose=False, print_prefix='[ROOT]'):
  """
  build the baseline set of compiler flags, these are passed to all translation unit parse attempts
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
  if verbose:
    pl.sync_print('\n'.join([f'{print_prefix} Compile flags:', *compiler_flags]))
  return compiler_flags

def build_precompiled_header(petsc_dir, compiler_flags, extra_header_includes=None, verbose=False, print_prefix='[ROOT]', pch_clang_options=None):
  """
  create a precompiled header from petsc.h, and all of the private headers, this not only saves a lot
  of time, but is critical to finding struct definitions. Header contents are not parsed during the
  actual linting, since this balloons the parsing time as libclang provides no builtin auto
  header-precompilation like the normal compiler does.

  Including petsc.h first should define almost everything we need so no side effects from including
  headers in the wrong order below.
  """
  if not isinstance(petsc_dir, pl.Path):
    petsc_dir = pl.Path(petsc_dir).resolve()

  if pch_clang_options is None:
    pch_clang_options = pl.util.base_pch_clang_options

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
      if verbose:
        pl.sync_print(print_prefix, 'Included header has errors, removing', diagerrs)
      mega_header_lines = [(hdr, hfi) for hdr, hfi in mega_header_lines if hdr not in diags]
    else:
      break
  if extra_header_includes:
    # now include the other headers but this time immediately crash on errors, let the
    # user figure out their own busted header files
    mega_header += '\n'.join(extra_header_includes)
    if verbose:
      pl.sync_print('\n'.join([f'{print_prefix} Mega header:', mega_header]))
    tu = index.parse(
      mega_header_name,
      args=compiler_flags, unsaved_files=[(mega_header_name, mega_header)], options=pch_clang_options
    )
    if tu.diagnostics:
      pl.sync_print('\n'.join(map(str, tu.diagnostics)))
      raise clx.LibclangError('\n\nWarnings or errors generated when creating the precompiled header. This usually means that the provided libclang setup is faulty. If you used the auto-detection mechanism to find libclang then perhaps try specifying the location directly.')
  elif verbose:
    pl.sync_print('\n'.join([f'{print_prefix} Mega header:', mega_header]))
  precompiled_header.unlink(missing_ok=True)
  tu.save(precompiled_header)
  compiler_flags.extend(['-include-pch', str(precompiled_header)])
  if verbose:
    pl.sync_print(print_prefix, 'Saving precompiled header', precompiled_header)
  return precompiled_header
