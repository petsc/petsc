#!/usr/bin/env python3
"""
# Created: Mon Jun 20 14:35:58 2022 (-0400)
# @author: Jacob Faibussowitsch
"""
import os
import sys

if __name__ == '__main__':
  # insert the parent directory into the sys path, otherwise import petsclinter does not
  # work!
  sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import petsclinter as pl

def main(
    petsc_dir, petsc_arch,
    src_path=None,
    clang_dir=None, clang_lib=None,
    verbose=False,
    workers=-1,
    check_function_filter=None,
    patch_dir=None, apply_patches=False,
    extra_compiler_flags=None, extra_header_includes=None,
    test_output_dir=None, replace_tests=False,
    werror=False
):
  """
  entry point for linter

  Positional arguments:
  petsc_dir  -- $PETSC_DIR
  petsc_arch -- $PETSC_ARCH

  Keyword arguments:
  src_path             -- alternative directory (or single file) to use as src root (default: $PETSC_DIR/src)
  clang_dir            -- directory containing libclang.[so|dylib|dll] (default: None)
  clang_lib            -- direct path to libclang.[so|dylib|dll], overrrides clang_dir if set (default: None)
  verbose             -- display debugging statements (default: False)
  workers             -- number of processes for multiprocessing, -1 is number of system CPU's-1, 0 or 1 for serial computation (default: -1)
  check_function_filter -- list of function names as strings to only check for, none == all of them. For example ["PetscValidPointer","PetscValidHeaderSpecific"] (default: None)
  patch_dir            -- directory to store patches if they are generated (default: $PETSC_DIR/petscLintPatches)
  apply_patches        -- automatically apply patch files to source if they are generated (default: False)
  extra_compiler_flags  -- list of extra compiler flags to append to petsc and system flags. For example ["-I/my/non/standard/include","-Wsome_warning"] (default: None)
  extra_header_includes -- list of #include statements to append to the precompiled mega-header, these must be in the include search path. Use extra_compiler_flags to make any other search path additions. For example ["#include <slepc/private/epsimpl.h>"] (default: None)
  test_output_dir       -- directory containing test output to compare patches against, use special keyword '__at_src__' to use src_path/output (default: None)
  replace_tests        -- replace output files in test_output_dir with patches generated (default: False)
  werror              -- treat all linter-generated warnings as errors (default: False)
  """
  if extra_compiler_flags is None:
    extra_compiler_flags = []

  if extra_header_includes is None:
    extra_header_includes = []

  # pre-processing setup
  if bool(apply_patches) and bool(test_output_dir):
    raise RuntimeError('Test directory and apply patches are both non-zero. It is probably not a good idea to apply patches over the test directory!')
  clang_dir, clang_lib = pl.util.initialize_libclang(clang_dir=clang_dir, clang_lib=clang_lib)

  petsc_dir = pl.Path(petsc_dir).resolve()
  src_path  = petsc_dir / 'src' if src_path is None else pl.Path(src_path).resolve()
  patch_dir = petsc_dir / 'petscLintPatches' if patch_dir is None else pl.Path(patch_dir).resolve()
  if test_output_dir == '__at_src__':
    if src_path.is_dir():
      test_output_dir = src_path / 'output'
    elif src_path.is_file():
      test_output_dir = src_path.parent / 'output'
    else:
      raise RuntimeError(f'Got neither a directory or file as src_path {src_path}')

  if test_output_dir is not None and not test_output_dir.exists():
    raise RuntimeError(f'Test Output Directory {test_output_dir} does not appear to exist')

  pl.checks.filter_check_function_map(check_function_filter)
  root_print_prefix = '[ROOT]'
  compiler_flags    = pl.util.build_compiler_flags(
    petsc_dir, petsc_arch, extra_compiler_flags=extra_compiler_flags, verbose=verbose
  )

  class PrecompiledHeader:
    __slots__ = ('pch',)

    def __init__(self, *args, **kwargs):
      self.pch = pl.util.build_precompiled_header(*args, **kwargs)
      return

    def __enter__(self):
      return self

    def __exit__(self, *args, **kwargs):
      if verbose:
        pl.sync_print(root_print_prefix, 'Deleting precompiled header', self.pch)
        self.pch.unlink()
      return

  with PrecompiledHeader(
      petsc_dir, compiler_flags, extra_header_includes=extra_header_includes, verbose=verbose
  ):
    warnings, errors_left, errors_fixed, patches = pl.WorkerPool(
      workers, verbose=verbose
    ).setup(compiler_flags, werror=werror).walk(src_path).finalize()

  if test_output_dir is not None:
    from petsclinter.test_main import test_main

    pl.sync_print('', end='', flush=True)
    return test_main(
      petsc_dir, src_path, test_output_dir, patches, errors_fixed, errors_left,
      replace=replace_tests, verbose=verbose
    )
  elif patches:
    import time

    patch_dir.mkdir(exist_ok=True)
    mangle_postfix = f'_{int(time.time())}.patch'
    root_dir       = f'--directory={patch_dir.anchor}'

    for fname, patch in patches:
      mangled_rel = fname.append_name(mangle_postfix)
      if mangled_rel.parent != src_path.parent:
        # not in same directory
        mangled_rel = mangled_rel.relative_to(src_path)
      mangled_file = patch_dir / str(mangled_rel).replace(os.path.sep, '_')
      if verbose: pl.sync_print(root_print_prefix, 'Writing patch to file', mangled_file)
      mangled_file.write_text(patch)

    if apply_patches:
      if verbose: pl.sync_print(root_print_prefix, 'Applying patches from patch directory', patch_dir)
      for patch_file in patch_dir.glob('*' + mangle_postfix):
        if verbose: pl.sync_print(root_print_prefix, 'Applying patch', patch_file)
        output = pl.util.subprocess_run(
          ['patch', root_dir, '--strip=0', '--unified', f'--input={patch_file}'],
          check=True, universal_newlines=True, capture_output=True
        )
        if verbose: pl.sync_print(output.stdout)

  ret        = 0
  format_str = ' '.join([root_print_prefix, '{:=^85}'])
  if warnings and verbose:
    pl.sync_print(format_str.format(' Found Warnings '))
    pl.sync_print('\n'.join(s for tup in warnings for _, s in tup))
    pl.sync_print(format_str.format(' End warnings '))
  if errors_fixed and verbose:
    pl.sync_print(format_str.format(' Fixed Errors ' if apply_patches else ' Fixable Errors '))
    pl.sync_print('\n'.join(e for _, e in errors_fixed))
    pl.sync_print(format_str.format(' End Fixed Errors '))
  if errors_left:
    pl.sync_print(format_str.format(' Unfixable Errors '))
    pl.sync_print('\n'.join(e for _, e in errors_left))
    pl.sync_print(format_str.format(' End Unfixable Errors '))
    pl.sync_print('Some errors or warnings could not be automatically corrected via the patch files')
    ret = 11
  if patches:
    if apply_patches:
      pl.sync_print('All fixable errors or warnings successfully patched')
    else:
      pl.sync_print('Patch files written to', patch_dir)
      pl.sync_print('Apply manually using:')
      pl.sync_print(
        f'  patch {root_dir} --strip=0 --unified --input={patch_dir/("*" + mangle_postfix)}'
      )
      if ret != 0:
        ret = 12
  return ret

def __build_arg_parser(parent_parsers=None):
  import argparse

  def add_bool_argument(prsr, *args, **kwargs):
    def str2bool(v):
      if isinstance(v, bool):
        return v
      v = v.casefold()
      if v in {'yes', 'true', 't', 'y', '1'}:
        return True
      if v in {'no', 'false', 'f', 'n', '0', ''}:
        return False
      raise argparse.ArgumentTypeError(f'Boolean value expected, got \'{v}\'')

    return prsr.add_argument(*args, **kwargs, metavar='bool', type=str2bool)

  if parent_parsers is None:
    parent_parsers = []

  clang_dir = pl.util.try_to_find_libclang_dir()
  try:
    petsc_dir       = os.environ['PETSC_DIR']
    default_src_dir = pl.Path(petsc_dir).resolve()/'src'
  except KeyError:
    petsc_dir       = None
    default_src_dir = '$PETSC_DIR/src'
  try:
    petsc_arch = os.environ['PETSC_ARCH']
  except KeyError:
    petsc_arch = None

  parser = argparse.ArgumentParser(
    prog='petsclinter',
    description='set options for clang static analysis tool',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    parents=parent_parsers
  )

  group_libclang = parser.add_argument_group(title='libClang location settings')
  group          = group_libclang.add_mutually_exclusive_group(required=False)
  group.add_argument('--clang_dir', metavar='path', type=pl.Path, nargs='?', default=clang_dir, help='directory containing libclang.[so|dylib|dll], if not given attempts to automatically detect it via llvm-config', dest='clang_dir')
  group.add_argument('--clang_lib', metavar='path', type=pl.Path, nargs='?', help='direct location of libclang.[so|dylib|dll], overrides clang directory if set', dest='clang_lib')

  group_petsc = parser.add_argument_group(title='PETSc location settings')
  group_petsc.add_argument('--PETSC_DIR', default=petsc_dir, help='if this option is unused defaults to environment variable $PETSC_DIR', dest='petsc_dir')
  group_petsc.add_argument('--PETSC_ARCH', default=petsc_arch, help='if this option is unused defaults to environment variable $PETSC_ARCH', dest='petsc_arch')

  parser.add_argument('-s', '--src-path', default=default_src_dir, help='path to file or directory containing source (e.g. $SLEPC_DIR/src)', dest='src_path')
  add_bool_argument(parser, '-v', '--verbose', nargs='?', const=True, default=False, help='verbose progress printed to screen')

  check_function_map_keys = list(pl.checks._register.check_function_map.keys())
  filter_func_choices     = ', '.join(check_function_map_keys)
  parser.add_argument('--functions', nargs='+', choices=check_function_map_keys, metavar='FUNCTIONNAME', help='filter to display errors only related to list of provided function names, default is all functions. Choose from available function names: '+filter_func_choices, dest='check_function_filter')
  foo = parser.add_argument('-j', '--jobs', type=int, const=-1, default=-1, nargs='?', help='number of multiprocessing jobs, -1 means number of processors on machine', dest='workers')
  parser.add_argument('-p', '--patch-dir', help='directory to store patches in if they are generated, defaults to SRC_DIR/../petscLintPatches', dest='patch_dir')
  add_bool_argument(parser, '-a', '--apply-patches', nargs='?', const=True, default=False, help='automatically apply patches that are saved to file', dest='apply_patches')
  parser.add_argument('--CXXFLAGS', nargs='+', default=[], help='extra flags to pass to CXX compiler', dest='extra_compiler_flags')
  parser.add_argument('--INCLUDEFLAGS', nargs='+', default=[], help='extra include flags to pass to CXX compiler', dest='extra_header_includes')
  parser.add_argument('--test', metavar='path', nargs='?', const='__at_src__', help='test the linter for correctness. Optionally provide a directory containing the files against which to compare patches, defaults to SRC_DIR/output if no argument is given. The files of correct patches must be in the format [path_from_src_dir_to_testFileName].out', dest='test_output_dir')
  add_bool_argument(parser, '--replace', nargs='?', const=True, default=False, help='replace output files in test directory with patches generated', dest='replace_tests')
  add_bool_argument(parser, '--werror', nargs='?', const=True, default=False, help='treat all warnings as errors')

  class CheckFilter(argparse.Action):
    def __call__(self, parser, namespace, values, *args, **kwargs):
      flag = self.dest.replace(pl.DiagnosticManager.flagprefix[1:], '', 1).replace('_', '-')
      if flag == 'diagnostics-all':
        for diag, _ in pl.DiagnosticManager.registered().items():
          pl.DiagnosticManager.set(diag, values)
      else:
        pl.DiagnosticManager.set(flag, values)
      setattr(namespace, flag, values)
      return

  group_diag = parser.add_argument_group(title='diagnostics')
  add_bool_argument(
    group_diag, '-fdiagnostics-all', nargs='?', const=True, default=True, action=CheckFilter,
    help='enable all diagnostics'
  )

  all_diagnostics = set()
  for diag, helpstr in sorted(pl.DiagnosticManager.registered().items()):
    diag_flag = f'-f{diag}'
    add_bool_argument(
      group_diag, diag_flag, nargs='?', const=True, default=True, action=CheckFilter, help=helpstr
    )
    all_diagnostics.add(diag_flag)

  return parser, all_diagnostics

def parse_command_line_args(argv=None, **kwargs):
  import re

  def expand_argv_globs(in_argv, diagnostics):
    argv  = []
    skip  = False
    nargv = len(in_argv)

    # always skip first entry of argv
    for i, argi in enumerate(in_argv[1:], start=1):
      if skip:
        skip = False
        continue
      if argi.startswith('-f') and '*' in argi:
        if i + 1 >= len(in_argv):
          parser.error(f'Glob argument {argi} must be followed by explicit value!')

        next_arg = in_argv[i+1]
        pattern  = re.compile(argi.replace('*', '.*'))
        for flag_to_add in filter(pattern.match, diagnostics):
          argv.extend((flag_to_add, next_arg))
        skip = True
      else:
        argv.append(argi)
    return argv

  if argv is None:
    argv = sys.argv

  parser, all_diagnostics = __build_arg_parser(**kwargs)
  args                    = parser.parse_args(args=expand_argv_globs(argv, tuple(all_diagnostics)))

  if args.petsc_dir is None:
    raise RuntimeError('Could not determine PETSC_DIR from environment, please set via options')
  if args.petsc_arch is None:
    raise RuntimeError('Could not determine PETSC_ARCH from environment, please set via options')

  if args.clang_lib:
    args.clang_dir = None

  return args, parser

def namespace_main(args):
  return main(
    args.petsc_dir, args.petsc_arch,
    src_path=args.src_path,
    clang_dir=args.clang_dir, clang_lib=args.clang_lib,
    verbose=args.verbose,
    workers=args.workers,
    check_function_filter=args.check_function_filter,
    patch_dir=args.patch_dir, apply_patches=args.apply_patches,
    extra_compiler_flags=args.extra_compiler_flags, extra_header_includes=args.extra_header_includes,
    test_output_dir=args.test_output_dir, replace_tests=args.replace_tests,
    werror=args.werror
  )

def command_line_main():
  args, _ = parse_command_line_args()
  return namespace_main(args)

if __name__ == '__main__':
  sys.exit(command_line_main())
