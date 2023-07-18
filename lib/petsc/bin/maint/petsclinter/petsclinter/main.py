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
import enum

@enum.unique
class ReturnCode(enum.IntFlag):
  SUCCESS           = 0
  ERROR_WERROR      = enum.auto()
  ERROR_ERROR_FIXED = enum.auto()
  ERROR_ERROR_LEFT  = enum.auto()
  ERROR_ERROR_TEST  = enum.auto()

def main(
    petsc_dir, petsc_arch,
    src_path=None,
    clang_dir=None, clang_lib=None, clang_compat_check=True,
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
  src_path              -- directory (or file) to lint (default: $PETSC_DIR/src)
  clang_dir             -- directory containing libclang.[so|dylib|dll] (default: None)
  clang_lib             -- direct path to libclang.[so|dylib|dll], overrrides clang_dir if set (default: None)
  clang_compat_check    -- do clang lib compatibility check
  verbose               -- display debugging statements (default: False)
  workers               -- number of processes for multiprocessing, -1 is number of system CPU's-1, 0 or 1 for serial computation (default: -1)
  check_function_filter -- list of function names as strings to only check for, none == all of them. For example ["PetscAssertPointer","PetscValidHeaderSpecific"] (default: None)
  patch_dir             -- directory to store patches if they are generated (default: $PETSC_DIR/petscLintPatches)
  apply_patches         -- automatically apply patch files to source if they are generated (default: False)
  extra_compiler_flags  -- list of extra compiler flags to append to petsc and system flags. For example ["-I/my/non/standard/include","-Wsome_warning"] (default: None)
  extra_header_includes -- list of #include statements to append to the precompiled mega-header, these must be in the include search path. Use extra_compiler_flags to make any other search path additions. For example ["#include <slepc/private/epsimpl.h>"] (default: None)
  test_output_dir       -- directory containing test output to compare patches against, use special keyword '__at_src__' to use src_path/output (default: None)
  replace_tests         -- replace output files in test_output_dir with patches generated (default: False)
  werror                -- treat all linter-generated warnings as errors (default: False)
  """
  if extra_compiler_flags is None:
    extra_compiler_flags = []

  if extra_header_includes is None:
    extra_header_includes = []

  def root_sync_print(*args, **kwargs):
    if args or kwargs:
      print('[ROOT]', *args, **kwargs)
    return
  pl.sync_print = root_sync_print

  # pre-processing setup
  if bool(apply_patches) and bool(test_output_dir):
    raise RuntimeError('Test directory and apply patches are both non-zero. It is probably not a good idea to apply patches over the test directory!')
  clang_dir, clang_lib = pl.util.initialize_libclang(
    clang_dir=clang_dir, clang_lib=clang_lib, compat_check=clang_compat_check
  )

  petsc_dir = pl.Path(petsc_dir).resolve()
  if src_path is None:
    src_path = [petsc_dir / 'src']
  else:
    if isinstance(src_path, (str, pl.Path)):
      src_path = [src_path]
    if not isinstance(src_path, (list, tuple)):
      raise RuntimeError(f'Source path must be a :ist or Tuple, not {type(src_path)}')
    src_path = [pl.Path(s).resolve() for s in src_path]

  for p in src_path:
    if not p.exists():
      raise RuntimeError(f'Path {p} does not exist!')

  patch_dir = petsc_dir / 'petscLintPatches' if patch_dir is None else pl.Path(patch_dir).resolve()
  if patch_dir.exists() and not patch_dir.is_dir():
    raise RuntimeError(f'Patch Directory (as the name suggests) must be a directory, not {patch_dir}')

  if test_output_dir == '__at_src__':
    if len(src_path) != 1:
      raise RuntimeError(
        f'Can only use default test output dir for single file or directory, not {len(src_path)}'
      )

    test_src_path = src_path[0]
    if test_src_path.is_dir():
      test_output_dir = test_src_path / 'output'
    elif test_src_path.is_file():
      test_output_dir = test_src_path.parent / 'output'
    else:
      raise RuntimeError(f'Got neither a directory or file as src_path {test_src_path}')

  if test_output_dir is not None and not test_output_dir.exists():
    raise RuntimeError(f'Test Output Directory {test_output_dir} does not appear to exist')

  pl.checks.filter_check_function_map(check_function_filter)
  compiler_flags    = pl.util.build_compiler_flags(
    petsc_dir, petsc_arch, extra_compiler_flags=extra_compiler_flags, verbose=verbose
  )

  if len(src_path) == 1 and src_path[0].is_file():
    if verbose:
      pl.sync_print(f'Only processing a single file ({src_path[0]}), setting number of workers to 1')
    workers = 1

  with pl.util.PrecompiledHeader.from_flags(
      petsc_dir, compiler_flags, extra_header_includes=extra_header_includes, verbose=verbose
  ):
    warnings, errors_left, errors_fixed, patches = pl.WorkerPool(
      workers, verbose=verbose
    ).setup(compiler_flags, clang_compat_check=clang_compat_check, werror=werror).walk(
      src_path
    ).finalize()

  if test_output_dir is not None:
    from petsclinter.test_main import test_main

    # reset the printer
    pl.sync_print = print
    pl.sync_print('', end='', flush=True)
    assert len(src_path) == 1
    return test_main(
      petsc_dir, src_path[0], test_output_dir, patches, errors_fixed, errors_left,
      replace=replace_tests, verbose=verbose
    )
  elif patches:
    import time
    import shutil

    patch_dir.mkdir(exist_ok=True)
    mangle_postfix = f'_{int(time.time())}.patch'
    root_dir       = f'--directory={patch_dir.anchor}'
    patch_exec     = shutil.which('patch')

    if patch_exec is None:
      # couldn't find it, but let's just try out the bare name and hope it works,
      # otherwise this will error below anyways
      patch_exec = 'patch'

    for fname, patch in patches:
      # mangled_rel = fname.append_name(mangle_postfix)
      # assert mangled_rel.parent == src_path[0].parent
      # not in same directory
      # mangled_rel = mangled_rel.relative_to(src_path)
      mangled_file = patch_dir / str(fname.append_name(mangle_postfix)).replace(os.path.sep, '_')
      if verbose: pl.sync_print('Writing patch to file', mangled_file)
      mangled_file.write_text(patch)

    if apply_patches:
      if verbose: pl.sync_print('Applying patches from patch directory', patch_dir)
      for patch_file in patch_dir.glob('*' + mangle_postfix):
        if verbose: pl.sync_print('Applying patch', patch_file)
        output = pl.util.subprocess_run(
          [patch_exec, root_dir, '--strip=0', '--unified', f'--input={patch_file}'],
          check=True, universal_newlines=True, capture_output=True
        )
        if verbose: pl.sync_print(output.stdout)

  ret        = ReturnCode.SUCCESS
  format_str = '{:=^85}'
  if warnings:
    if verbose:
      pl.sync_print(format_str.format(' Found Warnings '))
      pl.sync_print('\n'.join(s for tup in warnings for _, s in tup))
      pl.sync_print(format_str.format(' End warnings '))
    if werror:
      ret |= ReturnCode.ERROR_WERROR
  if errors_fixed:
    if verbose:
      pl.sync_print(format_str.format(' Fixed Errors ' if apply_patches else ' Fixable Errors '))
      pl.sync_print('\n'.join(e for _, e in errors_fixed))
      pl.sync_print(format_str.format(' End Fixed Errors '))
    ret |= ReturnCode.ERROR_ERROR_FIXED
  if errors_left:
    pl.sync_print(format_str.format(' Unfixable Errors '))
    pl.sync_print('\n'.join(e for _, e in errors_left))
    pl.sync_print(format_str.format(' End Unfixable Errors '))
    pl.sync_print('Some errors or warnings could not be automatically corrected via the patch files')
    ret |= ReturnCode.ERROR_ERROR_LEFT
  if patches:
    if apply_patches:
      pl.sync_print('All fixable errors or warnings successfully patched')
      if ret == ReturnCode.ERROR_ERROR_FIXED:
        # if the only error is fixed errors, then we don't actually have an error
        ret = ReturnCode.SUCCESS
    else:
      pl.sync_print('Patch files written to', patch_dir)
      pl.sync_print('Apply manually using:')
      pl.sync_print(
        f'  for patch_file in {patch_dir / ("*" + mangle_postfix)}; do {patch_exec} {root_dir} --strip=0 --unified --input=${{patch_file}}; done'
      )
      assert ret != ReturnCode.SUCCESS
  return int(ret)


__ADVANCED_HELP_FLAG__ = '--help-hidden'

def __build_arg_parser(parent_parsers=None, advanced_help=False):
  import argparse

  def add_advanced_argument(prsr, *args, help=None, **kwargs):
    def help_str(descr):
      return descr if advanced_help else argparse.SUPPRESS

    return prsr.add_argument(*args, help=help_str(help), **kwargs)

  def add_bool_argument(prsr, *args, advanced=False, **kwargs):
    def str2bool(v):
      if isinstance(v, bool):
        return v
      v = v.casefold()
      if v in {'yes', 'true', 't', 'y', '1'}:
        return True
      if v in {'no', 'false', 'f', 'n', '0', ''}:
        return False
      raise argparse.ArgumentTypeError(f'Boolean value expected, got \'{v}\'')

    kwargs.setdefault('nargs', '?')
    kwargs.setdefault('const', True)
    kwargs.setdefault('default', False)
    kwargs.setdefault('metavar', 'bool')
    kwargs['type'] = str2bool
    if advanced:
      return add_advanced_argument(prsr, *args, **kwargs)
    return prsr.add_argument(*args, **kwargs)

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

  # don't use an argument group for this so it appears directly next to default --help
  # description!
  add_bool_argument(
    parser, __ADVANCED_HELP_FLAG__, help='show more help output (e.g. the various check flags)'
  )

  group_general = parser.add_argument_group(title='General options')
  group_general.add_argument('--version', action='version', version=f'%(prog)s {pl.version_str()}')
  add_bool_argument(group_general, '-v', '--verbose', help='verbose progress printed to screen')
  add_bool_argument(group_general, '--pm', help='launch an IPython post_mortem() on any raised exceptions (implies -j/--jobs 1)')
  add_bool_argument(group_general, '--werror', help='treat all warnings as errors')
  group_general.add_argument('-j', '--jobs', type=int, const=-1, default=-1, nargs='?', help='number of multiprocessing jobs, -1 means number of processors on machine', dest='workers')
  group_general.add_argument('-p', '--patch-dir', help='directory to store patches in if they are generated, defaults to SRC_DIR/../petscLintPatches', dest='patch_dir')
  add_bool_argument(group_general, '-a', '--apply-patches', help='automatically apply patches that are saved to file', dest='apply_patches')
  group_general.add_argument('--CXXFLAGS', nargs='+', default=[], help='extra flags to pass to CXX compiler', dest='extra_compiler_flags')
  group_general.add_argument('--INCLUDEFLAGS', nargs='+', default=[], help='extra include flags to pass to CXX compiler', dest='extra_header_includes')

  group_libclang = parser.add_argument_group(title='libClang location settings')
  add_bool_argument(group_libclang, '--clang-compat-check', default=True, help='enable clang compatibility check')
  group          = group_libclang.add_mutually_exclusive_group(required=False)
  group.add_argument('--clang_dir', metavar='path', type=pl.Path, nargs='?', default=clang_dir, help='directory containing libclang.[so|dylib|dll], if not given attempts to automatically detect it via llvm-config', dest='clang_dir')
  group.add_argument('--clang_lib', metavar='path', type=pl.Path, nargs='?', help='direct location of libclang.[so|dylib|dll], overrides clang directory if set', dest='clang_lib')

  group_petsc = parser.add_argument_group(title='PETSc location settings')
  group_petsc.add_argument('--PETSC_DIR', default=petsc_dir, help='if this option is unused defaults to environment variable $PETSC_DIR', dest='petsc_dir')
  group_petsc.add_argument('--PETSC_ARCH', default=petsc_arch, help='if this option is unused defaults to environment variable $PETSC_ARCH', dest='petsc_arch')

  group_test = parser.add_argument_group(title='Testing settings')
  group_test.add_argument('--test', metavar='path', nargs='?', const='__at_src__', help='test the linter for correctness. Optionally provide a directory containing the files against which to compare patches, defaults to SRC_DIR/output if no argument is given. The files of correct patches must be in the format [path_from_src_dir_to_testFileName].out', dest='test_output_dir')
  add_bool_argument(group_test, '--replace', help='replace output files in test directory with patches generated', dest='replace_tests')

  group_diag = parser.add_argument_group(title='Diagnostics settings')
  check_function_map_keys = list(pl.checks._register.check_function_map.keys())
  filter_func_choices     = ', '.join(check_function_map_keys)
  add_advanced_argument(group_diag, '--functions', nargs='+', choices=check_function_map_keys, metavar='FUNCTIONNAME', help='filter to display errors only related to list of provided function names, default is all functions. Choose from available function names: '+filter_func_choices, dest='check_function_filter')

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

  add_bool_argument(
    group_diag, '-fdiagnostics-all', default=True, action=CheckFilter, advanced=True,
    help='enable all diagnostics'
  )

  all_diagnostics = set()
  flag_prefix     = pl.DiagnosticManager.flagprefix
  for diag, helpstr in sorted(pl.DiagnosticManager.registered().items()):
    diag_flag = f'{flag_prefix}{diag}'
    add_bool_argument(
      group_diag, diag_flag, default=True, action=CheckFilter, advanced=True, help=helpstr
    )
    all_diagnostics.add(diag_flag)

  parser.add_argument('src_path', default=default_src_dir, help='path to files or directory containing source (e.g. $SLEPC_DIR/src)', nargs='*')
  return parser, all_diagnostics

def parse_command_line_args(argv=None, **kwargs):
  import re

  def expand_argv_globs(in_argv, diagnostics):
    argv        = []
    skip        = False
    nargv       = len(in_argv)
    flag_prefix = pl.DiagnosticManager.flagprefix

    # always skip first entry of argv
    for i, argi in enumerate(in_argv[1:], start=1):
      if skip:
        skip = False
        continue
      if argi.startswith(flag_prefix) and '*' in argi:
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

  kwargs.setdefault('advanced_help', __ADVANCED_HELP_FLAG__ in argv)
  parser, all_diagnostics = __build_arg_parser(**kwargs)
  args                    = parser.parse_args(args=expand_argv_globs(argv, tuple(all_diagnostics)))

  if getattr(args, __ADVANCED_HELP_FLAG__.replace('-', '_').lstrip('_')):
    parser.print_help()
    parser.exit(0)

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
    clang_dir=args.clang_dir, clang_lib=args.clang_lib, clang_compat_check=args.clang_compat_check,
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
  have_pm = args.pm
  if have_pm:
    if args.verbose:
      pl.sync_print('Running with --pm flag, setting number of workers to 1')
    args.workers = 1
    try:
      import ipdb as py_db # LINT IGNORE
    except ModuleNotFoundError:
      import pdb as py_db # LINT IGNORE

  try:
    return namespace_main(args)
  except:
    if have_pm:
      py_db.post_mortem()
    raise

if __name__ == '__main__':
  sys.exit(command_line_main())
