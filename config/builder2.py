#!/usr/bin/env python
import builder
import os, sys

# Find PETSc/BuildSystem
if 'PETSC_DIR' in os.environ:
  configDir = os.path.join(os.environ['PETSC_DIR'], 'config')
  bsDir     = os.path.join(configDir, 'BuildSystem')
  sys.path.insert(0, bsDir)
  sys.path.insert(0, configDir)

def build(args):
  '''Compile all out of date source and generate a new shared library
  Any files specified will be added to the source database if not already present'''
  # TODO: Make rootDir an option (this should really be a filter for sourceDB)
  maker = builder.PETScMaker()
  maker.setup()
  maker.updateDependencies('libpetsc', maker.rootDir)
  for p in args.files:
    filename = os.path.abspath(p)
    if not maker.sourceDatabase.hasNode(filename):
      maker.logPrint('Adding %s to the source database' % filename)
      maker.sourceDatabase.setNode(filename, [])
  if maker.buildLibraries('libpetsc', maker.rootDir, args.parallel):
    # This is overkill, but right now it is cheap
    maker.rebuildDependencies('libpetsc', maker.rootDir)
  maker.cleanup()
  return 0

def check(args):
  '''Check that build is functional'''
  import shutil

  extraArgs = ' '+' '.join(args.args)
  maker = builder.PETScMaker()
  maker.setup()
  # C test
  if len(args.files):
    examples = []
    for f in args.files:
      if f[0] == '[':
        examples.append(map(os.path.abspath, f[1:-1].split(',')))
      else:
        examples.append(os.path.abspath(f))
  else:
    examples = [os.path.join(maker.petscDir, 'src', 'snes', 'examples', 'tutorials', 'ex5.c')]
  # Fortran test
  if hasattr(maker.configInfo.compilers, 'FC'):
    if maker.configInfo.fortrancpp.fortranDatatypes:
      examples.append(os.path.join(maker.petscDir, 'src', 'snes', 'examples', 'tutorials', 'ex5f90t.F'))
    elif maker.configInfo.compilers.fortranIsF90:
      examples.append(os.path.join(maker.petscDir, 'src', 'snes', 'examples', 'tutorials', 'ex5f90.F'))
    else:
      examples.append(os.path.join(maker.petscDir, 'src', 'snes', 'examples', 'tutorials', 'ex5f.F'))
  for ex in examples:
    if isinstance(ex, list):
      exampleName = os.path.splitext(os.path.basename(ex[0]))[0]
      exampleDir  = os.path.dirname(ex[0])
    else:
      exampleName = os.path.splitext(os.path.basename(ex))[0]
      exampleDir  = os.path.dirname(ex)
    objDir      = maker.getObjDir(exampleName)
    if os.path.isdir(objDir): shutil.rmtree(objDir)
    os.mkdir(objDir)
    objects     = maker.buildFile(ex, objDir)
    if not len(objects):
      print 'TEST FAILED (check make.log for details)'
      return 1
    executable = os.path.splitext(objects[0])[0]
    paramKey   = os.path.join(os.path.relpath(exampleDir, maker.petscDir), os.path.basename(executable))
    if paramKey in builder.regressionRequirements:
      if not builder.regressionRequirements[paramKey].issubset(packageNames):
        raise RuntimeError('This test requires packages: %s' % builder.regressionRequirements[paramKey])
    maker.link(executable, objects, maker.configInfo.languages.clanguage)
    params = builder.regressionParameters.get(paramKey, {})
    if isinstance(params, list):
      for testnum, param in enumerate(params):
        if not 'args' in param: param['args'] = ''
        param['args'] += extraArgs
        if maker.runTest(exampleDir, executable, testnum, **param):
          print 'TEST FAILED (check make.log for details)'
          return 1
    else:
      if not 'args' in params: params['args'] = ''
      params['args'] += extraArgs
      if maker.runTest(exampleDir, executable, 1, **params):
        print 'TEST FAILED (check make.log for details)'
        return 1
    if not args.retain and os.path.isdir(objDir): shutil.rmtree(objDir)
  print 'All tests pass'
  maker.cleanup()
  return 0

def clean(args):
  '''Remove source database and all objects'''
  maker = builder.PETScMaker()
  maker.setup()
  maker.clean('libpetsc')
  maker.cleanup()
  return 0

def purge(args):
  '''Remove a sets of files from the source database'''
  maker = builder.PETScMaker()
  maker.setup()
  maker.updateDependencies('libpetsc', maker.rootDir)
  for p in args.files:
    filename = os.path.abspath(p)
    maker.logPrint('Removing %s from the source database' % filename)
    maker.sourceDatabase.removeNode(filename)
  maker.cleanup()
  return 0

def stubs(args):
  '''Build stubs for certain languages'''
  maker = builder.PETScMaker()
  maker.setup()
  for language in args.languages:
    print language
    getattr(maker, 'build'+language.capitalize()+'Stubs')()
  maker.cleanup()
  return 0

# Argumnt parsing
import argparse

parser = argparse.ArgumentParser(description     = 'PETSc BuildSystem',
                                 epilog          = 'For more information, visit http://www.mcs.anl.gov/petsc',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--version', action='version', version='PETSc BuildSystem 0.1')
subparsers = parser.add_subparsers(help='build actions')

parser_build = subparsers.add_parser('build', help='Compile all out of date source and generate a new shared library')
parser_build.add_argument('files', nargs='*', help='Extra files to incorporate into the build')
parser_build.add_argument('--parallel', action='store_true', default=False, help='Execute the build in parallel')
parser_build.set_defaults(func=build)
parser_check = subparsers.add_parser('check', help='Check that build is functional')
parser_check.add_argument('files', nargs='*', help='Extra examples to test')
parser_check.add_argument('--args', action='append', default=[], help='Extra execution arguments for test')
parser_check.add_argument('--retain', action='store_true', default=False, help='Retain the executable after testing')
parser_check.set_defaults(func=check)
parser_clean = subparsers.add_parser('clean', help='Remove source database and all objects')
parser_clean.set_defaults(func=clean)
parser_purge = subparsers.add_parser('purge', help='Remove a set of files from the source database')
parser_purge.add_argument('files', nargs='+', help='Files to remove from the source database')
parser_purge.set_defaults(func=purge)
parser_stubs = subparsers.add_parser('stubs', help='Build stubs for certain languages')
parser_stubs.add_argument('languages', nargs='+', help='Stub languages')
parser_stubs.set_defaults(func=stubs)

args = parser.parse_args()
print(args)
args.func(args)
