#!/usr/bin/env python
from __future__ import print_function
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
  if maker.buildLibraries('libpetsc', maker.rootDir, args.parallel) and args.rebuildDep:
    # This is overkill, but right now it is cheap
    maker.rebuildDependencies('libpetsc', maker.rootDir)
  maker.cleanup()
  return 0

def buildSingleExample(maker, ex):
  import shutil

  if isinstance(ex, list):
    exampleName = os.path.splitext(os.path.basename(ex[0]))[0]
    exampleDir  = os.path.dirname(ex[0])
  else:
    exampleName = os.path.splitext(os.path.basename(ex))[0]
    exampleDir  = os.path.dirname(ex)
  objDir        = maker.getObjDir(exampleName)
  if os.path.isdir(objDir): shutil.rmtree(objDir)
  os.mkdir(objDir)
  executable = os.path.join(objDir, exampleName)
  objects    = maker.buildFile(ex, objDir)
  if not len(objects):
    print('EXAMPLE BUILD FAILED (check example.log for details)')
    return 1
  maker.link(executable, objects, maker.configInfo.languages.clanguage)
  return 0

def buildExample(args):
  '''Build and link an example'''
  ret   = 0
  maker = builder.PETScMaker('example.log')
  maker.setup()
  examples = []
  for f in args.files:
    if f[0] == '[':
      examples.append(map(os.path.abspath, f[1:-1].split(',')))
    else:
      examples.append(os.path.abspath(f))
  for ex in examples:
    ret = buildSingleExample(maker, ex)
    if ret: break
  maker.cleanup()
  return ret

def checkSingleRun(maker, ex, replace, extraArgs = '', isRegression = False):
  import shutil

  packageNames = set([p.name for p in maker.framework.packages])
  if isinstance(ex, list):
    exampleName = os.path.splitext(os.path.basename(ex[0]))[0]
    exampleDir  = os.path.dirname(ex[0])
  else:
    exampleName = os.path.splitext(os.path.basename(ex))[0]
    exampleDir  = os.path.dirname(ex)
  objDir        = maker.getObjDir(exampleName)
  if os.path.isdir(objDir): shutil.rmtree(objDir)
  os.mkdir(objDir)
  executable  = os.path.join(objDir, exampleName)
  paramKey    = os.path.join(os.path.relpath(exampleDir, maker.petscDir), os.path.basename(executable))
  params = builder.regressionParameters.get(paramKey, {})
  if not params:
    params = builder.getRegressionParameters(maker, exampleDir).get(paramKey, {})
    if params: maker.logPrint('Retrieved test options from makefile: %s\n' % (str(params),))
  if isRegression and not params:
    return
  if not isinstance(params, list):
    params = [params]
  # Process testnum
  if args.testnum is not None:
    if args.testnum[0] == '[':
      validTestnum = args.testnum[1:-1].split(',')
    elif args.testnum[0] == '@':
      import re
      validTestnum = [str(num) if key is None else key for num,key in enumerate(map(lambda p: p.get('num', None), params)) if re.match(args.testnum[1:], str(num) if key is None else key)]
    else:
      validTestnum = [args.testnum]
    numtests = len(validTestnum)
  else:
    numtests = len(params)
  rebuildTest = True
  maker.logPrint('Running %d tests\n' % (numtests,), debugSection='screen', forceScroll=True)
  for testnum, param in enumerate(params):
    testnum = str(testnum)
    if 'requires' in param:
      reqs = param['requires']
      if not isinstance(reqs,list):
        reqs = [reqs]
      if not set(reqs).issubset(packageNames):
        maker.logPrint('Test %s requires packages %s\n' % (testnum, reqs), debugSection='screen', forceScroll=True)
        continue
    if 'num' in param: testnum = param['num']
    if 'numProcs' in args and not args.numProcs is None:
      param['numProcs'] = args.numProcs
    if not args.testnum is None and not testnum in validTestnum: continue
    if 'setup' in param:
      print(param['setup'])
      os.system(sys.executable+' '+param['setup'])
      rebuildTest = True
    if 'source' in param:
      if not isinstance(ex, list):
        ex = [ex]+param['source']
      else:
        ex = ex+param['source']
    # TODO: Fix this hack
    if ex[-1] == 'F':
      linkLanguage = 'FC'
    else:
      linkLanguage = maker.configInfo.languages.clanguage
    if rebuildTest:
      objects = maker.buildFile(ex, objDir)
      if not len(objects):
        print('TEST BUILD FAILED (check example.log for details)')
        return 1
      maker.link(executable, objects, linkLanguage)
    if not 'args' in param: param['args'] = ''
    param['args'] += extraArgs
    if maker.runTest(exampleDir, executable, testnum, replace, **param):
      print('TEST RUN FAILED (check example.log for details)')
      return 1
    rebuildTest = False
  if not args.retain and os.path.isdir(objDir): shutil.rmtree(objDir)
  return 0

def check(args):
  '''Check that build is functional'''
  ret       = 0
  extraArgs = ' '+' '.join(args.args)
  maker     = builder.PETScMaker('example.log')
  maker.setup()
  if 'regParams' in args and not args.regParams is None:
    mod = __import__(args.regParams)
    builder.localRegressionParameters[os.path.dirname(mod.__file__)] = mod.regressionParameters
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
  if not len(args.files):
    if hasattr(maker.configInfo.compilers, 'FC'):
      examples.append(os.path.join(maker.petscDir, 'src', 'snes', 'examples', 'tutorials', 'ex5f90t.F'))
      examples.append(os.path.join(maker.petscDir, 'src', 'snes', 'examples', 'tutorials', 'ex5f90.F'))
      examples.append(os.path.join(maker.petscDir, 'src', 'snes', 'examples', 'tutorials', 'ex5f.F'))
  for ex in examples:
    ret = checkSingleRun(maker, ex, args.replace, extraArgs)
    if ret: break
  if not ret:
    print('All tests pass')
  maker.cleanup()
  return ret

def regression(args):
  '''Run complete regression suite'''
  ret   = 0
  gret  = 0
  maker = builder.PETScMaker('regression.log')
  maker.setup()
  haltOnError = False

  args.retain  = False
  args.testnum = None
  if len(args.dirs):
    regdirs = args.dirs
  else:
    regdirs = map(lambda d: os.path.join('src', d), ['inline', 'sys', 'vec', 'mat', 'dm', 'ksp', 'snes', 'ts', 'docs', 'tops'])
  walker  = builder.DirectoryTreeWalker(maker.argDB, maker.log, maker.configInfo, allowExamples = True)
  dirs    = map(lambda d: os.path.join(maker.petscDir, d), regdirs)
  for d in dirs:
    for root, files in walker.walk(d):
      baseDir = os.path.basename(root)
      if not baseDir == 'tests' and not baseDir == 'tutorials': continue
      maker.logPrint('Running regression tests in %s\n' % (baseDir,), debugSection='screen', forceScroll=True)
      for f in files:
        basename, ext = os.path.splitext(f)
        if not basename.startswith('ex'): continue
        if not ext in ['.c', '.F']: continue
        ex  = os.path.join(root, f)
        ret = checkSingleRun(maker, ex, False, isRegression = True)
        if ret:
          gret = ret
          if haltOnError: break
      if ret and haltOnError: break
    if ret and haltOnError: break
  if not gret:
    maker.logPrint('All regression tests pass\n', debugSection='screen', forceScroll=True)
  maker.cleanup()
  return gret

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
    print(language)
    getattr(maker, 'build'+language.capitalize()+'Stubs')()
  maker.cleanup()
  return 0

def showSingleRun(maker, ex, extraArgs = ''):
  packageNames = set([p.name for p in maker.framework.packages])
  if isinstance(ex, list):
    exampleName = os.path.splitext(os.path.basename(ex[0]))[0]
    exampleDir  = os.path.dirname(ex[0])
  else:
    exampleName = os.path.splitext(os.path.basename(ex))[0]
    exampleDir  = os.path.dirname(ex)
  objDir        = maker.getObjDir(exampleName)
  executable    = os.path.join(objDir, exampleName)
  paramKey      = os.path.join(os.path.relpath(exampleDir, maker.petscDir), os.path.basename(executable))
  params = builder.regressionParameters.get(paramKey, {})
  if not params:
    params = builder.getRegressionParameters(maker, exampleDir).get(paramKey, {})
    maker.logPrint('Makefile params '+str(params))
  if not isinstance(params, list):
    params = [params]
  # Process testnum
  if args.testnum is not None:
    if args.testnum[0] == '[':
      validTestnum = args.testnum[1:-1].split(',')
    else:
      validTestnum = [args.testnum]
    numtests = len(validTestnum)
  else:
    numtests = len(params)
  maker.logPrint('Running %d tests\n' % (numtests,), debugSection='screen', forceScroll=True)
  for testnum, param in enumerate(params):
    testnum = str(testnum)
    if 'requires' in param:
      if not set(param['requires']).issubset(packageNames):
        maker.logPrint('Test %s requires packages %s\n' % (testnum, param['requires']), debugSection='screen', forceScroll=True)
        continue
    if 'num' in param: testnum = param['num']
    if not args.testnum is None and not testnum in validTestnum: continue
    if not 'args' in param: param['args'] = ''
    param['args'] += extraArgs
    print(str(testnum)+':  '+maker.getTestCommand(executable, **param))
  return 0

def show(args):
  '''Show run information'''
  ret       = 0
  extraArgs = ' '+' '.join(args.args)
  maker     = builder.PETScMaker()
  maker.setup()
  # C test
  examples = []
  for f in args.files:
    if f[0] == '[':
      examples.append(map(os.path.abspath, f[1:-1].split(',')))
    else:
      examples.append(os.path.abspath(f))
  for ex in examples:
    ret = showSingleRun(maker, ex, extraArgs)
    if ret: break
  maker.cleanup()
  return ret

if __name__ == '__main__':
  # Argumnt parsing
  import argparse

  parser = argparse.ArgumentParser(description     = 'PETSc BuildSystem',
                                   epilog          = 'For more information, visit http://www.mcs.anl.gov/petsc',
                                   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--version', action='version', version='PETSc BuildSystem 0.1')
  subparsers = parser.add_subparsers(help='build actions')

  parser_build = subparsers.add_parser('build', help='Compile all out of date source and generate a new shared library')
  parser_build.add_argument('files', nargs='*', help='Extra files to incorporate into the build')
  parser_build.add_argument('--parallel',   action='store_true',  default=False, help='Execute the build in parallel')
  parser_build.add_argument('--rebuildDep', action='store_false', default=True,  help='Rebuild dependencies')
  parser_build.set_defaults(func=build)
  parser_check = subparsers.add_parser('buildExample', help='Compile and link an example')
  parser_check.add_argument('files', nargs='+', help='The example files')
  parser_check.set_defaults(func=buildExample)
  parser_check = subparsers.add_parser('check', help='Check that build is functional')
  parser_check.add_argument('files', nargs='*', help='Extra examples to test')
  parser_check.add_argument('--args', action='append', default=[], help='Extra execution arguments for test')
  parser_check.add_argument('--retain', action='store_true', default=False, help='Retain the executable after testing')
  parser_check.add_argument('--testnum', help='The test to execute')
  parser_check.add_argument('--replace', action='store_true', default=False, help='Replace stored output with test output')
  parser_check.add_argument('--numProcs', help='The number of processes to use')
  parser_check.add_argument('--regParams', help='A module for regression parameters')
  parser_check.set_defaults(func=check)
  parser_regression = subparsers.add_parser('regression', help='Execute regression tests')
  parser_regression.add_argument('dirs', nargs='*', help='Directories for regression tests')
  parser_regression.set_defaults(func=regression)
  parser_clean = subparsers.add_parser('clean', help='Remove source database and all objects')
  parser_clean.set_defaults(func=clean)
  parser_purge = subparsers.add_parser('purge', help='Remove a set of files from the source database')
  parser_purge.add_argument('files', nargs='+', help='Files to remove from the source database')
  parser_purge.set_defaults(func=purge)
  parser_stubs = subparsers.add_parser('stubs', help='Build stubs for certain languages')
  parser_stubs.add_argument('languages', nargs='+', help='Stub languages')
  parser_stubs.set_defaults(func=stubs)
  parser_show = subparsers.add_parser('show', help='Print run information')
  parser_show.add_argument('files', nargs='+', help='Examples to display run info for')
  parser_show.add_argument('--args', action='append', default=[], help='Extra execution arguments for test')
  parser_show.add_argument('--testnum', type=int, help='The test number to execute')
  parser_show.set_defaults(func=show)

  args = parser.parse_args()
  print(args)
  args.func(args)
