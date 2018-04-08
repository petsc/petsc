#!/usr/bin/env python

# This file generates $PETSC_DIR/CMakeLists.txt by parsing the makefiles
# throughout the source tree, reading their constraints and included
# sources, and encoding the rules through CMake conditionals. When CMake
# runs, it will use the conditionals written to
#
#     $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/PETScBuildInternal.cmake
#
# by BuildSystem after a successful configure.
#
# The generated CMakeLists.txt is independent of PETSC_ARCH.
#
# This script supports one option:
#   --verbose : Show mismatches between makefiles and the filesystem

from __future__ import print_function
import os
from collections import deque

# compatibility code for python-2.4 from http://code.activestate.com/recipes/523034-emulate-collectionsdefaultdict/
try:
    from collections import defaultdict
except:
    class defaultdict(dict):
        def __init__(self, default_factory=None, *a, **kw):
            if (default_factory is not None and
                not hasattr(default_factory, '__call__')):
                raise TypeError('first argument must be callable')
            dict.__init__(self, *a, **kw)
            self.default_factory = default_factory
        def __getitem__(self, key):
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                return self.__missing__(key)
        def __missing__(self, key):
            if self.default_factory is None:
                raise KeyError(key)
            self[key] = value = self.default_factory()
            return value
        def __reduce__(self):
            if self.default_factory is None:
                args = tuple()
            else:
                args = self.default_factory,
            return type(self), args, None, None, self.items()
        def copy(self):
            return self.__copy__()
        def __copy__(self):
            return type(self)(self.default_factory, self)
        def __deepcopy__(self, memo):
            import copy
            return type(self)(self.default_factory,
                              copy.deepcopy(self.items()))
        def __repr__(self):
            return 'defaultdict(%s, %s)' % (self.default_factory,
                                            dict.__repr__(self))

class StdoutLogger(object):
  def write(self,str):
    print(str)

def cmakeconditional(key,val):
  def unexpected():
    raise RuntimeError('Unexpected')
  if key in ['package', 'function', 'define']:
    return val
  if key == 'precision':
    if val == 'double':
      return 'PETSC_USE_REAL_DOUBLE'
    elif val == 'single':
      return 'PETSC_USE_REAL_SINGLE'
    raise RuntimeError('Unexpected precision: %r'%val)
  if key == 'scalar':
    if val == 'real':
      return 'NOT PETSC_USE_COMPLEX'
    if val == 'complex':
      return 'PETSC_USE_COMPLEX'
    raise RuntimeError('Unexpected scalar: %r'%val)
  if key == 'language':
    if val == 'CXXONLY':
      return 'PETSC_CLANGUAGE_Cxx'
    if val == 'CONLY':
      return 'PETSC_CLANGUAGE_C'
    raise RuntimeError('Unexpected language: %r'%val)
  raise RuntimeError('Unhandled case: %r=%r'%(key,val))

AUTODIRS = set('ftn-auto ftn-custom f90-custom'.split()) # Automatically recurse into these, if they exist
SKIPDIRS = set('benchmarks'.split())                     # Skip these during the build
NOWARNDIRS = set('tests tutorials'.split())              # Do not warn about mismatch in these

def pathsplit(path):
    """Recursively split a path, returns a tuple"""
    stem, basename = os.path.split(path)
    if stem == '':
        return (basename,)
    if stem == path:            # fixed point, likely '/'
        return (path,)
    return pathsplit(stem) + (basename,)

class Mistakes(object):
    def __init__(self, log, verbose=False):
        self.mistakes = []
        self.verbose = verbose
        self.log = log

    def compareDirLists(self,root, mdirs, dirs):
        if NOWARNDIRS.intersection(pathsplit(root)):
            return
        smdirs = set(mdirs)
        sdirs  = set(dirs).difference(AUTODIRS)
        if not smdirs.issubset(sdirs):
            self.mistakes.append('Makefile contains directory not on filesystem: %s: %r' % (root, sorted(smdirs - sdirs)))
        if not self.verbose: return
        if smdirs != sdirs:
            from sys import stderr
            stderr.write('Directory mismatch at %s:\n\t%s: %r\n\t%s: %r\n\t%s: %r\n'
                         % (root,
                            'in makefile   ',sorted(smdirs),
                            'on filesystem ',sorted(sdirs),
                            'symmetric diff',sorted(smdirs.symmetric_difference(sdirs))))

    def compareSourceLists(self, root, msources, files):
        if NOWARNDIRS.intersection(pathsplit(root)):
            return
        smsources = set(msources)
        ssources  = set(f for f in files if os.path.splitext(f)[1] in ['.c', '.cxx', '.cc', '.cu', '.cpp', '.F'])
        if not smsources.issubset(ssources):
            self.mistakes.append('Makefile contains file not on filesystem: %s: %r' % (root, sorted(smsources - ssources)))
        if not self.verbose: return
        if smsources != ssources:
            from sys import stderr
            stderr.write('Source mismatch at %s:\n\t%s: %r\n\t%s: %r\n\t%s: %r\n'
                         % (root,
                            'in makefile   ',sorted(smsources),
                            'on filesystem ',sorted(ssources),
                            'symmetric diff',sorted(smsources.symmetric_difference(ssources))))

    def summary(self):
        for m in self.mistakes:
            self.log.write(m + '\n')
        if self.mistakes:
            raise RuntimeError('PETSc makefiles contain mistakes or files are missing on filesystem.\n%s\nPossible reasons:\n\t1. Files were deleted locally, try "hg revert filename" or "git checkout filename".\n\t2. Files were deleted from repository, but were not removed from makefile. Send mail to petsc-maint@mcs.anl.gov.\n\t3. Someone forgot to "add" new files to the repository. Send mail to petsc-maint@mcs.anl.gov.' % ('\n'.join(self.mistakes)))

def stripsplit(line):
  return line[len('#requires'):].replace("'","").split()

def pkgsources(pkg, mistakes):
  '''
  Walks the source tree associated with 'pkg', analyzes the conditional written into the makefiles,
  and returns a list of sources associated with each unique conditional (as a dictionary).
  '''
  from distutils.sysconfig import parse_makefile
  allconditions = defaultdict(set)
  sources = defaultdict(deque)
  for root,dirs,files in os.walk(os.path.join('src',pkg)):
    dirs.sort()
    files.sort()
    conditions = allconditions[os.path.dirname(root)].copy()
    makefile = os.path.join(root,'makefile')
    if not os.path.exists(makefile):
      continue
    makevars = parse_makefile(makefile)
    mdirs = makevars.get('DIRS','').split() # Directories specified in the makefile
    mistakes.compareDirLists(root,mdirs,dirs) # diagnostic output to find unused directories
    candidates = set(mdirs).union(AUTODIRS).difference(SKIPDIRS)
    dirs[:] = list(candidates.intersection(dirs))
    lines = open(makefile)
    conditions.update(set(tuple(stripsplit(line)) for line in lines if line.startswith('#requires')))
    lines.close()
    def relpath(filename):
      return os.path.join(root,filename)
    sourcecu = makevars.get('SOURCECU','').split()
    sourcec = makevars.get('SOURCEC','').split()
    sourcecxx = makevars.get('SOURCECXX','').split()
    sourcef = makevars.get('SOURCEF','').split()
    mistakes.compareSourceLists(root,sourcec+sourcecxx+sourcef+sourcecu, files) # Diagnostic output about unused source files
    sources[repr(sorted(conditions))].extend(relpath(f) for f in sourcec + sourcecxx + sourcef + sourcecu)
    allconditions[root] = conditions
  return sources

def writeRoot(f):
  f.write(r'''cmake_minimum_required (VERSION 2.6.2)
project (PETSc C)

include (${PETSC_CMAKE_ARCH}/lib/petsc/conf/PETScBuildInternal.cmake)

if (PETSC_HAVE_FORTRAN)
  enable_language (Fortran)
endif ()
if (PETSC_CLANGUAGE_Cxx OR PETSC_HAVE_CXX)
  enable_language (CXX)
endif ()

if (APPLE)
  SET(CMAKE_C_ARCHIVE_FINISH "<CMAKE_RANLIB> -c <TARGET> ")
  SET(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -c <TARGET> ")
  SET(CMAKE_Fortran_ARCHIVE_FINISH "<CMAKE_RANLIB> -c <TARGET> ")
endif ()

if (PETSC_HAVE_CUDA)
  find_package (CUDA REQUIRED)
  set (CUDA_PROPAGATE_HOST_FLAGS OFF)
  set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options ${PETSC_CUDA_HOST_FLAGS})
endif ()

include_directories ("${PETSc_SOURCE_DIR}/include" "${PETSc_BINARY_DIR}/include")

set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PETSc_BINARY_DIR}/lib" CACHE PATH "Output directory for PETSc archives")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PETSc_BINARY_DIR}/lib" CACHE PATH "Output directory for PETSc libraries")
set (CMAKE_Fortran_MODULE_DIRECTORY "${PETSc_BINARY_DIR}/include" CACHE PATH "Output directory for fortran *.mod files")
mark_as_advanced (CMAKE_ARCHIVE_OUTPUT_DIRECTORY CMAKE_LIBRARY_OUTPUT_DIRECTORY CMAKE_Fortran_MODULE_DIRECTORY)
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

###################  The following describes the build  ####################

''')

def writePackage(f,pkg,pkgdeps,mistakes):
  for conds, srcs in pkgsources(pkg,mistakes).items():
    conds = eval(conds)
    def body(indentlevel):
      indent = ' '*(indentlevel+2)
      lfindent = '\n'+indent
      return ' '*indentlevel + 'list (APPEND PETSC' + pkg.upper() + '_SRCS' + lfindent + lfindent.join(srcs) + lfindent + ')\n'
    if conds:
      f.write('if (%s)\n%sendif ()\n' % (' AND '.join(cmakeconditional(*c) for c in conds), body(2)))
    else:
      f.write(body(0))
  f.write('''
if (NOT PETSC_USE_SINGLE_LIBRARY)
  if (PETSC_HAVE_CUDA)
    cuda_add_library (petsc%(pkg)s ${PETSC%(PKG)s_SRCS})
  else ()
    add_library (petsc%(pkg)s ${PETSC%(PKG)s_SRCS})
  endif ()
  target_link_libraries (petsc%(pkg)s %(pkgdeps)s ${PETSC_PACKAGE_LIBS})
  if (PETSC_WIN32FE)
    set_target_properties (petsc%(pkg)s PROPERTIES RULE_LAUNCH_COMPILE "${PETSC_WIN32FE}")
    set_target_properties (petsc%(pkg)s PROPERTIES RULE_LAUNCH_LINK "${PETSC_WIN32FE}")
  endif ()
endif ()
''' % dict(pkg=pkg, PKG=pkg.upper(), pkgdeps=' '.join('petsc%s'%p for p in pkgdeps)))

def main(petscdir, log=StdoutLogger(), verbose=False):
  import tempfile, shutil
  written = False               # We delete the temporary file if it wasn't finished, otherwise rename (atomic)
  mistakes = Mistakes(log=log, verbose=verbose)
  fd,tmplists = tempfile.mkstemp(prefix='CMakeLists.txt.',dir=petscdir,text=True)
  try:
    f = os.fdopen(fd,'w')
    writeRoot(f)
    f.write('include_directories (${PETSC_PACKAGE_INCLUDES})\n')
    pkglist = [('sys'            , ''),
               ('vec'            , 'sys'),
               ('mat'            , 'vec sys'),
               ('dm'             , 'mat vec sys'),
               ('ksp'            , 'dm mat vec sys'),
               ('snes'           , 'ksp dm mat vec sys'),
               ('ts'             , 'snes ksp dm mat vec sys'),
               ('tao'            , 'snes ksp dm mat vec sys')]
    for pkg,deps in pkglist:
      writePackage(f,pkg,deps.split(),mistakes)
    f.write ('''
if (PETSC_USE_SINGLE_LIBRARY)
  if (PETSC_HAVE_CUDA)
    cuda_add_library (petsc %(allsrc)s)
  else ()
    add_library (petsc %(allsrc)s)
  endif ()
  target_link_libraries (petsc ${PETSC_PACKAGE_LIBS})
  if (PETSC_WIN32FE)
    set_target_properties (petsc PROPERTIES RULE_LAUNCH_COMPILE "${PETSC_WIN32FE}")
    set_target_properties (petsc PROPERTIES RULE_LAUNCH_LINK "${PETSC_WIN32FE}")
  endif ()

endif ()
''' % dict(allsrc=' '.join([r'${PETSC' + pkg.upper() + r'_SRCS}' for pkg,deps in pkglist])))
    f.write('''
if (PETSC_CLANGUAGE_Cxx)
  foreach (file IN LISTS %s)
    if (file MATCHES "^.*\\\\.c$")
      set_source_files_properties(${file} PROPERTIES LANGUAGE CXX)
    endif ()
  endforeach ()
endif()''' % ('\n  '.join([r'PETSC' + pkg.upper() + r'_SRCS' for (pkg,_) in pkglist])))
    written = True
  finally:
    f.close()
    if written:
      shutil.move(tmplists,os.path.join(petscdir,'CMakeLists.txt'))
    else:
      os.remove(tmplists)
  mistakes.summary()

if __name__ == "__main__":
  import optparse
  parser = optparse.OptionParser()
  parser.add_option('--verbose', help='Show mismatches between makefiles and the filesystem', dest='verbose', action='store_true', default=False)
  (opts, extra_args) = parser.parse_args()
  main(petscdir=os.environ['PETSC_DIR'], verbose=opts.verbose)
