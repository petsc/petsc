#!/usr/bin/env python

# This file generates $PETSC_DIR/CMakeLists.txt by parsing the makefiles
# throughout the source tree, reading their constraints and included
# sources, and encoding the rules through CMake conditionals. When CMake
# runs, it will use the conditionals written to
#
#     $PETSC_DIR/$PETSC_ARCH/conf/PETScConfig.cmake
#
# by BuildSystem after a successful configure.
#
# The generated CMakeLists.txt is independent of PETSC_ARCH.
#
# This script supports one option:
#   --verbose : Show mismatches between makefiles and the filesystem

from __future__ import with_statement  # For python-2.5

import os
from collections import defaultdict, deque

# Run with --verbose
VERBOSE = False
MISTAKES = []

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

def pkgsources(pkg):
  '''
  Walks the source tree associated with 'pkg', analyzes the conditional written into the makefiles,
  and returns a list of sources associated with each unique conditional (as a dictionary).
  '''
  from distutils.sysconfig import parse_makefile
  autodirs = set('ftn-auto ftn-custom f90-custom'.split()) # Automatically recurse into these, if they exist
  skipdirs = set('examples benchmarks'.split())            # Skip these during the build
  def compareDirLists(mdirs,dirs):
    smdirs = set(mdirs)
    sdirs  = set(dirs).difference(autodirs)
    if not smdirs.issubset(sdirs):
      MISTAKES.append('Makefile contains directory not on filesystem: %s: %r' % (root, sorted(smdirs - sdirs)))
    if not VERBOSE: return
    if smdirs != sdirs:
      from sys import stderr
      print >>stderr, ('Directory mismatch at %s:\n\t%s: %r\n\t%s: %r\n\t%s: %r'
                       % (root,
                          'in makefile   ',sorted(smdirs),
                          'on filesystem ',sorted(sdirs),
                          'symmetric diff',sorted(smdirs.symmetric_difference(sdirs))))
  def compareSourceLists(msources, files):
    smsources = set(msources)
    ssources  = set(f for f in files if os.path.splitext(f)[1] in ['.c', '.cxx', '.cc', '.cpp', '.F'])
    if not smsources.issubset(ssources):
      MISTAKES.append('Makefile contains file not on filesystem: %s: %r' % (root, sorted(smsources - ssources)))
    if not VERBOSE: return
    if smsources != ssources:
      from sys import stderr
      print >>stderr, ('Source mismatch at %s:\n\t%s: %r\n\t%s: %r\n\t%s: %r'
                       % (root,
                          'in makefile   ',sorted(smsources),
                          'on filesystem ',sorted(ssources),
                          'symmetric diff',sorted(smsources.symmetric_difference(ssources))))
  allconditions = defaultdict(set)
  sources = defaultdict(deque)
  for root,dirs,files in os.walk(os.path.join('src',pkg)):
    conditions = allconditions[os.path.dirname(root)].copy()
    makefile = os.path.join(root,'makefile')
    if not os.path.exists(makefile):
      continue
    makevars = parse_makefile(makefile)
    mdirs = makevars.get('DIRS','').split() # Directories specified in the makefile
    compareDirLists(mdirs,dirs) # diagnostic output to find unused directories
    candidates = set(mdirs).union(autodirs).difference(skipdirs)
    dirs[:] = list(candidates.intersection(dirs))
    with open(makefile) as lines:
      def stripsplit(line):
        return filter(lambda c: c!="'", line[len('#requires'):]).split()
      conditions.update(set(tuple(stripsplit(line)) for line in lines if line.startswith('#requires')))
    def relpath(filename):
      return os.path.join(root,filename)
    sourcec = makevars.get('SOURCEC','').split()
    sourcef = makevars.get('SOURCEF','').split()
    compareSourceLists(sourcec+sourcef, files) # Diagnostic output about unused source files
    sources[repr(sorted(conditions))].extend(relpath(f) for f in sourcec + sourcef)
    allconditions[root] = conditions
  return sources

def writeRoot(f):
  f.write(r'''cmake_minimum_required (VERSION 2.6.2)
project (PETSc C)

include (${PETSc_BINARY_DIR}/conf/PETScConfig.cmake)

if (PETSC_HAVE_FORTRAN)
  enable_language (Fortran)
endif ()
if (PETSC_CLANGUAGE_Cxx)
  enable_language (CXX)
endif ()

include_directories ("${PETSc_SOURCE_DIR}/include" "${PETSc_BINARY_DIR}/include")

add_definitions (-D__INSDIR__= ) # CMake always uses the absolute path
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PETSc_BINARY_DIR}/lib" CACHE PATH "Output directory for PETSc archives")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PETSc_BINARY_DIR}/lib" CACHE PATH "Output directory for PETSc libraries")
set (CMAKE_Fortran_MODULE_DIRECTORY "${PETSc_BINARY_DIR}/include" CACHE PATH "Output directory for fortran *.mod files")
mark_as_advanced (CMAKE_ARCHIVE_OUTPUT_DIRECTORY CMAKE_LIBRARY_OUTPUT_DIRECTORY CMAKE_Fortran_MODULE_DIRECTORY)
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

###################  The following describes the build  ####################
  
''')

def writePackage(f,pkg,pkgdeps):
  for conds, srcs in pkgsources(pkg).items():
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
  add_library (petsc%(pkg)s ${PETSC%(PKG)s_SRCS})
  target_link_libraries (petsc%(pkg)s %(pkgdeps)s ${PETSC_PACKAGE_LIBS})
  if (PETSC_WIN32FE)
    set_target_properties (petsc%(pkg)s PROPERTIES RULE_LAUNCH_COMPILE "${PETSC_WIN32FE}")
    set_target_properties (petsc%(pkg)s PROPERTIES RULE_LAUNCH_LINK "${PETSC_WIN32FE}")
  endif ()
endif ()
''' % dict(pkg=pkg, PKG=pkg.upper(), pkgdeps=' '.join('petsc%s'%p for p in pkgdeps)))

def main(petscdir, log=StdoutLogger()):
  import tempfile, shutil
  written = False               # We delete the temporary file if it wasn't finished, otherwise rename (atomic)
  fd,tmplists = tempfile.mkstemp(prefix='CMakeLists.txt.',dir=petscdir,text=True)
  try:
    with os.fdopen(fd,'w') as f:
      writeRoot(f)
      f.write('include_directories (${PETSC_PACKAGE_INCLUDES})\n')
      pkglist = [('sys'            , ''),
                 ('vec'            , 'sys'),
                 ('mat'            , 'vec sys'),
                 ('dm'             , 'mat vec sys'),
                 ('ksp'            , 'dm mat vec sys'),
                 ('snes'           , 'ksp dm mat vec sys'),
                 ('ts'             , 'snes ksp dm mat vec sys')]
      for pkg,deps in pkglist:
        writePackage(f,pkg,deps.split())
      f.write ('''
if (PETSC_USE_SINGLE_LIBRARY)
  add_library (petsc %s)
  target_link_libraries (petsc ${PETSC_PACKAGE_LIBS})
  if (PETSC_WIN32FE)
    set_target_properties (petsc PROPERTIES RULE_LAUNCH_COMPILE "${PETSC_WIN32FE}")
    set_target_properties (petsc PROPERTIES RULE_LAUNCH_LINK "${PETSC_WIN32FE}")
  endif ()

endif ()
''' % (' '.join([r'${PETSC' + pkg.upper() + r'_SRCS}' for pkg,deps in pkglist]),))
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
    if written:
      shutil.move(tmplists,os.path.join(petscdir,'CMakeLists.txt'))
    else:
      os.remove(tmplists)
  if MISTAKES:
    for m in MISTAKES:
      log.write(m + '\n')
    raise RuntimeError('PETSc makefiles contain mistakes or files are missing on filesystem.\n%s\nPossible reasons:\n\t1. Files were deleted locally, try "hg update".\n\t2. Files were deleted from mercurial, but were not removed from makefile. Send mail to petsc-maint@mcs.anl.gov.\n\t3. Someone forgot "hg add" new files. Send mail to petsc-maint@mcs.anl.gov.' % ('\n'.join(MISTAKES)))

if __name__ == "__main__":
  import optparse
  parser = optparse.OptionParser()
  parser.add_option('--verbose', help='Show mismatches between makefiles and the filesystem', dest='verbose', action='store_true', default=False)
  (opts, extra_args) = parser.parse_args()
  if opts.verbose:
    VERBOSE = True
  main(petscdir=os.environ['PETSC_DIR'])
