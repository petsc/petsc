#!/usr/bin/env python

from __future__ import with_statement  # For python-2.5

import os
from collections import defaultdict, deque

def cmakeconditional(key,val):
  def unexpected():
    raise RuntimeError('Unexpected')
  if key == 'package':
    return val
  if key == 'precision':
    if val == 'double':
      return 'PETSC_USE_SCALAR_DOUBLE'
    raise RuntimeError('Unexpected precision: %r'%val)
  if key == 'scalar':
    if val == 'real':
      return 'NOT PETSC_USE_COMPLEX'
    if val == 'complex':
      return 'PETSC_USE_COMPLEX'
    raise RuntimeError('Unexpected scalar: %r'%val)
  if key == 'language':
    if val == 'CXXONLY':
      return 'PETSC_CLANGUAGE_CXX'
    if val == 'CONLY':
      return 'PETSC_CLANGUAGE_C'
    raise RuntimeError('Unexpected language: %r'%val)
  raise RuntimeException('Unhandled case: %r=%r'%(key,val))

def pkgsources(pkg):
  '''
  Walks the source tree associated with 'pkg', analyzes the conditional written into the makefiles,
  and returns a list of sources associated with each unique conditional (as a dictionary).
  '''
  allconditions = defaultdict(set)
  sources = defaultdict(deque)
  for root,dirs,files in os.walk(os.path.join('src',pkg)):
    conditions = allconditions[os.path.dirname(root)].copy()
    dirs[:] = [dir for dir in dirs if dir not in 'examples benchmarks'.split()]
    makefile = os.path.join(root,'makefile')
    if not os.path.exists(makefile):
      continue
    with open(makefile) as lines:
      def stripsplit(line):
        return filter(lambda c: c!="'", line[len('#requires'):]).split()
      conditions.update(set(tuple(stripsplit(line)) for line in lines if line.startswith('#requires')))
    def relpath(filename):
      return os.path.join(root,filename)
    sources[repr(sorted(conditions))].extend(relpath(f) for f in files if os.path.splitext(f)[1] in ['.c', '.cxx', '.F'])
    allconditions[root] = conditions
  return sources

def writeRoot(f):
  f.write(r'''cmake_minimum_required (VERSION 2.6)
project (PETSc C)

include (${PETSc_BINARY_DIR}/conf/PETScConfig.cmake)

if (PETSC_HAVE_FORTRAN)
  enable_language (Fortran)
endif ()
if (PETSC_CLANGUAGE_CXX)
  enable_language (CXX)
endif ()
if (PETSC_USE_DEBUG)
  set (CMAKE_BUILD_TYPE "Debug")
endif ()

include_directories ("${PETSc_SOURCE_DIR}/include" "${PETSc_BINARY_DIR}/include")

set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PETSc_BINARY_DIR}/lib CACHE PATH "Output directory for PETSc archives")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PETSc_BINARY_DIR}/lib CACHE PATH "Output directory for PETSc libraries")
mark_as_advanced (CMAKE_ARCHIVE_OUTPUT_DIRECTORY CMAKE_LIBRARY_OUTPUT_DIRECTORY)
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

###################  The following describes the build  ####################
  
''')

def writePackage(f,pkg):
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
  target_link_libraries (petsc%(pkg)s ${PETSC_PACKAGE_LIBS})
endif ()
''' % dict(pkg=pkg, PKG=pkg.upper()))

def main():
  os.chdir(os.environ['PETSC_DIR'])
  with open('CMakeLists.txt', 'w') as f:
    writeRoot(f)
    f.write('include_directories (${PETSC_PACKAGE_INCLUDES})\n')
    pkglist = 'sys vec mat dm ksp snes ts characteristic'.split()
    for pkg in pkglist:
      writePackage(f,pkg)
    f.write ('''
if (PETSC_USE_SINGLE_LIBRARY)
  add_library (petsc %s)
  target_link_libraries (petsc ${PETSC_PACKAGE_LIBS})
endif ()
''' % (' '.join([r'${PETSC' + pkg.upper() + r'_SRCS}' for pkg in pkglist]),))

if __name__ == "__main__":
  main()
