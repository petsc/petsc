#!/usr/bin/env python

from __future__ import with_statement  # For python-2.5

import os
from collections import defaultdict, deque

# Run with PETSC_VERBOSE=1 to see files and directories that do not conform to the standard structure
VERBOSE = int(os.environ.get('PETSC_VERBOSE',0))

def cmakeconditional(key,val):
  def unexpected():
    raise RuntimeError('Unexpected')
  if key == 'package':
    return val
  if key == 'precision':
    if val == 'double':
      return 'PETSC_USE_SCALAR_DOUBLE'
    elif val == 'single':
      return 'PETSC_USE_SCALAR_SINGLE'
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
  raise RuntimeException('Unhandled case: %r=%r'%(key,val))

def pkgsources(pkg):
  '''
  Walks the source tree associated with 'pkg', analyzes the conditional written into the makefiles,
  and returns a list of sources associated with each unique conditional (as a dictionary).
  '''
  from distutils.sysconfig import parse_makefile
  autodirs = set('ftn-auto ftn-custom f90-custom'.split()) # Automatically recurse into these, if they exist
  skipdirs = set('examples benchmarks'.split())            # Skip these during the build
  def compareDirLists(mdirs,dirs):
    if not VERBOSE: return
    smdirs = set(mdirs)
    sdirs  = set(dirs).difference(autodirs)
    if smdirs != sdirs:
      from sys import stderr
      print >>stderr, 'Directory mismatch at %s:\n\tmdirs=%r\n\t dirs=%r\n\t  sym=%r' % (root,sorted(smdirs),sorted(sdirs),smdirs.symmetric_difference(sdirs))
  def compareSourceLists(msources, files):
    if not VERBOSE: return
    smsources = set(msources)
    ssources  = set(f for f in files if os.path.splitext(f)[1] in ['.c', '.cxx', '.cc', '.cpp', '.F'])
    if smsources != ssources:
      from sys import stderr
      print >>stderr, 'Source mismatch at %s:\n\tmsources=%r\n\t sources=%r\n\t  sym=%r' % (root,sorted(smsources),sorted(ssources),smsources.symmetric_difference(ssources))
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
endif ()
''' % dict(pkg=pkg, PKG=pkg.upper(), pkgdeps=' '.join('petsc%s'%p for p in pkgdeps)))

def main(petscdir):
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
                 ('characteristic' , 'dm vec sys'),
                 ('ksp'            , 'dm mat vec sys'),
                 ('snes'           , 'ksp dm mat vec sys'),
                 ('ts'             , 'snes ksp dm mat vec sys')]
      for pkg,deps in pkglist:
        writePackage(f,pkg,deps.split())
      f.write ('''
if (PETSC_USE_SINGLE_LIBRARY)
  add_library (petsc %s)
  target_link_libraries (petsc ${PETSC_PACKAGE_LIBS})
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

if __name__ == "__main__":
  main(petscdir=os.environ['PETSC_DIR'])
