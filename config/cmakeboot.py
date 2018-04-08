#!/usr/bin/env python

# This file initializes a CMake build in $PETSC_DIR/$PETSC_ARCH using
# the compilers and flags determined by BuildSystem. It is imported and
# called by Configure.py during configures in which CMake was detected,
# but it can also be run as a stand-alone program. The library paths and
# flags should have been written to
#
#     $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/PETScBuildInternal.cmake
#
# by configure before running this script.

from __future__ import print_function
import os,sys,string
from collections import deque
sys.path.insert(0, os.path.join(os.path.abspath('config')))
sys.path.insert(0, os.path.join(os.path.abspath('config'),'BuildSystem'))
import script

def noCheck(command, status, output, error):
  return

def quoteIfNeeded(path):
  "Don't need quotes unless the path has bits that would confuse the shell"
  safe = string.letters + string.digits + os.path.sep + os.path.pardir + '-_'
  if set(path).issubset(safe):
    return path
  else:
    return '"' + path + '"'

class StdoutLogger(object):
  def write(self,str):
    print(str)

class PETScMaker(script.Script):
 def __init__(self, petscdir, petscarch, argDB = None, framework = None):
   import RDict

   if not argDB:
     argDB = RDict.RDict(None, None, 0, 0, readonly = True)
     argDB.saveFilename = os.path.join(petscdir,petscarch,'lib','petsc','conf','RDict.db')
     argDB.load()
   script.Script.__init__(self, argDB = argDB)
   self.framework = framework

 def __str__(self):
   return ''

 def setupModules(self):
   self.mpi           = self.framework.require('config.packages.MPI',         None)
   self.base          = self.framework.require('config.base',                 None)
   self.setCompilers  = self.framework.require('config.setCompilers',         None)
   self.arch          = self.framework.require('PETSc.options.arch',        None)
   self.petscdir      = self.framework.require('PETSc.options.petscdir',    None)
   self.languages     = self.framework.require('PETSc.options.languages',   None)
   self.debugging     = self.framework.require('PETSc.options.debugging',   None)
   self.cmake         = self.framework.require('config.packages.cmake',       None)
   self.compilers     = self.framework.require('config.compilers',            None)
   self.types         = self.framework.require('config.types',                None)
   self.headers       = self.framework.require('config.headers',              None)
   self.functions     = self.framework.require('config.functions',            None)
   self.libraries     = self.framework.require('config.libraries',            None)
   self.scalarType    = self.framework.require('PETSc.options.scalarTypes', None)
   self.memAlign      = self.framework.require('PETSc.options.memAlign',    None)
   self.compilerFlags = self.framework.require('config.compilerFlags', self)
   return

 def setup(self):
   script.Script.setup(self)
   if not self.framework:
     self.framework = self.loadConfigure()
   self.setupModules()

 def cmakeboot(self, args, log):
   import shlex
   self.setup()
   options = deque()

   output,error,retcode = self.executeShellCommand([self.cmake.cmake, '--version'], checkCommand=noCheck, log=log)
   import re
   m = re.match(r'cmake version (.+)', output)
   if not m:
       self.logPrint('Could not parse CMake version: %s, disabling cmake build option' % output)
       return False
   from distutils.version import LooseVersion
   version = LooseVersion(m.groups()[0])
   if version < LooseVersion('2.6.2'):
       self.logPrint('CMake version %s < 2.6.2, disabling cmake build option' % version.vstring)
       return False
   if self.languages.clanguage == 'Cxx' and version < LooseVersion('2.8'):
       self.logPrint('Cannot use --with-clanguage=C++ with CMake version %s < 2.8, disabling cmake build option' % version.vstring)
       return False # no support for: set_source_files_properties(${file} PROPERTIES LANGUAGE CXX)

   langlist = [('C','C')]
   if hasattr(self.compilers,'FC'):
     langlist.append(('FC','Fortran'))
   if hasattr(self.compilers,'CUDAC'):
     langlist.append(('CUDA','CUDA'))
   if hasattr(self.compilers,'CXX'):
     langlist.append(('Cxx','CXX'))
   win32fe = None
   for petsclanguage,cmakelanguage in langlist:
     self.setCompilers.pushLanguage(petsclanguage)
     compiler = self.setCompilers.getCompiler()
     if (cmakelanguage == 'CUDA'):
       self.cuda = self.framework.require('config.packages.cuda',       None)
       if (self.cuda.directory != None):
         options.append('CUDA_TOOLKIT_ROOT_DIR ' + self.cuda.directory + ' CACHE FILEPATH')
       options.append('CUDA_NVCC_FLAGS ' + self.setCompilers.getCompilerFlags() + ' CACHE STRING')
     else:
       flags = [self.setCompilers.getCompilerFlags(),
                self.setCompilers.CPPFLAGS]
       if compiler.split()[0].endswith('win32fe'): # Hack to support win32fe without changing the rest of configure
         win32fe = compiler.split()[0] + '.exe'
         compiler = ' '.join(compiler.split()[1:])
       options.append('CMAKE_'+cmakelanguage+'_COMPILER ' + compiler + ' CACHE FILEPATH')
       options.append('CMAKE_'+cmakelanguage+'_FLAGS "' + ''.join(flags) + '" CACHE STRING')
       if (petsclanguage == self.languages.clanguage): #CUDA host compiler is fed with the flags for the standard host compiler
         flagstring = ''
         for flag in flags:
           for f in flag.split():
             flagstring += ',' + f
         options.append('PETSC_CUDA_HOST_FLAGS ' + flagstring + ' CACHE STRING')
       self.setCompilers.popLanguage()
   options.append('CMAKE_AR '+self.setCompilers.AR + " CACHE FILEPATH")
   ranlib = shlex.split(self.setCompilers.RANLIB)[0]
   options.append('CMAKE_RANLIB '+ranlib + " CACHE FILEPATH")
   if win32fe:
     options.append('PETSC_WIN32FE %s' % win32fe)
     
   archdir = os.path.join(self.petscdir.dir, self.arch.arch)
   initial_cache_filename = os.path.join(archdir, 'initial_cache_file.cmake')  
   cmd = [self.cmake.cmake, '--trace', '--debug-output', '-C' + str(initial_cache_filename), '-DPETSC_CMAKE_ARCH:STRING='+str(self.arch.arch), self.petscdir.dir] + args
   if win32fe:
     # Default on Windows is to generate Visual Studio project files, but
     # 1. the build process for those is different, need to give different build instructions
     # 2. the current WIN32FE workaround does not work with VS project files
     cmd.append('-GUnix Makefiles')

   # Create inital cache file:
   initial_cache_file = open(initial_cache_filename, 'w')
   self.logPrint('Contents of initial cache file %s :' % initial_cache_filename)
   for option in options:
     initial_cache_file.write('SET (' + option + ' "Dummy comment" FORCE)\n')
     self.logPrint('SET (' + option + ' "Dummy comment" FORCE)\n')
   initial_cache_file.close()   
   try:
     # Try to remove the old cache because some versions of CMake lose CMAKE_C_FLAGS when reconfiguring this way
     self.logPrint('Removing: %s' % os.path.join(archdir, 'CMakeCache.txt'))
     os.remove(os.path.join(archdir, 'CMakeCache.txt'))
   except OSError:
     pass
   import shutil
   # Try to remove all the old CMake files to avoid infinite loop (CMake-2.8.10.2, maybe other versions)
   # http://www.mail-archive.com/cmake@cmake.org/msg44765.html
   self.logPrint('Removing: %s' % os.path.join(archdir, 'CMakeFiles', version.vstring))
   shutil.rmtree(os.path.join(archdir, 'CMakeFiles', version.vstring), ignore_errors=True)
   log.write('Invoking: %s\n' % cmd)
   output,error,retcode = self.executeShellCommand(cmd, checkCommand = noCheck, log=log, cwd=archdir,timeout=300)
   if retcode:
     self.logPrint('CMake setup incomplete (status %d), disabling cmake build option' % (retcode,))
     self.logPrint('Output: '+output+'\nError: '+error)
     cachetxt = os.path.join(archdir, 'CMakeCache.txt')
     try:
       f = open(cachetxt, 'r')
       log.write('Contents of %s:\n' % cachetxt)
       log.write(f.read())
       f.close()
     except IOError as e:
       log.write('Could not read file %s: %r\n' % (cachetxt, e))
     return False
   else:
     return True # Configure successful

def main(petscdir, petscarch, argDB=None, framework=None, log=StdoutLogger(), args=[]):
  # This can be called as a stand-alone program, or by importing it from
  # python.  The latter functionality is needed because argDB does not
  # get written until the very end of configure, but we want to run this
  # automatically during configure (if CMake is available).
  #
  # Strangely, we can't store log in the PETScMaker because
  # (somewhere) it creeps into framework (which I don't want to modify)
  # and makes the result unpickleable.  This is not a problem when run
  # as a standalone program (because the database is read-only), but is
  # not okay when called from configure.
  return PETScMaker(petscdir,petscarch,argDB,framework).cmakeboot(args,log)

if __name__ == "__main__":
  main(petscdir=os.environ['PETSC_DIR'], petscarch=os.environ['PETSC_ARCH'], args=sys.argv[1:])
