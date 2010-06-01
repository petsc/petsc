#!/usr/bin/env python

import os,sys,subprocess,string
from collections import deque
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config'))
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))
import script

class PETScMaker(script.Script):
 def __init__(self):
   import RDict

   argDB = RDict.RDict(None, None, 0, 0, readonly = True)
   argDB.saveFilename = os.path.join(os.environ['PETSC_DIR'], os.environ['PETSC_ARCH'], 'conf', 'RDict.db')
   argDB.load()
   script.Script.__init__(self, argDB = argDB)
   self.log = sys.stdout
   return  

 def setupModules(self):
   self.mpi           = self.framework.require('config.packages.MPI',         None)
   self.base          = self.framework.require('config.base',                 None)
   self.setCompilers  = self.framework.require('config.setCompilers',         None)   
   self.arch          = self.framework.require('PETSc.utilities.arch',        None)
   self.petscdir      = self.framework.require('PETSc.utilities.petscdir',    None)
   self.languages     = self.framework.require('PETSc.utilities.languages',   None)
   self.debugging     = self.framework.require('PETSc.utilities.debugging',   None)
   self.make          = self.framework.require('PETSc.utilities.Make',        None)
   self.CHUD          = self.framework.require('PETSc.utilities.CHUD',        None)
   self.compilers     = self.framework.require('config.compilers',            None)
   self.types         = self.framework.require('config.types',                None)
   self.headers       = self.framework.require('config.headers',              None)
   self.functions     = self.framework.require('config.functions',            None)
   self.libraries     = self.framework.require('config.libraries',            None)
   self.scalarType    = self.framework.require('PETSc.utilities.scalarTypes', None)
   self.memAlign      = self.framework.require('PETSc.utilities.memAlign',    None)
   self.libraryOptions= self.framework.require('PETSc.utilities.libraryOptions', None)      
   self.compilerFlags = self.framework.require('config.compilerFlags', self)
   return

 def setup(self):
   script.Script.setup(self)
   self.framework = self.loadConfigure()
   self.setupModules()

 def cmakeboot(self):
   self.setup()
   petscdir  = os.environ['PETSC_DIR']
   petscarch = os.environ['PETSC_ARCH']
   os.chdir(os.path.join(petscdir,petscarch))
   options = deque()
   langlist = [('C','C')]
   if hasattr(self.compilers,'FC'):
     langlist.append(('FC','Fortran'))
   if (self.languages.clanguage == 'Cxx'):
     langlist.append(('Cxx','CXX'))
   for petsclanguage,cmakelanguage in langlist:
     self.setCompilers.pushLanguage(petsclanguage)
     options.append('-DCMAKE_'+cmakelanguage+'_COMPILER=' + self.compilers.getCompiler())
     flags = [self.setCompilers.getCompilerFlags(),
              self.setCompilers.CPPFLAGS,
              self.CHUD.CPPFLAGS]
     options.append('-DCMAKE_'+cmakelanguage+'_FLAGS=' + ''.join(flags))
     options.append('-DCMAKE_'+cmakelanguage+'_COMPILER=' + self.setCompilers.getCompiler())
     self.setCompilers.popLanguage()
   cmd = ['cmake', petscdir] + map(lambda x:x.strip(), options) + sys.argv[1:]
   print 'Invoking: ', cmd
   retcode = subprocess.call(cmd)
   if retcode < 0:
     print >>sys.stderr, "CMake process was terminated by signal", -retcode
     sys.exit(-retcode)
   if retcode > 0:
     print >>sys.stderr, "CMake process failed with status", retcode
     sys.exit(retcode)
   print('CMake configuration completed successfully.')
   def quoteIfNeeded(path):
     "Don't need quotes unless the path has bits that would confuse the shell"
     safe = string.letters + string.digits + os.path.sep + os.path.pardir + '-_'
     if set(path).issubset(safe):
       return path
     else:
       return '"' + path + '"'
   print('Build the library with: make -C %s' % quoteIfNeeded(os.path.join(petscdir,petscarch)))

if __name__ == "__main__":
  PETScMaker().cmakeboot()
