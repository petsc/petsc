#!/usr/bin/env python
import os, sys

sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config'))
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))

import script

class PETScMaker(script.Script):
 def __init__(self):
   import RDict
   import os

   argDB = RDict.RDict(None, None, 0, 0)
   argDB.saveFilename = os.path.join(os.environ['PETSC_DIR'], os.environ['PETSC_ARCH'], 'conf', 'RDict.db')
   argDB.load()
   script.Script.__init__(self, argDB = argDB)
   self.debug = 1
   self.log = sys.stdout
   return

 def setupModules(self):
#    self.mpi           = self.framework.require('config.packages.MPI',      None)
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
   self.libraryOptions= self.framework.require('PETSc.utilities.libraryOptions', None)      
   return

 def setup(self):
   script.Script.setup(self)
   self.framework = self.loadConfigure()
   self.setupModules()
   return

 def compileC(self, source):
   '''PETSC_INCLUDE	        = -I${PETSC_DIR}/${PETSC_ARCH}/include -I${PETSC_DIR}/include
                              ${PACKAGES_INCLUDES} ${TAU_DEFS} ${TAU_INCLUDE}
      PETSC_CC_INCLUDES     = ${PETSC_INCLUDE}
      PETSC_CCPPFLAGS	    = ${PETSC_CC_INCLUDES} ${PETSCFLAGS} ${CPP_FLAGS} ${CPPFLAGS}  -D__SDIR__='"${LOCDIR}"'
      CCPPFLAGS	            = ${PETSC_CCPPFLAGS}
      PETSC_COMPILE         = ${PCC} -c ${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}  ${SOURCEC} ${SSOURCE}
      PETSC_COMPILE_SINGLE  = ${PCC} -o $*.o -c ${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}'''
   # PETSCFLAGS, CFLAGS and CPPFLAGS are taken from user input (or empty)
   flags           = []
   packageIncludes = []
   packageLibs     = []
   for p in self.framework.packages:
     # Could put on compile line, self.addDefine('HAVE_'+i.PACKAGE, 1)
     if hasattr(p, 'lib'):
       if not isinstance(p.lib, list):
         packageLibs.append(p.lib)
       else:
         packageLibs.extend(p.lib)
     if hasattr(p, 'include'):
       if not isinstance(p.include, list):
         packageIncludes.append(p.include)
       else:
         packageIncludes.extend(p.include)
   packageLibs     = self.libraries.toStringNoDupes(packageLibs+self.libraries.math)
   packageIncludes = self.headers.toStringNoDupes(packageIncludes)

   includes = ['-I'+inc for inc in [os.path.join(self.petscdir.dir, self.arch.arch, 'include'), os.path.join(self.petscdir.dir, 'include')]]
   self.setCompilers.pushLanguage(self.languages.clanguage)
   compiler      = self.setCompilers.getCompiler()
   flags.append(self.setCompilers.getCompilerFlags())             # PCC_FLAGS
   flags.extend([self.setCompilers.CPPFLAGS, self.CHUD.CPPFLAGS]) # CPP_FLAGS
   flags.append('-D__SDIR__=\'"'+os.getcwd()+'"\'')
   cmd = ' '.join([compiler]+['-c']+includes+[packageIncludes]+flags+source)
   if self.debug: print cmd
   self.executeShellCommand(cmd,log=self.log)
   self.setCompilers.popLanguage()
   return

 def linkAR(self, libname,objects):
   '''
   '''

   flags           = []
   self.setCompilers.pushLanguage(self.languages.clanguage)
   linker = self.setCompilers.AR
   # should add FAST_AR_FLAGS to setCompilers
   if linker.endswith('ar'):
     flags.append('Scq')
   else:
     flags.append(self.setCompilers.AR_FLAGS)
   cmd = ' '.join([linker]+flags+[libname+'.'+self.setCompilers.AR_LIB_SUFFIX]+objects)
   if self.debug: print cmd
   self.executeShellCommand(cmd,log=self.log)   
   for i in objects:
     try:
       os.unlink(i)
     except:
       print 'File '+i+' was not compiled'
   self.setCompilers.popLanguage()
   return

 def runbase(self):
   ''' This is always run in one of the PETSc base package directories: vec, mat, ksp etc'''
   self.setup()
   
   libname = os.path.join(self.petscdir.dir,self.arch.arch,'lib','libpetsc'+os.path.basename(os.getcwd()))
   if libname.endswith('sys'): libname = libname[:-3]
   try:
     os.unlink(libname+'.'+self.setCompilers.AR_LIB_SUFFIX)
   except:
     pass
   
   os.path.walk(os.getcwd(),self.rundir,libname)
  
 def rundir(self,libname,dir,fnames):
   ''' This is run in a PETSc source directory'''
   basename = os.path.basename(dir)
   os.chdir(dir)

   # First, remove all subdirectories we should not enter
   if 'examples' in fnames: fnames.remove('examples')

   if not hasattr(self.compilers, 'FC'):
     rmnames = []
     for name in fnames:
       if name.startswith('ftn-'): rmnames.append(name)
       if name.startswith('f90-'): rmnames.append(name)
     for name in rmnames:
       fnames.remove(name)

   rmnames = []
   for name in fnames:
     if os.path.isdir(name):
       if not self.checkDir(name):rmnames.append(name)
   for name in rmnames:
     fnames.remove(name)

   # Get list of source files in the directory 
   if self.debug: print 'entering '+dir
   cnames = []
   onames = []
   fnames = []
   for f in os.listdir(dir):
     ext = os.path.splitext(f)[1]
     if ext == '.c':
       cnames.append(f)
       onames.append(f.replace('.c','.o'))

     if hasattr(self.compilers, 'FC'):
       if ext == '.F': 
         fnames.append(f)
         onames.append(f.replace('.F','.o'))                     
       if ext == '.F90':
         fnames.append(f)
         onames.append(f.replace('.F90','.o'))                     
       

   if cnames:
     if self.debug: print 'Compiling C files ',cnames
     self.compileC(cnames)
   if fnames:
     if self.debug:print 'Compiling F files ',fnames
     #self.compileF(fnames)
   if onames:
     self.linkAR(libname,onames)

 def checkDir(self,dir):
   '''Checks makefile to see if compiler is allowed to visit this directory for this configuration'''
   import re
   reg   = re.compile(' [ ]*')
   fname = os.path.join(dir,'makefile')
   fd = open(fname)
   text = fd.readline()
   while text:
     if text.startswith('#requires'):
       text = text[9:-1].strip()
       text = reg.sub(' ',text)
       rtype = text.split(' ')[0]
       rvalue = text.split(' ')[1]
       if rtype == 'scalar' and not self.scalarType.scalartype == rvalue:
         if self.debug: print 'rejecting because scalar type '+self.scalarType.scalartype+' is not '+rvalue
         return 0
       if rtype == 'language':
         if rvalue == 'CXXONLY' and self.languages.clanguage == 'C':
           if self.debug: print 'rejecting because language is '+self.languages.clanguage+' is not C++'
           return 0
       if rtype == 'precision' and not rvalue == self.scalarType.precision:
         if self.debug: print 'rejecting because precision '+self.scalarType.precision+' is not '+rvalue
         return 0
       # handles both missing packages and other random stuff that is treated as a package, that should be changed
       if rtype == 'package':
         if rvalue == "'"+'PETSC_HAVE_FORTRAN'+"'" or rvalue == "'"+'PETSC_USING_F90'+"'":
           if not hasattr(self.compilers, 'FC'):
             if self.debug: print 'rejecting because fortran is not being used'
             return 0
         elif rvalue == "'"+'PETSC_USE_LOG'+"'":
           if not self.libraryOptions.useLog:
             if self.debug: print 'rejecting because logging is turned off'
             return 0
         elif rvalue == "'"+'PETSC_USE_FORTRAN_KERNELS'+"'":
           if not self.libraryOptions.useFortranKernels:
             if self.debug: print 'rejecting because fortran kernels are turned off'
             return 0
         else:    
           found = 0
           for i in self.framework.packages:
             pname = 'PETSC_HAVE_'+i.PACKAGE
             pname = "'"+pname+"'"
             if pname == rvalue: found = 1
           if not found:
             if self.debug: print 'rejecting because package '+rvalue+' is not installed'
             return 0
         
     text = fd.readline()
   fd.close()
   return 1
   
if __name__ == '__main__':
  PETScMaker().runbase()
