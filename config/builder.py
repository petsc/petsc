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
   return

 def setupModules(self):
#    self.mpi           = self.framework.require('config.packages.MPI',      None)
   self.setCompilers  = self.framework.require('config.setCompilers',      None)
   self.arch          = self.framework.require('PETSc.utilities.arch',     None)
   self.petscdir      = self.framework.require('PETSc.utilities.petscdir', None)
   self.languages     = self.framework.require('PETSc.utilities.languages',None)
   self.debugging     = self.framework.require('PETSc.utilities.debugging',None)
   self.make          = self.framework.require('PETSc.utilities.Make',     None)
   self.CHUD          = self.framework.require('PETSc.utilities.CHUD',     None)
   self.compilers     = self.framework.require('config.compilers',         None)
   self.types         = self.framework.require('config.types',             None)
   self.headers       = self.framework.require('config.headers',           None)
   self.functions     = self.framework.require('config.functions',         None)
   self.libraries     = self.framework.require('config.libraries',         None)
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
   print cmd
   self.setCompilers.popLanguage()
   return

 def linkAR(self, libname,source):
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
   cmd = ' '.join([linker]+flags+source+[libname+'.'+self.setCompilers.AR_LIB_SUFFIX])
#   print cmd
   self.setCompilers.popLanguage()
   return

 def runbase(self):
   ''' This is always run in one of the PETSc base package directories: vec, mat, ksp etc'''
   self.setup()
   
   libname = 'lib'+os.path.basename(os.getcwd())
   os.path.walk(os.getcwd(),self.rundir,libname)
  
 def rundir(self,libname,dir,fnames):
   ''' This is run in a PETSc source directory'''
   basename = os.path.basename(dir)
   if 'examples' in fnames: fnames.remove('examples')

   # if no Fortran compiler
   rmnames = []
   for name in fnames:
     if name.startswith('ftn-'): rmnames.append(name)
     if name.startswith('f90-'): rmnames.append(name)
   for name in rmnames:
     fnames.remove(name)

   print dir
   cnames = []
   onames = []
   fnames = []
   for f in os.listdir(dir):
     ext = os.path.splitext(f)[1]
     if ext == '.c':
       cnames.append(f)
       onames.append(f.replace('.c','.o'))

     # if fortran compiler
     if ext == '.F': 
       fnames.append(f)
       onames.append(f.replace('.F','.o'))                     
     if ext == '.F90':
       fnames.append(f)
       onames.append(f.replace('.F90','.o'))                     
       

   if cnames:
     print 'Compiling C files ',cnames
     self.compileC(cnames)
   if fnames:
     print 'Compiling F files ',fnames
     #self.compileF(fnames)
   if onames:
     self.linkAR(libname,onames)

if __name__ == '__main__':
  PETScMaker().runbase()
