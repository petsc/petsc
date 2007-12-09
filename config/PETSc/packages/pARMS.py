#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/pARMS_3.tar.gz']
    self.functions = ['parms_VecCreate']
    self.includes  = ['parms.h']
    self.liblist   = [['libparms.a']]
    self.license   = 'http://www-users.cs.umn.edu/~saad/software/pARMS/pARMS.html'
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    self.types      = framework.require('config.types', self)
    return

  def Install(self):

    
    # Configure and Build pARMS
    g = open(os.path.join(self.packageDir,'makefile.in'),'w')
    g.write('SHELL =	/bin/sh\n')
    g.write('.SUFFIXES:\n')
    g.write('.SUFFIXES: .c .o .f .F\n')
    g.write('PARMS_ROOT = '+self.packageDir+'\n')
    
    # path of the header files of pARMS
    g.write('IFLAGS     = -I${PARMS_ROOT}/include\n')
    
    # C compiler
    self.setCompilers.pushLanguage('C')
    g.write('CC         = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS     = '+self.setCompilers.getCompilerFlags()+'-DUSE_MPI -DREAL=double -DDBL -DHAS_BLAS -DGCC3\n')
    self.setCompilers.popLanguage()
    
    # FORTRAN compiler
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      g.write('FC         = '+self.setCompilers.getCompiler()+'\n')
      g.write('FFLAGS     = '+self.setCompilers.getCompilerFlags()+' -DVOID_POINTER_SIZE_'+str(self.types.sizes['sizeof_void_p'])+'\n')
      # set fortran name mangling
      if self.compilers.fortranMangling == 'underscore':
        g.write('CFDEFS     = -DFORTRAN_UNDERSCORE\n')
      elif self.compilers.fortranMangling == 'capitalize':
        g.write('CFDEFS     = -DFORTRAN_CAPS\n')
      else:
        g.write('CFDEFS     = -DFORTRAN_DOUBLE_UNDERSCORE\n') 
      g.write('CFFLAGS    = ${CFDEFS} -DVOID_POINTER_SIZE_'+str(self.types.sizes['sizeof_void_p'])+'\n')
      self.setCompilers.popLanguage()
      
    else:
      raise RuntimeError('pARMS requires a fortran compiler! No fortran compiler configured!')
    
    g.write('RM         = rm\n')
    g.write('RMFLAGS    = -rf\n')
    g.write('EXTFLAGS   = -x\n')

    # archive and options
    g.write('AR         = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')

    # pARMS lib and its directory
    g.write('LIBDIR     = '+self.installDir+'/lib\n') 
    g.write('LIB        = ${LIBDIR}/libparms.a\n') 
    g.write('LIBFLAGS   = -L${LIBDIR}\n') 
    g.write('PARMS_LIBS = -lparms\n') 
    
    g.write('.c.o:  \n')
    g.write('	${CC} ${IFLAGS} ${ISRCINC} ${XIFLAGS} $(COPTFLAGS)  ${CFLAGS} ${CFFLAGS} $< -c -o $@\n')

    g.write('.F.o: \n') 
    g.write('	${FC} -FR ${IFLAGS} ${FFLAGS} $< -c -o $(@F) \n')

    g.write('.f.o: \n')
    g.write('	${FC} ${FFLAGS} $< -c -o $(@F) \n')
    #-----------------------------------------
    g.close()

    if self.installNeeded('makefile.in'):
      try:
        self.logPrintBox('Compiling pARMS; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';PARMS_INSTALL_DIR='+self.installDir+';export PARMS_INSTALL_DIR; mkdir '+os.path.join(self.installDir,self.libdir)+'; make clean; make; cp include/*.h '+os.path.join(self.installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on pARMS: '+str(e))
      self.checkInstall(output,'makefile.in')
    return self.installDir

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
