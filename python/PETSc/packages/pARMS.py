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
    # Get the pARMS directories
    parmsDir = self.getDir()
    
    # Configure and Build pARMS
    if os.path.isfile(os.path.join(parmsDir,'makefile.in')):
      output  = config.base.Configure.executeShellCommand('cd '+parmsDir+'; rm -f makefile.in', timeout=2500, log = self.framework.log)[0]
    g = open(os.path.join(parmsDir,'makefile.in'),'w')
    g.write('SHELL =	/bin/sh\n')
    g.write('.SUFFIXES:\n')
    g.write('.SUFFIXES: .c .o .f .F\n')
    g.write('PARMS_ROOT = '+parmsDir+'\n')
    
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
    if not os.path.isdir(self.installDir):
      os.mkdir(self.installDir)
    if not os.path.isfile(os.path.join(self.confDir,'pARMS')) or not (self.getChecksum(os.path.join(self.confDir,'pARMS')) == self.getChecksum(os.path.join(parmsDir,'makefile.in'))):  
      self.framework.log.write('Have to rebuild pARMS, makefile.in != '+self.confDir+'/pARMS\n')
      try:
        self.logPrintBox('Compiling pARMS; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+parmsDir+';PARMS_INSTALL_DIR='+self.installDir+';export PARMS_INSTALL_DIR; mkdir '+os.path.join(self.installDir,self.libdir)+'; make clean; make; cp include/*.h '+os.path.join(self.installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on pARMS: '+str(e))
      if not os.path.isfile(os.path.join(self.installDir,self.libdir,'libparms.a')):
        self.framework.log.write('Error running make on pARMS   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on pARMS follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on pARMS *******\n')
        raise RuntimeError('Error running make on pARMS, libraries not installed')
      
      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(parmsDir,'makefile.in')+' '+self.confDir+'/pARMS', timeout=5, log = self.framework.log)[0]
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed pARMS into '+self.installDir)
    return self.installDir

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
