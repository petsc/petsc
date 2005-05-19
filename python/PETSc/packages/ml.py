#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/ml-3.1-pre.tar.gz']
    self.functions = ['ML_Set_PrintLevel']
    self.includes  = ['ml_include.h']
    self.liblist   = [['libml.a']]
    self.license   = 'http://software.sandia.gov/trilinos/downloads.html'
    self.fc        = 1 # looks like trilinos requires Fortran
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('PETSc.packages.MPI',self)
    self.blasLapack = framework.require('PETSc.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return
          
  def Install(self):
    # Get the ML directories
    mlDir = self.getDir()
    installDir  = os.path.join(mlDir, self.arch.arch)
    
    # Configure ML 
    args = ['--prefix='+installDir]
    
    self.framework.pushLanguage('C')
    CCenv = self.framework.getCompiler()
    args.append('--with-ccflags="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()

    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      F77env = self.framework.getCompiler()
      args.append('--with-fflags="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    else:
      F77env = ''

    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      CXXenv = self.framework.getCompiler()
      args.append('--with-cxxflags="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    else:
      CXXenv = ''
  
    if self.mpi.directory:
      args.append('--with-mpi="'+self.mpi.directory+'"') 
    else:
      raise RuntimeError("Installing ML requires explicit root directory of MPI\nRun config/configure.py again with the additional argument --with-mpi-dir=rootdir")

    libs = []
    for l in self.mpi.lib:
      ll = os.path.basename(l)
      libs.append('-l'+ll[3:-2])
    libs = ' '.join(libs) # '-lmpich -lpmpich'
    args.append('--with-mpi-libs="'+libs+'"')
    args.append('--with-blas="'+self.libraries.toString(self.blasLapack.dlib)+'"') 
    
    args = ' '.join(args)
    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild ML oldargs = '+oldargs+'\n new args ='+args+'\n')
      try:
        self.logPrintBox('Configuring ml; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('CC='+CCenv+'; export CC; F77='+F77env+'; export F77; CXX='+CXXenv+'; export CXX; cd '+mlDir+'; ./configure '+args+' --disable-epetra --disable-aztecoo --disable-ml-examples', timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on ML: '+str(e))
      # Build ML
      try:
        self.logPrintBox('Compiling ml; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+mlDir+'; ML_INSTALL_DIR='+installDir+'; export ML_INSTALL_DIR; make clean; make; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on ML: '+str(e))
      if not os.path.isdir(os.path.join(installDir,'lib')):
        self.framework.log.write('Error running make on ML   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on ML follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on ML *******\n')
        raise RuntimeError('Error running make on ML, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed ML into '+installDir)
    return self.getDir()
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
