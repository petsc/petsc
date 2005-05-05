#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/sundials.tar.gz']
    self.functions = ['CVSpgmr']
    self.includes  = ['sundialstypes.h']
    self.liblist   = [['libsundials_cvode.a','libsundials_fcvode.a','libsundials_nvecserial.a','libsundials_fnvecserial.a','libsundials_nvecparallel.a','libsundials_fnvecparallel.a','libsundials_shared.a']] #currently only support CVODE
    self.license   = 'http://www.llnl.gov/CASC/sundials/download/download.html'
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('PETSc.packages.MPI',self)
    self.blasLapack = framework.require('PETSc.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return
          
  def Install(self):
    # Get the SUNDIALS directories
    sundialsDir = self.getDir()
    installDir  = os.path.join(sundialsDir, self.arch.arch)
    
    # Configure SUNDIALS 
    args = ['--disable-examples']
    
    self.framework.pushLanguage('C')
    args.append('--with-ccflags="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()

    if 'FC' in self.framework.argDB:    
      self.framework.pushLanguage('FC')
      args.append('--with-fflags="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()

    if 'CXX' in self.framework.argDB:    
      self.framework.pushLanguage('Cxx')
      args.append('--with-cxxflags="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()

    (mpiDir,dummy) = os.path.split(self.mpi.lib[0])
    (mpiDir,dummy) = os.path.split(mpiDir)
    args.append('--with-mpi-root="'+mpiDir+'"') #better way to get mpiDir?

    args.append('--with-blas="'+self.libraries.toString(self.blasLapack.dlib)+'"') 
    
    args = ' '.join(args)
    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild SUNDIALS oldargs = '+oldargs+'\n new args ='+args+'\n')
      try:
        self.logPrintBox('Configuring sundials; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+installDir+'; ../configure '+args, timeout=900, log = self.framework.log)[0]

      except RuntimeError, e:
        raise RuntimeError('Error running configure on SUNDIALS: '+str(e))
      # Build SUNDIALS
      try:
        self.logPrintBox('Compiling sundials; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+installDir+'; SUNDIALS_INSTALL_DIR='+installDir+'; export SUNDIALS_INSTALL_DIR; make clean; make; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUNDIALS: '+str(e))
      if not os.path.isdir(os.path.join(installDir,'lib')):
        self.framework.log.write('Error running make on SUNDIALS   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on SUNDIALS follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on SUNDIALS *******\n')
        raise RuntimeError('Error running make on SUNDIALS, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed SUNDIALS into '+installDir)
    return self.getDir()
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
