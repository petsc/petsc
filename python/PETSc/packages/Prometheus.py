#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.parmetis     = self.framework.require('PETSc.packages.ParMetis',self)
    self.download     = ['http://www.cs.berkeley.edu/~madams/Prometheus-1.8.0.tar.gz']
    self.deps         = [self.parmetis,self.mpi,self.blasLapack]
    self.functions    = ['HYPRE_IJMatrixCreate']
    self.includes     = ['HYPRE.h']
    self.liblist      = ['libprometheus.a'] 
    self.cxx          = 1   # requires C++
    return
        
  def Install(self):
    prometheusDir = self.getDir()

    # Get the PROMETHEUS directories
    installDir = os.path.join(prometheusDir, self.arch.arch)
    # Configure and Build PROMETHEUS
    self.framework.pushLanguage('C')
    args = ['--prefix='+installDir, '--with-CC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"']
    self.framework.popLanguage()
    if 'CXX' in self.framework.argDB:
      self.framework.pushLanguage('Cxx')
      args.append('--with-CXX="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    if 'FC' in self.framework.argDB:
      self.framework.pushLanguage('FC')
      args.append('--with-F77="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    if self.mpi.include:
      if len(self.mpi.include) > 1:
        raise RuntimeError("prometheus assumes there is a single MPI include directory")
      args.append('--with-mpi-include="'+self.mpi.include[0].replace('-I','')+'"')
    libdirs = []
    for l in self.mpi.lib:
      ll = os.path.dirname(l)
      libdirs.append(ll)
    libdirs = ' '.join(libdirs)
    args.append('--with-mpi-lib-dirs="'+libdirs+'"')
    libs = []
    for l in self.mpi.lib:
      ll = os.path.basename(l)
      libs.append(ll[3:-2])
    libs = ' '.join(libs)
    args.append('--with-mpi-libs="'+libs+'"')
    args.append('--with-babel=0')
    args.append('--with-mli=0')    
    args.append('--with-FEI=0')    
    args.append('--with-blas="'+self.libraries.toString(self.blasLapack.dlib)+'"')        
    args = ' '.join(args)

    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild PROMETHEUS oldargs = '+oldargs+' new args '+args+'\n')
      try:
        self.logPrint("Configuring prometheus; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(prometheusDir,'src')+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on PROMETHEUS: '+str(e))
      try:
        self.logPrint("Compiling prometheus; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(prometheusDir,'src')+';PROMETHEUS_INSTALL_DIR='+installDir+';export PROMETHEUS_INSTALL_DIR; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on PROMETHEUS: '+str(e))
      if not os.path.isdir(os.path.join(installDir,'lib')):
        self.framework.log.write('Error running make on PROMETHEUS   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on PROMETHEUS follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on PROMETHEUS *******\n')
        raise RuntimeError('Error running make on PROMETHEUS, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      #need to run ranlib on the libraries using the full path
      try:
        output  = config.base.Configure.executeShellCommand('ranlib '+os.path.join(installDir,'lib')+'/lib*.a', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on PROMETHEUS libraries: '+str(e))
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed PROMETHEUS into '+installDir)
    return self.getDir()

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
