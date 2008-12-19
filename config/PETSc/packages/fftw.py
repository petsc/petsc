
#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/fftw-3.2alpha3.tar.gz']
    self.functions = ['fftw_malloc'] 
    self.includes  = ['fftw3.h']  
    self.liblist   = [['libfftw3_mpi.a','libfftw3.a']]
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi  = framework.require('config.packages.MPI',self)
    self.deps = [self.mpi]
    self.libraries = framework.require('config.libraries',self)
    return
          
  def Install(self):

    args = ['--prefix='+self.installDir]

    self.framework.pushLanguage('C')
    ccompiler=self.framework.getCompiler()
    args.append('CC="'+self.framework.getCompiler()+'"')
    args.append('MPICC="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CXX="'+self.framework.getCompiler()+'"')
      args.append('CXXFLAGS="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
     # else error?
    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('F77="'+self.framework.getCompiler()+'"')
      args.append('FFLAGS="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
     #else error?

    # MPI args need fixing
    args.append('--enable-mpi')
    if self.mpi.lib:
      args.append('LIBS="'+self.libraries.toStringNoDupes(self.mpi.lib)+'"')

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'fftw'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('fftw'):
      try:
        self.logPrintBox('Configuring FFTW; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; ./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on FFTW: '+str(e))
      try:
        self.logPrintBox('Compiling FFTW; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; make; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on FFTW: '+str(e))
      self.checkInstall(output,'fftw')
    return self.installDir

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by FFTW'''
    '''Normally you do not need to provide this method'''
    PETSc.package.Package.configureLibrary(self)
    # FFTW requires complex precision
    if not self.scalartypes.scalartype.lower() == 'complex':
      raise RuntimeError('FFTW requires the complex precision, run config/configure.py --with-scalar-type=complex')
    return
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
