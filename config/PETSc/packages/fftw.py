
#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/fftw-3.2alpha2.tar.gz']
    self.functions = ['fftw_malloc'] 
    self.includes  = ['fftw3.h']  
    self.liblist   = [['libfftw3.a']]
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.deps = []
    return
          
  def Install(self):

    self.framework.pushLanguage('C')
    ccompiler=self.framework.getCompiler()
    args = ['--prefix='+self.installDir, 'CC="'+self.framework.getCompiler()+'"']
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CXX="'+self.framework.getCompiler()+'"')
      args.append('--with-cppflags="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()

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
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
