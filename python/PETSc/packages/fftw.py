
#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/fftw-3.1.1.tar.gz']
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
    # Get the FFTW directories
    fftwDir = self.getDir()
    installDir = os.path.join(fftwDir, self.arch.arch) #fftw-3.1.1/$PETSC_ARCH

    # Configure FFTW 
    self.framework.pushLanguage('C')
    ccompiler=self.framework.getCompiler()
    args = ['--prefix='+installDir, 'CC="'+self.framework.getCompiler()+'"']
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CXX="'+self.framework.getCompiler()+'"')
      args.append('--with-cppflags="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()

    args = ' '.join(args)
    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild FFTW oldargs = '+oldargs+'\n new args ='+args+'\n')
      try:
        self.logPrintBox('Configuring FFTW; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+fftwDir+'; ./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on FFTW: '+str(e))
      # Build FFTW
      try:
        self.logPrintBox('Compiling FFTW; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+fftwDir+'; make; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on FFTW: '+str(e))
      if not os.path.isdir(os.path.join(installDir,'lib')):
        self.framework.log.write('Error running make on FFTW   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on FFTW follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on FFTW *******\n')
        raise RuntimeError('Error running make on FFTW, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed FFTW into '+installDir)
    return self.getDir()
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
