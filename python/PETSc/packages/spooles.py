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
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/spooles-2.2.tar.gz']
    self.deps         = [self.mpi,self.blasLapack]
    self.functions    = ['InpMtx_init']
    self.includes     = ['MPI/spoolesMPI.h']
    self.libdir       = ''
    self.includedir   = ''
    return

  def generateLibList(self,dir):
    libs = ['MPI/src/spoolesMPI', 'spooles']
    alllibs = []
    for l in libs:
      alllibs.append(os.path.join(dir,l+'.a'))
    import config.setCompilers
    self.framework.pushLanguage('C')
    self.framework.popLanguage()    
    return alllibs
          
  def Install(self):
    # Get the SPOOLES directories
    spoolesDir = self.getDir()
    installDir = os.path.join(spoolesDir, self.arch.arch)
    # Configure and Build SPOOLES
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
        raise RuntimeError("spooles assumes there is a single MPI include directory")
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
    args.append('--with-blas="'+self.libraries.toString(self.blasLapack.dlib)+'"')        
    args = ' '.join(args)

    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild SPOOLES oldargs = '+oldargs+' new args '+args+'\n')
      try:
        self.logPrint("Compiling spooles; this may take several minutes\n", debugSection='screen')
        # cp spoolesDir into spoolesDir/self.arch.arch
        #  -- can be improved by mkdir subdirectories and cp needed header files
        output  = config.base.Configure.executeShellCommand('cd '+spoolesDir+'; SPOOLES_INSTALL_DIR='+installDir+'; export SPOOLES_INSTALL_DIR; rm -rf '+self.arch.arch+'; cd ..; cp -r '+spoolesDir+' '+self.arch.arch+'; cd '+spoolesDir+'; mv ../'+self.arch.arch+' '+spoolesDir+'; cd '+spoolesDir+'; cd '+self.arch.arch+'; make lib', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SPOOLES: '+str(e))
      if not os.path.isdir(os.path.join(installDir,self.libdir)):
        self.framework.log.write('Error running make on SPOOLES   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on SPOOLES follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on SPOOLES *******\n')
        raise RuntimeError('Error running make on SPOOLES, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed SPOOLES into '+installDir)
    return self.getDir()

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
