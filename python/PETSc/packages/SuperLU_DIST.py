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
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/SuperLU_DIST_2.0-Jul_21_2004.tar.gz']
    self.deps         = [self.mpi,self.blasLapack]
    self.functions    = ['set_default_options_dist']
    self.includes     = ['superlu_ddefs.h']
    self.libdir       = ''
    self.includedir   = 'SRC'
    return


  def generateLibList(self,dir):
    '''Normally the one in package.py is used, but superlu_dist requires the extra C++ library'''
    libs = ['superlu_linux']
    alllibs = []
    for l in libs:
      alllibs.append(os.path.join(dir,l+'.a'))
    import config.setCompilers
    self.framework.pushLanguage('C')
    self.framework.popLanguage()    
    return alllibs
              
  def Install(self):
    # Get the SUPERLU_DIST directories
    superlu_distDir = self.getDir()
    installDir = os.path.join(superlu_distDir, self.arch.arch)
    # Configure and Build SUPERLU_DIST
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
        raise RuntimeError("superlu_dist assumes there is a single MPI include directory")
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
      self.framework.log.write('Have to rebuild SUPERLU_DIST oldargs = '+oldargs+' new args '+args+'\n')
      try:
        self.logPrint("Compiling superlu_dist; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+superlu_distDir+';SUPERLU_DIST_INSTALL_DIR='+installDir+';export SUPERLU_DIST_INSTALL_DIR; make lib; mv *.a '+os.path.join(installDir,self.libdir)+'; mkdir '+os.path.join(installDir,self.includedir)+'; cp SRC/*.h '+os.path.join(installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUPERLU_DIST: '+str(e))
      if not os.path.isdir(os.path.join(installDir,self.libdir)):
        self.framework.log.write('Error running make on SUPERLU_DIST   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on SUPERLU_DIST follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on SUPERLU_DIST *******\n')
        raise RuntimeError('Error running make on SUPERLU_DIST, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed SUPERLU_DIST into '+installDir)
    return self.getDir()

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
