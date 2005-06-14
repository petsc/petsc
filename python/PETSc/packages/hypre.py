#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/hypre-1.9.0b.tar.gz']
    self.functions = ['HYPRE_IJMatrixCreate']
    self.includes  = ['HYPRE.h']
    self.license   = 'http://www.llnl.gov/CASC/hypre/download/hyprebeta_cur_agree.html'
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('PETSc.packages.MPI',self)
    self.blasLapack = framework.require('PETSc.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def generateLibList(self,dir):
    '''Normally the one in package.py is used, but hypre requires the extra C++ library'''
    libs = ['DistributedMatrix',
            'DistributedMatrixPilutSolver',
            'Euclid',
            'IJ_mv',
            'LSI',
            'MatrixMatrix',
            'ParaSails',
            'krylov',
            'lobpcg',
            'parcsr_ls',
            'parcsr_mv',
            'seq_mv',
            'sstruct_ls',
            'sstruct_mv',
            'struct_ls',
            'struct_mv',
            'utilities'
            ]
    alllibs = []
    for l in libs:
      alllibs.append('libHYPRE_'+l+'.a')
    # Now specify -L hypre-lib-path only to the first library
    alllibs[0] = os.path.join(dir,alllibs[0])
    import config.setCompilers
    self.framework.pushLanguage('C')
    if config.setCompilers.Configure.isGNU(self.framework.getCompiler()):
      alllibs.append('-lstdc++')
    self.framework.popLanguage()    
    return [alllibs]
          
        
  def Install(self):
    hypreDir = self.getDir()

    # Get the HYPRE directories
    installDir = os.path.join(hypreDir, self.arch.arch)
    # Configure and Build HYPRE
    self.framework.pushLanguage('C')
    args = ['--prefix='+installDir, '--with-CC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"']
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('--with-CXX="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('--with-F77="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    if self.mpi.include:
      # just use the first dir - and assume the subsequent one isn't necessary [relavant only on AIX?]
      args.append('--with-MPI-include="'+self.mpi.include[0].replace('-I','')+'"')
    libdirs = []
    for l in self.mpi.lib:
      ll = os.path.dirname(l)
      libdirs.append(ll)
    libdirs = ' '.join(libdirs)
    args.append('--with-MPI-lib-dirs="'+libdirs+'"')
    libs = []
    for l in self.mpi.lib:
      ll = os.path.basename(l)
      libs.append(ll[3:-2])
    libs = ' '.join(libs)
    args.append('--with-MPI-libs="'+libs+'"')
    args.append('--without-babel')
    args.append('--without-mli')    
    args.append('--without-FEI')
    args.append('--without-blas')
    args = ' '.join(args)

    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
      self.framework.log.write('Have to rebuild HYPRE oldargs = '+oldargs+'\n new args ='+args+'\n')
      try:
        self.logPrintBox('Configuring hypre; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(hypreDir,'src')+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on HYPRE: '+str(e))
      try:
        self.logPrintBox('Compiling hypre; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(hypreDir,'src')+';HYPRE_INSTALL_DIR='+installDir+';export HYPRE_INSTALL_DIR; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on HYPRE: '+str(e))
      if not os.path.isdir(os.path.join(installDir,'lib')):
        self.framework.log.write('Error running make on HYPRE   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on HYPRE follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on HYPRE *******\n')
        raise RuntimeError('Error running make on HYPRE, libraries not installed')
      
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()

      #need to run ranlib on the libraries using the full path
      try:
        output  = config.base.Configure.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(installDir,'lib')+'/lib*.a', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on HYPRE libraries: '+str(e))
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed HYPRE into '+installDir)
    return self.getDir()
  
  def configureLibraryOld(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by hypre'''
    '''Normally you do not need to provide this method'''
    PETSc.package.Package.configureLibrary(self)
    # hypre requires LAPACK routine dgels()
    if not self.blasLapack.checkForRoutine('dgels'):
      raise RuntimeError('hypre requires the LAPACK routine dgels(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it')
    self.framework.log.write('Found dgels() in Lapack library as needed by hypre\n')
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
