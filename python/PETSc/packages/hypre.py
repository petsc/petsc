#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/hypre-2.0.0.tar.gz']
    self.functions = ['HYPRE_IJMatrixCreate']
    self.includes  = ['HYPRE.h']
    self.liblist   = [['libHYPRE.a']]
    self.license   = 'http://www.llnl.gov/CASC/hypre/download/hyprebeta_cur_agree.html'
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]  
    return

  def generateLibList(self,dir):
    '''Normally the one in package.py is used, but hypre requires the extra C++ library'''
    alllibs = PETSc.package.Package.generateLibList(self,dir)
    import config.setCompilers
    if self.languages.clanguage == 'C':
      alllibs[0].extend(self.compilers.cxxlibs)
    return alllibs
        
  def Install(self):

    self.framework.pushLanguage('C')
    args = ['--prefix='+self.installDir, 'CC="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"']
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CXX="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    else:
      raise RuntimeError('Error: Hypre requires C++ compiler. None specified')
    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('F77="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags().replace('-Mfree','')+'"')
      self.framework.popLanguage()
    if self.mpi.include:
      # just use the first dir - and assume the subsequent one isn't necessary [relavant only on AIX?]
      print 'using: ' + '--with-MPI-include="'+self.mpi.include[0]+'"'
      args.append('--with-MPI-include="'+self.mpi.include[0]+'"')
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

    # tell hypre configure not to look for blas/lapack [and not use hypre-internal blas]
    args.append('--with-blas-libs=')
    args.append('--with-blas-lib-dir=')
    args.append('--with-lapack-libs=')
    args.append('--with-lapack-lib-dir=')
    args.append('--with-blas=yes')
    args.append('--with-lapack=yes')
    
    args.append('--without-babel')
    args.append('--without-mli')
    args.append('--without-fei')
    args.append('--without-superlu')
    args = ' '.join(args)
    fd = file(os.path.join(self.installDir,'hypre'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('hypre'):
      try:
        self.logPrintBox('Configuring hypre; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+';make distclean;./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on HYPRE: '+str(e))
      try:
        self.logPrintBox('Compiling hypre; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+';HYPRE_INSTALL_DIR='+self.installDir+';export HYPRE_INSTALL_DIR; make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on HYPRE: '+str(e))
      #need to run ranlib on the libraries using the full path
      try:
        output  = config.base.Configure.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(self.installDir,'lib')+'/lib*.a', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on HYPRE libraries: '+str(e))
      self.checkInstall(output,'hypre')
    return self.installDir
  
  def configureLibrary(self):
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
