#!/usr/bin/env python
import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download          = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/parmetis-4.0.2-p1.tar.gz']
    self.functions         = ['ParMETIS_V3_PartKway']
    self.includes          = ['parmetis.h']
    self.liblist           = [['libparmetis.a']]
    self.needsMath         = 1
    self.double            = 0
    self.complex           = 1
    self.requires32bitint  = 0
    self.worksonWindows    = 1
    self.downloadonWindows = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.cmake           = framework.require('PETSc.utilities.CMake',self)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries', self)
    self.metis           = framework.require('PETSc.packages.metis',self)
    self.deps            = [self.mpi, self.metis]
    return

  def Install(self):
    import os

    if not self.cmake.found:
      raise RuntimeError('CMake > 2.8.5 is needed to build METIS')

    self.framework.pushLanguage('C')
    args = ['prefix='+self.installDir]
    args.append('cc="'+self.framework.getCompiler()+'"')
    self.framework.popLanguage()

    if self.sharedLibraries.useShared:
      args.append('shared=1')

    if self.compilerFlags.debugging:
      args.append('debug=1')

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'parmetis'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('parmetis'):    # Now compile & install
      try:
        self.logPrintBox('Configuring ParMETIS; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make distclean && make config '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on ParMETIS: '+str(e))
      try:
        self.logPrintBox('Compiling ParMETIS; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on ParMETIS: '+str(e))
      self.postInstall(output1+err1+output2+err2,'parmetis')
    return self.installDir
