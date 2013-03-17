#!/usr/bin/env python
import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download          = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/parmetis-4.0.2-p3.tar.gz']
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
    self.cmake           = framework.require('PETSc.packages.cmake',self)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries', self)
    self.metis           = framework.require('PETSc.packages.metis',self)
    self.deps            = [self.mpi, self.metis]
    return

  def Install(self):
    import os
    import shlex

    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build METIS')

    args = ['-DCMAKE_INSTALL_PREFIX='+self.installDir]
    args.append('-DCMAKE_VERBOSE_MAKEFILE=1')
    args.append('-DGKLIB_PATH=../headers') # assumes that the 'build' folder is only one directory down

    self.framework.pushLanguage('C')
    args.append('-DCMAKE_C_COMPILER="'+self.framework.getCompiler()+'"')
    args.append('-DCMAKE_AR='+self.setCompilers.AR)
    ranlib = shlex.split(self.setCompilers.RANLIB)[0]
    args.append('-DCMAKE_RANLIB='+ranlib)

    cflags = self.setCompilers.getCompilerFlags()
    args.append('-DCMAKE_C_FLAGS:STRING="'+cflags+'"')
    self.framework.popLanguage()

    if self.sharedLibraries.useShared:
      args.append('-DSHARED=1')

    if self.compilerFlags.debugging:
      args.append('-DDEBUG=1')

    if self.libraryOptions.integerSize == 64:
      args.append('-DMETIS_USE_LONGINDEX=1')

    if self.scalartypes.precision == 'double':
      args.append('-DMETIS_USE_DOUBLEPRECISION=1')
    elif self.scalartypes.precision == 'quad':
      raise RuntimeError('ParMETIS cannot be built with quad precision')

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'parmetis'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('parmetis'):    # Now compile & install

      # effectively, this is 'make clean'
      folder = os.path.join(self.packageDir, self.arch)
      if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
        os.mkdir(folder)

      try:
        self.logPrintBox('Configuring ParMETIS; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+folder+' && '+self.cmake.cmake+' .. '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on ParMETIS: '+str(e))
      try:
        self.logPrintBox('Compiling ParMETIS; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+folder+' && make && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on ParMETIS: '+str(e))
      self.postInstall(output1+err1+output2+err2,'parmetis')
    return self.installDir
