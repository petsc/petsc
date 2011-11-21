import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download          = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/metis-5.0.2-p1.tar.gz']
    self.functions         = ['METIS_PartGraphKway']
    self.includes          = ['metis.h']
    self.liblist           = [['libmetis.a']]
    self.needsMath         = 1
    self.complex           = 1
    self.worksonWindows    = 1
    self.downloadonWindows = 1
    self.requires32bitint  = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.cmake           = framework.require('PETSc.utilities.CMake',self)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries', self)
    self.scalartypes     = framework.require('PETSc.utilities.scalarTypes', self)
    self.deps = []
    return

  def Install(self):
    import os

    if not self.cmake.found:
      raise RuntimeError('CMake > 2.8.5 is needed to build METIS')

    self.framework.pushLanguage('C')
    args = ['prefix='+self.installDir]
    args.append('cc="'+self.framework.getCompiler()+'"')

    if self.setCompilers.isDarwin() or self.setCompilers.isPGI(self.framework.getCompiler()):
      args.append('cflags=-D__thread=\"\"')
    self.framework.popLanguage()

    if self.sharedLibraries.useShared:
      args.append('shared=1')

    if self.compilerFlags.debugging:
      args.append('debug=1')

    if self.libraryOptions.integerSize == 64:
      args.append('longindex=1')

    if self.scalartypes.precision == 'double':
      args.append('doubleprecision=1')
    elif self.scalartypes.precision == 'quad':
      raise RuntimeError('METIS cannot be built with quad precision')

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'metis'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('metis'):
      try:
        self.logPrintBox('Configuring METIS; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make distclean && make config '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on METIS: '+str(e))
      try:
        self.logPrintBox('Compiling METIS; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on METIS: '+str(e))
      self.postInstall(output1+err1+output2+err2,'metis')
    return self.installDir
