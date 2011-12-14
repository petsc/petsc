import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download     =      ['http://www.open-mpi.org/software/hwloc/v1.3/downloads/hwloc-1.3.tar.gz']
    self.functions =         ['hwloc_topology_init']
    self.includes  =         ['hwloc.h']
    self.liblist   =         [['libhwloc.a']]
    self.worksonWindows =    1
    self.downloadonWindows = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    return

  def Install(self):
    import os

    self.framework.pushLanguage('C')
    args = ['--prefix='+self.installDir]
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
    args.append('CC="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'hwloc'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('hwloc'):
      try:
        self.logPrintBox('Configuring hwloc; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on hwloc: '+str(e))
      try:
        self.logPrintBox('Compiling hwloc; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on hwloc: '+str(e))
      self.postInstall(output1+err1+output2+err2,'hwloc')
    return self.installDir

