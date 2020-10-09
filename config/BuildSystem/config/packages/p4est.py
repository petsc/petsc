import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = '1727693b446ee0c987be701320db0cd5de617cfc'
    self.download          = ['git://https://github.com/tisaac/p4est','https://github.com/tisaac/p4est/archive/'+self.gitcommit+'.tar.gz']
    self.functions         = ['p4est_init']
    self.includes          = ['p4est_bits.h']
    self.liblist           = [['libp4est.a', 'libsc.a']]
    self.downloadonWindows = 1
    return

  def setupHelp(self,help):
    '''Default GNU setupHelp, but p4est debugging option'''
    config.package.GNUPackage.setupHelp(self,help)
    import nargs
    help.addArgument(self.PACKAGE,'-with-p4est-debugging=<bool>',nargs.ArgBool(None,0,"Use p4est's (sometimes computationally intensive) debugging"))
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.zlib       = framework.require('config.packages.zlib',self)
    self.memalign   = framework.argDB['with-memalign']
    self.deps       = [self.mpi,self.blasLapack,self.zlib]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if self.argDB['with-p4est-debugging']:
      args.append('--enable-debug')
    args.append('--enable-mpi')
    args.append('CPPFLAGS="'+self.headers.toStringNoDupes(self.dinclude)+'"')
    args.append('LIBS="'+self.libraries.toString(self.dlib)+'"')
    args.append('--enable-memalign='+self.memalign)
    return args

  def updateGitDir(self):
    import os
    config.package.GNUPackage.updateGitDir(self)
    if not hasattr(self.sourceControl, 'git') or (self.packageDir != os.path.join(self.externalPackagesDir,'git.'+self.package)):
      return
    Dir = self.getDir()
    try:
      libsc = self.libsc
    except AttributeError:
      try:
        self.executeShellCommand([self.sourceControl.git, 'submodule', 'update', '--init'], cwd=Dir, log=self.log)
        import os
        if os.path.isfile(os.path.join(Dir,'sc','README')):
          self.libsc = os.path.join(Dir,'sc')
        else:
          raise RuntimeError
      except RuntimeError:
        raise RuntimeError('Could not initialize sc submodule needed by p4est')
    return

  def preInstall(self):
    '''check for configure script - and run bootstrap - if needed'''
    import os
    if not os.path.isfile(os.path.join(self.packageDir,'configure')):
      if not self.programs.libtoolize:
        raise RuntimeError('Could not bootstrap p4est using autotools: libtoolize not found')
      if not self.programs.autoreconf:
        raise RuntimeError('Could not bootstrap p4est using autotools: autoreconf not found')
      self.logPrintBox('Trying to bootstrap p4est using autotools; this may take several minutes')
      try:
        self.executeShellCommand('./bootstrap',cwd=self.packageDir,log=self.log)
      except RuntimeError as e:
        raise RuntimeError('Could not bootstrap p4est using autotools: maybe autotools (or recent enough autotools) could not be found?\nError: '+str(e))
