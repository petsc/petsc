import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = 'v2.0'
    self.download          = ['git://https://github.com/cburstedde/p4est','https://github.com/p4est/p4est.github.io/raw/master/release/p4est-2.0.tar.gz']
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
    self.blasLapack = self.framework.require('config.packages.BlasLapack',self)
    self.deps = [self.mpi,self.blasLapack]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if self.argDB['with-p4est-debugging']:
      args.append('--enable-debug')
    args.append('--enable-mpi')
    args.append('LIBS="'+self.libraries.toString(self.blasLapack.dlib)+'"')
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

  def Install(self):
    '''bootstrap, then standar GNU configure; make; make install'''
    import os
    if not os.path.isfile(os.path.join(self.packageDir,'configure')):
      self.logPrintBox('Trying to bootstrap p4est using autotools; this make take several minutes')
      try:
        self.executeShellCommand('./bootstrap',cwd=self.packageDir,log=self.log)
      except RuntimeError as e:
        raise RuntimeError('Could not bootstrap p4est using autotools: maybe autotools (or recent enough autotools) could not be found?\nError: '+str(e))
    return config.package.GNUPackage.Install(self)
