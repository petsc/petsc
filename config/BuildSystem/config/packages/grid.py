import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit      = '796abfad80625d81bb16af7ff6ec612a836f17d8'
    self.download       = ['git://https://github.com/paboyle/Grid.git']
    self.buildLanguages = ['Cxx']
    self.maxCxxVersion  = 'c++17'
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.ssl   = framework.require('config.packages.ssl', self)
    self.gmp   = framework.require('config.packages.gmp', self)
    self.mpfr  = framework.require('config.packages.mpfr', self)
    self.eigen = framework.require('config.packages.eigen', self)
    self.deps  = [self.ssl, self.gmp, self.mpfr, self.eigen]
    return

  def formGNUConfigureArgs(self):
    import os
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--exec-prefix='+os.path.join(self.installDir, 'Grid'))
    args.append('--with-openssl='+self.ssl.directory)
    args.append('--with-gmp='+self.gmp.getInstallDir())
    args.append('--with-mpfr='+self.mpfr.getInstallDir())
    # Check for --enable-simd=AVX
    args.append('--enable-comms=mpi-auto')
    return args

  def preInstall(self):
    import os

    # Link Eigen directories
    try:
      self.logPrintBox('Linking Eigen to ' +self.PACKAGE)
      eigenDir = os.path.join(self.installDir, 'include', 'eigen3', 'Eigen')
      gridDir  = os.path.join(self.packageDir, 'Grid', 'Eigen')
      if not os.path.lexists(gridDir):
        os.symlink(eigenDir, gridDir)
      eigenDir = os.path.join(self.installDir, 'include', 'eigen3', 'unsupported', 'Eigen')
      gridDir  = os.path.join(self.packageDir, 'Grid', 'Eigen', 'unsupported')
      if not os.path.lexists(gridDir):
        os.symlink(eigenDir, gridDir)
    except OSError as e:
      raise RuntimeError('Error linking Eigen to ' + self.PACKAGE+': '+str(e))

    # Create Eigen.inc
    try:
      self.logPrintBox('Creating Eigen.inc in ' +self.PACKAGE)
      eigenDir = os.path.join(self.installDir, 'include', 'eigen3', 'Eigen')
      files    = []
      for root, directories, filenames in os.walk(eigenDir):
        for filename in filenames:
          files.append(os.path.join(root, filename))
      with open(os.path.join(self.packageDir, 'Grid', 'Eigen.inc'), 'w') as f:
        f.write('eigen_files =\\\n')
        for filename in files[:-1]:
          f.write('  ' + os.path.join(root, filename) + ' \\\n')
        f.write('  ' + os.path.join(root, files[-1]) + '\n')

    except RuntimeError as e:
      raise RuntimeError('Error creating Eigen.inc in ' + self.PACKAGE+': '+str(e))

    try:
      self.logPrintBox('Generating Make.inc files for ' +self.PACKAGE+'; this may take several minutes')
      output,err,ret = config.base.Configure.executeShellCommand('./scripts/filelist', cwd=self.packageDir, timeout=100, log=self.log)
      if ret:
        raise RuntimeError('Error generating Make.inc: ' + output+err)
    except RuntimeError as e:
      raise RuntimeError('Error generating Make.inc in ' + self.PACKAGE+': '+str(e))
    config.package.GNUPackage.preInstall(self)
    return
