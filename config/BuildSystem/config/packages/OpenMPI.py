import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version                = '4.1.4'
    self.download               = ['https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-'+self.version+'.tar.gz',
                                   'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/openmpi-'+self.version+'.tar.gz']
    self.downloaddirnames       = ['openmpi','ompi']
    self.skippackagewithoptions = 1
    self.isMPI                  = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.cuda           = framework.require('config.packages.cuda',self)
    self.hwloc          = framework.require('config.packages.hwloc',self)
    self.odeps          = [self.hwloc]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--with-rsh=ssh')
    args.append('--disable-man-pages')
    args.append('MAKE='+self.make.make)
    if not hasattr(self.compilers, 'CXX'):
      raise RuntimeError('Error: OpenMPI requires C++ compiler. None specified')
    if hasattr(self.compilers, 'FC'):
      self.pushLanguage('FC')
      if not self.fortran.fortranIsF90:
        args.append('--disable-mpi-f90')
        args.append('FC=""')
      self.popLanguage()
    else:
      args.append('--disable-mpi-f77')
      args.append('--disable-mpi-f90')
      args.append('F77=""')
      args.append('FC=""')
      args.append('--enable-mpi-fortran=no')
    if not self.argDB['with-shared-libraries']:
      args.append('--enable-shared=no')
      args.append('--enable-static=yes')
    args.append('--disable-vt')
    if self.cuda.found:
      args.append('--with-cuda='+self.cuda.cudaDir)
    if self.hwloc.found:
      args.append('--with-hwloc="'+self.hwloc.directory+'"')
    else:
      args.append('--with-hwloc=internal')
    # https://www.open-mpi.org/faq/?category=building#libevent-or-hwloc-errors-when-linking-fortran
    args.append('--with-libevent=internal')
    return args

  def updateGitDir(self):
    import os
    config.package.GNUPackage.updateGitDir(self)
    if not hasattr(self.sourceControl, 'git') or (self.packageDir != os.path.join(self.externalPackagesDir,'git.'+self.package)):
      return
    Dir = self.getDir()
    try:
      thirdparty = self.thirdparty
    except AttributeError:
      try:
        self.executeShellCommand([self.sourceControl.git, 'submodule', 'update', '--init', '--recursive'], cwd=Dir, log=self.log)
        import os
        if os.path.isfile(os.path.join(Dir,'3rd-party','openpmix','README')):
          self.thirdparty = os.path.join(Dir,'3rd-party')
        else:
          raise RuntimeError
      except RuntimeError:
        raise RuntimeError('Could not initialize 3rd-party submodule needed by OpenMPI')
    return

  def preInstall(self):
    self.Bootstrap('AUTOMAKE_JOBS=%d ./autogen.pl' % self.make.make_np)

  def checkDownload(self):
    if config.setCompilers.Configure.isCygwin(self.log):
      if config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log):
        raise RuntimeError('Cannot download-install OpenMPI on Windows with cygwin compilers. Suggest installing OpenMPI via cygwin installer')
      else:
        raise RuntimeError('Cannot download-install OpenMPI on Windows with Microsoft or Intel Compilers. Suggest using MS-MPI or Intel-MPI (do not use MPICH2')
    if self.argDB['download-'+self.downloadname.lower()] and  'package-prefix-hash' in self.argDB and self.argDB['package-prefix-hash'] == 'reuse':
      self.logWrite('Reusing package prefix install of '+self.defaultInstallDir+' for OpenMPI')
      self.installDir = self.defaultInstallDir
      self.updateCompilers(self.installDir,'mpicc','mpicxx','mpif77','mpif90')
      return self.installDir
    if self.argDB['download-'+self.downloadname.lower()]:
      return self.getInstallDir()
    return ''

  def Install(self):
    '''After downloading and installing OpenMPI we need to reset the compilers to use those defined by the OpenMPI install'''
    if 'package-prefix-hash' in self.argDB and self.argDB['package-prefix-hash'] == 'reuse':
      return self.defaultInstallDir
    installDir = config.package.GNUPackage.Install(self)
    self.updateCompilers(installDir,'mpicc','mpicxx','mpif77','mpif90')
    return installDir

