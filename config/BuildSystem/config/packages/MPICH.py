import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download         = ['http://www.mpich.org/static/downloads/3.1.3/mpich-3.1.3.tar.gz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpich-3.1.3.tar.gz']
    self.download_cygwin  = ['http://www.mpich.org/static/downloads/3.1/mpich-3.1.tar.gz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpich-3.1.tar.gz']
    self.download_osx     = ['http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpich-3.2.tar.gz']
    self.downloaddirname  = 'mpich'
    self.skippackagewithoptions = 1
    self.isMPI = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    return

  def setupHelp(self, help):
    config.package.GNUPackage.setupHelp(self,help)
    import nargs
    help.addArgument('MPI', '-download-mpich-pm=<hydra, gforker or mpd>',              nargs.Arg(None, 'hydra', 'Launcher for MPI processes'))
    help.addArgument('MPI', '-download-mpich-device=<ch3:nemesis or see mpich2 docs>', nargs.Arg(None, 'ch3:sock', 'Communicator for MPI processes'))
    return

  def checkDownload(self):
    if config.setCompilers.Configure.isCygwin(self.log):
      if not config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log):
        raise RuntimeError('Sorry, cannot download-install MPICH on Windows with Microsoft or Intel Compilers. Suggest installing Windows version of MPICH manually')
    if self.argDB['download-'+self.downloadname.lower()]:
      return self.getInstallDir()
    return ''


  def formGNUConfigureArgs(self):
    '''MPICH has many specific extra configure arguments'''
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if 'download-mpich-device' in self.argDB:
      args.append('--with-device='+self.argDB['download-mpich-device'])
    args.append('--with-pm='+self.argDB['download-mpich-pm'])
    # make sure MPICH does not build with optimization for debug version of PETSc, so we can debug through MPICH
    if self.compilerFlags.debugging:
      args.append("--enable-fast=no")
    # make MPICH behave properly for valgrind
    args.append('--enable-g=meminit')
    # MPICH configure errors out on certain standard configure arguments
    args = self.rmArgs(args,['--disable-f90','--enable-f90'])
    args = self.rmArgsStartsWith(args,['F90=','F90FLAGS='])
    return args

  def Install(self):
    '''After downloading and installing MPICH we need to reset the compilers to use those defined by the MPICH install'''
    installDir = config.package.GNUPackage.Install(self)
    self.updateCompilers(installDir,'mpicc','mpicxx','mpif77','mpif90')
    return installDir

  def configure(self):
    if config.setCompilers.Configure.isCygwin(self.log) and config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log):
      self.download = self.download_cygwin
    elif self.setCompilers.isDarwin(self.log):
      (output, error, status) = config.base.Configure.executeShellCommand('uname -r')
      ver = tuple(map(int,output.split('.')))
      if ver >= (15,0,0): # ElCapitan/10.11
        self.download = self.download_osx
    return config.package.Package.configure(self)

