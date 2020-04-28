import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download         = ['https://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpich-3.3.2.tar.gz']
    self.download_31      = ['http://www.mpich.org/static/downloads/3.1/mpich-3.1.tar.gz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpich-3.1.tar.gz']
    self.downloaddirnames  = ['mpich']
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
      if config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log):
        if self.argDB['with-shared-libraries']:
          raise RuntimeError('Sorry, --download-mpich does not work with shared-libraries. Suggest installing OpenMPI via cygwin or use --with-shared-libraries=0')
      else:
        raise RuntimeError('Sorry, cannot download-install MPICH on Windows with Microsoft or Intel Compilers. Suggest using MS-MPI or Intel-MPI (do not use MPICH2')
    if self.argDB['download-'+self.downloadname.lower()] and  'package-prefix-hash' in self.argDB and self.argDB['package-prefix-hash'] == 'reuse':
      self.logWrite('Reusing package prefix install of '+self.defaultInstallDir+' for MPICH')
      self.installDir = self.defaultInstallDir
      self.updateCompilers(self.installDir,'mpicc','mpicxx','mpif77','mpif90')
      return self.installDir
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
      args.append("--enable-error-messages=all")
    # make MPICH behave properly for valgrind
    args.append('--enable-g=meminit')
    if not self.sharedLibraries.useShared and config.setCompilers.Configure.isDarwin(self.log):
      args.append('--disable-opencl')

    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      if config.setCompilers.Configure.isNAG(self.setCompilers.getLinker(), self.log):
        args = self.addArgStartsWith(args,'FFLAGS','-mismatch')
      elif config.setCompilers.Configure.isGfortran100plus(self.setCompilers.getCompiler(), self.log):
        args = self.addArgStartsWith(args,'FFLAGS','-fallow-argument-mismatch')
      self.setCompilers.popLanguage()

    # MPICH configure errors out on certain standard configure arguments
    args = self.rmArgs(args,['--disable-f90','--enable-f90'])
    args = self.rmArgsStartsWith(args,['F90=','F90FLAGS='])
    return args

  def Install(self):
    '''After downloading and installing MPICH we need to reset the compilers to use those defined by the MPICH install'''
    if 'package-prefix-hash' in self.argDB and self.argDB['package-prefix-hash'] == 'reuse':
      return self.defaultInstallDir
    installDir = config.package.GNUPackage.Install(self)
    self.updateCompilers(installDir,'mpicc','mpicxx','mpif77','mpif90')
    return installDir

  def configure(self):
    if config.setCompilers.Configure.isCygwin(self.log) and config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log):
      self.download = self.download_31
    return config.package.Package.configure(self)

