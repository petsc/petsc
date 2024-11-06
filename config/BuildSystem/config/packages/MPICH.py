import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version          = '4.2.3'
    self.download         = ['https://github.com/pmodels/mpich/releases/download/v'+self.version+'/mpich-'+self.version+'.tar.gz',
                             'https://www.mpich.org/static/downloads/'+self.version+'/mpich-'+self.version+'.tar.gz', # does not always work from Python? So add in web.cels URL below
                             'https://web.cels.anl.gov/projects/petsc/download/externalpackages'+'/mpich-'+self.version+'.tar.gz']
    self.download_git     = ['git://https://github.com/pmodels/mpich.git']
    self.versionname      = 'MPICH_NUMVERSION'
    self.includes         = ['mpi.h']
    self.gitsubmodules    = ['.']
    self.downloaddirnames = ['mpich']
    self.skippackagewithoptions = 1
    self.isMPI = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags',self)
    self.cuda            = framework.require('config.packages.cuda',self)
    self.hip             = framework.require('config.packages.hip',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.python          = framework.require('config.packages.python',self)
    self.odeps           = [self.cuda, self.hip, self.hwloc]
    return

  def versionToStandardForm(self,ver):
    '''Converts from MPICH 10007201 notation to standard notation 1.0.7'''
    # See the format at https://github.com/pmodels/mpich/blob/main/src/include/mpi.h.in#L78
    # 1 digit for MAJ, 2 digits for MIN, 2 digits for REV, 1 digit for EXT and 2 digits for EXT_NUMBER
    return ".".join(map(str,[int(ver)//10000000, int(ver)//100000%100, int(ver)//1000%100]))

  def setupHelp(self, help):
    config.package.GNUPackage.setupHelp(self,help)
    import nargs
    help.addArgument('MPICH', '-download-mpich-pm=<hydra, gforker or mpd>',              nargs.Arg(None, 'hydra', 'Launcher for MPI processes'))
    help.addArgument('MPICH', '-download-mpich-device=<ch3:nemesis or see MPICH docs>', nargs.Arg(None, None, 'Communicator for MPI processes'))
    return

  def checkDownload(self):
    if config.setCompilers.Configure.isCygwin(self.log):
      if config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log):
        raise RuntimeError('Cannot download-install MPICH on Windows with cygwin compilers. Suggest installing Open MPI via cygwin installer')
      else:
        raise RuntimeError('Cannot download-install MPICH on Windows with Microsoft or Intel Compilers. Suggest using MS-MPI or Intel-MPI (do not use MPICH2')
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
    args.append('--with-pm='+self.argDB['download-mpich-pm'])
    args.append('--disable-java')
    if self.hwloc.found:
      args.append('--with-hwloc="'+self.hwloc.directory+'"')
      args.append('--with-hwloc-prefix="'+self.hwloc.directory+'"')
    elif 'with-hwloc' in self.framework.clArgDB and not self.argDB['with-hwloc'] :
      args.append('--without-hwloc')
    else:
      args.append('--with-hwloc=embedded')
    # make sure MPICH does not build with optimization for debug version of PETSc, so we can debug through MPICH
    if self.compilerFlags.debugging:
      args.append("--enable-fast=no")
      args.append("--enable-error-messages=all")
      mpich_device = 'ch3:sock'
    else:
      mpich_device = 'ch3:nemesis'
    if self.cuda.found:
      args.append('--with-cuda='+self.cuda.cudaDir)
      if hasattr(self.cuda,'cudaArch'): # MPICH's default to --with-cuda-sm=XX is 'auto', to auto-detect the arch of the visible GPUs (similar to our `native`).
        if self.cuda.cudaArch == 'all':
          args.append('--with-cuda-sm=all-major') # MPICH stopped supporting 'all' thus we do it with 'all-major'
        else:
          args.append('--with-cuda-sm='+self.cuda.cudaArch)
      mpich_device = 'ch4:ucx'
    elif self.hip.found:
      args.append('--with-hip='+self.hip.hipDir)
      mpich_device = 'ch4:ofi' # per https://github.com/pmodels/mpich/wiki/Using-MPICH-on-Crusher@OLCF

    if 'download-mpich-device' in self.argDB:
      mpich_device = self.argDB['download-mpich-device']
    args.append('--with-device='+mpich_device)
    # meminit: preinitialize memory associated structures and unions to eliminate access warnings from programs like valgrind
    # dbg: add compiler flag, -g, to all internal compiler flag i.e. MPICHLIB_CFLAGS, MPICHLIB_CXXFLAGS, MPICHLIB_FFLAGS, and MPICHLIB_FCFLAGS, to make debugging easier
    args.append('--enable-g=meminit,dbg')
    if not self.setCompilers.isDarwin(self.log) and config.setCompilers.Configure.isClang(self.setCompilers.CC, self.log):
      args.append('pac_cv_have_float16=no')
    if config.setCompilers.Configure.isDarwin(self.log):
      args.append('--disable-opencl')

    # MPICH configure errors out on certain standard configure arguments
    args = self.rmArgs(args,['--disable-f90','--enable-f90'])
    args = self.rmArgsStartsWith(args,['F90=','F90FLAGS='])
    args.append('PYTHON='+self.python.pyexe)
    args.append('--disable-maintainer-mode')
    args.append('--disable-dependency-tracking')
    return args

  def gitPreReqCheck(self):
    return self.programs.autoreconf and self.programs.libtoolize

  def preInstall(self):
    if self.retriever.isDirectoryGitRepo(self.packageDir):
      # no need to bootstrap tarballs
      self.Bootstrap('./autogen.sh')

  def Install(self):
    '''After downloading and installing MPICH we need to reset the compilers to use those defined by the MPICH install'''
    if 'package-prefix-hash' in self.argDB and self.argDB['package-prefix-hash'] == 'reuse':
      return self.defaultInstallDir
    installDir = config.package.GNUPackage.Install(self)
    self.updateCompilers(installDir,'mpicc','mpicxx','mpif77','mpif90')
    return installDir

  def configure(self):
    return config.package.Package.configure(self)
