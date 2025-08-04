import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version                = '5.0.8'
    self.download               = ['https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-'+self.version+'.tar.gz',
                                   'https://web.cels.anl.gov/projects/petsc/download/externalpackages/openmpi-'+self.version+'.tar.gz']
    self.download_git           = ['git://https://github.com/open-mpi/ompi.git']
    self.versionname            = 'OMPI_MAJOR_VERSION.OMPI_MINOR_VERSION.OMPI_RELEASE_VERSION'
    self.includes               = ['mpi.h']
    self.gitsubmodules          = ['.']
    self.downloaddirnames       = ['openmpi','ompi']
    self.skippackagewithoptions = 1
    self.skipMPIDependency      = 1
    self.buildLanguages         = ['C','Cxx']
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.cuda           = framework.require('config.packages.CUDA',self)
    self.hip            = framework.require('config.packages.HIP',self)
    self.ucx            = framework.require('config.packages.ucx',self)
    self.hwloc          = framework.require('config.packages.hwloc',self)
    self.odeps          = [self.hwloc, self.cuda, self.hip, self.ucx]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--with-rsh=ssh')
    args.append('--disable-man-pages')
    args.append('--disable-sphinx')
    args.append('MAKE='+self.make.make)
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
      if not hasattr(self.cuda, 'cudaDir'):
        raise RuntimeError('CUDA directory not detected! Mail configure.log to petsc-maint@mcs.anl.gov.')
      args.append('--with-cuda='+self.cuda.cudaDir) # use openmpi's cuda support until it switches to ucx
    elif self.hip.found:
      if not self.ucx.found:
        self.logPrintWarning('Found ROCm but not UCX, so Open MPI will NOT be configured with ROCm support. Consider having UCX by simply adding --download-ucx')
      elif not self.ucx.enabled_rocm:
        self.logPrintWarning('Found ROCm and UCX, but UCX was not configured with ROCm, so Open MPI will NOT be configured with ROCm support. Consider using a ROCm-enabled UCX, or letting PETSc build one for you with --download-ucx')
      else:
        # see https://docs.open-mpi.org/en/main/tuning-apps/networking/rocm.html#building-open-mpi-with-rocm-support
        # One may need either mpirun -n 2 --mca pml ucx ./myapp or export OMPI_MCA_pml="ucx" to use UCX
        args.append('--with-rocm='+self.hip.hipDir)
        args.append('--with-ucx='+self.ucx.directory)
    if self.hwloc.found:
      args.append('--with-hwloc="'+self.hwloc.directory+'"')
    else:
      args.append('--with-hwloc=internal')
    # https://www.open-mpi.org/faq/?category=building#libevent-or-hwloc-errors-when-linking-fortran
    args.append('--with-libevent=internal')
    args.append('--with-pmix=internal')
    return args

  def preInstall(self):
    if not self.getExecutable('perl'):
      raise RuntimeError('Cannot find perl required by --download-openmpi, install perl (possibly with a package manager) and run ./configure again')
    self.Bootstrap('AUTOMAKE_JOBS=%d ./autogen.pl' % self.make.make_np)

  def checkDownload(self):
    if config.setCompilers.Configure.isCygwin(self.log):
      if config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log):
        raise RuntimeError('Cannot download-install Open MPI on Windows with cygwin compilers. Suggest installing Open MPI via cygwin installer')
      else:
        raise RuntimeError('Cannot download-install Open MPI on Windows with Microsoft or Intel Compilers. Suggest using MS-MPI or Intel-MPI (do not use MPICH2')
    if self.argDB['download-'+self.downloadname.lower()] and  'package-prefix-hash' in self.argDB and self.argDB['package-prefix-hash'] == 'reuse':
      self.logWrite('Reusing package prefix install of '+self.defaultInstallDir+' for Open MPI')
      self.installDir = self.defaultInstallDir
      self.updateCompilers(self.installDir,'mpicc','mpicxx','mpif77','mpif90')
      return self.installDir
    if self.argDB['download-'+self.downloadname.lower()]:
      return self.getInstallDir()
    return ''

  def Install(self):
    '''After downloading and installing Open MPI we need to reset the compilers to use those defined by the Open MPI install'''
    if 'package-prefix-hash' in self.argDB and self.argDB['package-prefix-hash'] == 'reuse':
      return self.defaultInstallDir
    installDir = config.package.GNUPackage.Install(self)
    self.updateCompilers(installDir,'mpicc','mpicxx','mpif77','mpif90')
    return installDir

