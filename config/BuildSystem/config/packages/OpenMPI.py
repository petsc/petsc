import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.downloadpath     = 'http://www.open-mpi.org/software/ompi/v1.8/downloads/'
    self.downloadname     = 'openmpi'
    self.downloadext      = 'tar.gz'
    self.downloadversion  = '1.8.1'
    self.downloadfilename = 'openmpi'
    return

  def formGNUConfigureExtraArgs(self):
    args = ['--with-rsh=ssh']
    args.append('MAKE='+self.make.make)
    if not hasattr(self.compilers, 'CXX'):
      raise RuntimeError('Error: OpenMPI requires C++ compiler. None specified')
    if hasattr(self.compilers, 'FC'):
      self.pushLanguage('FC')
      if not self.compilers.fortranIsF90:
        args.append('--disable-mpi-f90')
        args.append('FC=""')
      self.popLanguage()
    else:
      args.append('--disable-mpi-f77')
      args.append('--disable-mpi-f90')
      args.append('F77=""')
      args.append('FC=""')
    if not self.framework.argDB['with-shared-libraries']:
      args.append('--enable-shared=no')
      args.append('--enable-static=yes')
    args.append('--disable-vt')
    return args

  def Install(self):
    '''After downloading and installing OpenMPI we need to reset the compilers to use those defined by the OpenMPI install'''
    installDir = config.package.GNUPackage.Install(self)
    self.updateCompilers(installDir,'mpicc','mpicxx','mpif77','mpif90')
    return installDir

