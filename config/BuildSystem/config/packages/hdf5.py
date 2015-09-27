import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download     = ['http://www.hdfgroup.org/ftp/HDF5/prev-releases/hdf5-1.8.12/src/hdf5-1.8.12.tar.gz',
                         'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/hdf5-1.8.12.tar.gz']
    self.functions = ['H5T_init']
    self.includes  = ['hdf5.h']
    self.liblist   = [['libhdf5_hl.a', 'libhdf5.a']]
    self.needsMath = 1
    self.needsCompression = 0
    self.complex          = 1
    self.hastests         = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi  = framework.require('config.packages.MPI',self)
    self.deps = [self.mpi]
    return

  def generateLibList(self, framework):
    '''First try library list without compression libraries (zlib) then try with'''
    list = []
    for l in self.liblist:
      list.append(l)
    if self.libraries.compression:
      for l in self.liblist:
        list.append(l + self.libraries.compression)
    self.liblist = list
    return config.package.Package.generateLibList(self,framework)

  def formGNUConfigureArgs(self):
    ''' Add HDF5 specific --enable-parallel flag and enable Fortran if available '''
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--enable-parallel')
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      args.append('--enable-fortran')
      args.append('F9X="'+self.setCompilers.getCompiler()+'"')
      self.setCompilers.popLanguage()
    return args

  def configureLibrary(self):
    if hasattr(self.compilers, 'FC'):
      # PETSc does not need the Fortran interface, but some users will call the Fortran interface
      # and expect our standard linking to be sufficient.  Thus we try to link the Fortran
      # libraries, but fall back to linking only C.
      self.liblist = [['libhdf5hl_fortran.a','libhdf5_fortran.a'] + libs for libs in self.liblist] + self.liblist
    config.package.GNUPackage.configureLibrary(self)
    if self.libraries.check(self.dlib, 'H5Pset_fapl_mpio'):
      self.addDefine('HAVE_H5PSET_FAPL_MPIO', 1)
    return
