import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version           = '2.4.0'
    self.download          = ['http://www.open-mpi.org/software/hwloc/v2.4/downloads/hwloc-'+self.version+'.tar.gz',
                              'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/hwloc-'+self.version+'.tar.gz']
    self.functions         = ['hwloc_topology_init']
    self.includes          = ['hwloc.h']
    self.liblist           = [['libhwloc.a'],['libhwloc.a','libxml2.a']]
    self.downloadonWindows = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.x       = framework.require('config.packages.X',self)
    self.odeps   = [self.x]
    return

  def getSearchDirectories(self):
    yield ''
    yield os.path.join('/usr','local')
    yield os.path.join('/opt','local')
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if self.x.found:
      args.append('--with-x=yes')
    else:
      args.append('--with-x=no')
    # don't require unneeded external dependency
    args.append('--disable-libxml2')
    args.append('--disable-opencl')
    args.append('--disable-cuda')
    args.append('--disable-nvml')
    args.append('--disable-gl')
    args.append('CPPFLAGS="'+self.headers.toStringNoDupes(self.dinclude)+'"')
    args.append('LIBS="'+self.libraries.toStringNoDupes(self.dlib)+'"')
    return args

  def configure(self):
    config.package.GNUPackage.configure(self)
    if self.found and self.directory:
      self.getExecutable('lstopo',    path=os.path.join(self.directory,'bin'), getFullPath = 1)



