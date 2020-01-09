import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download          = ['http://www.open-mpi.org/software/hwloc/v1.11/downloads/hwloc-2.1.0.tar.gz',
                              'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/hwloc-2.1.0.tar.gz']
    self.functions         = ['hwloc_topology_init']
    self.includes          = ['hwloc.h']
    self.liblist           = [['libhwloc.a'],['libhwloc.a','libxml2.a']]
    self.downloadonWindows = 1
    return

  def getSearchDirectories(self):
    yield ''
    yield os.path.join('/usr','local')
    yield os.path.join('/opt','local')
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    # Don't require x libraries since they may not always be available or hwloc may not be able to locate them on Apple
    if self.setCompilers.isDarwin(self.log):
      args.append('--without-x')
    # don't require unneeded external dependency
    args.append('--disable-libxml2')
    return args

  def configure(self):
    config.package.GNUPackage.configure(self)
    if self.found and self.directory:
      self.getExecutable('lstopo',    path=os.path.join(self.directory,'bin'), getFullPath = 1)



