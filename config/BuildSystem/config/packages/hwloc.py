import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download          = ['http://www.open-mpi.org/software/hwloc/v1.9/downloads/hwloc-1.9.1.tar.gz']
    self.functions         = ['hwloc_topology_init']
    self.includes          = ['hwloc.h']
    self.liblist           = [['libhwloc.a']]
    self.downloadonWindows = 1
    return

  def formGNUConfigureArgs(self):
    '''Don't require x libraries since they may not always be available or hwloc may not be able to locate them (on Apple)'''
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--without-x')
    return args



