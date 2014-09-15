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
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    # Don't require x libraries since they may not always be available or hwloc may not be able to locate them on Apple
    if self.setCompilers.isDarwin():
      args.append('--without-x')
    return args

  def configure(self):
    '''Download by default '''
    if self.framework.clArgDB.has_key('download-hwloc') and not self.framework.argDB['download-hwloc']:
      self.framework.logPrint("Not downloading hwloc on user request\n")
      return
    if self.argDB['with-batch']:
      return
    self.framework.argDB['download-hwloc'] = 1
    config.package.GNUPackage.configure(self)



