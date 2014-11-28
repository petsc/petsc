import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download          = ['http://www.open-mpi.org/software/hwloc/v1.10/downloads/hwloc-1.10.0.tar.gz']
    self.functions         = ['hwloc_topology_init']
    self.includes          = ['hwloc.h']
    self.liblist           = [['libhwloc.a'],['libhwloc.a','libxml2.a']]
    self.downloadonWindows = 1
    return

  def getSearchDirectories(self):
    yield ''
    yield '/usr'
    yield os.path.join('/usr','local')
    yield os.path.join('/opt','local')
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    # Don't require x libraries since they may not always be available or hwloc may not be able to locate them on Apple
    if self.setCompilers.isDarwin():
      args.append('--without-x')
    # don't require unneeded external dependency
    args.append('--disable-libxml2')
    return args

  def configure(self):
    '''Searches for hwloc and if not found downloads by default and just continue if it does not build '''
    if self.framework.clArgDB.has_key('with-hwloc') and not self.framework.argDB['with-hwloc']:
      self.framework.logPrint("Not using hwloc on user request\n")
      return
    if self.framework.clArgDB.has_key('download-hwloc') and not self.framework.argDB['download-hwloc']:
      self.framework.logPrint("Not downloading hwloc on user request\n")
      return
    # if PETSc libraries start using hwloc directly then we should remove the following if test
    if self.argDB['with-batch']:
      return
    self.framework.argDB['with-hwloc'] = 1
    try:
      config.package.GNUPackage.configure(self)
    except:
      self.found = 0
    if not self.found:
      if (not self.framework.argDB['download-hwloc']): self.framework.argDB['download-hwloc'] = 1
      try:
        config.package.GNUPackage.configure(self)
      except:
        self.found = 0
    if self.found and self.directory:
      self.getExecutable('lstopo',    path=os.path.join(self.directory,'bin'), getFullPath = 1)



