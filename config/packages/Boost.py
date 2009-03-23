import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/boost.tar.gz']
    self.includes        = ['boost/multi_index_container.hpp']
    self.cxx             = 1
    self.includedir      = ''
    self.archIndependent = 1
    return

  def Install(self):
    import sys
    boostDir = self.getDir()
    self.framework.actions.addArgument('Boost', 'Install', 'Installed Boost into '+boostDir)
    return boostDir
