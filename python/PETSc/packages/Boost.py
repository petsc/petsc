import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download        = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/boost.tar.gz']
    self.includes        = ['boost/multi_index_container.hpp']
    self.includedir      = ''
    self.archIndependent = 1
    self.cxx             = 1
    self.complex         = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.deps = []
    return

  def Install(self):
    import sys
    boostDir = self.getDir()
    self.framework.actions.addArgument('Boost', 'Install', 'Installed Boost into '+boostDir)
    return boostDir
