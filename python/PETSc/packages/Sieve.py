import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download        = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/sieve.tar.gz']
    self.includes        = ['Mesh.hh']
    self.includedir      = ''
    self.archIndependent = 1
    self.cxx             = 1
    self.complex         = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.boost = self.framework.require('PETSc.packages.Boost',self)
    #self.boost = self.framework.require('PETSc.packages.Triangle',self)
    #self.boost = self.framework.require('PETSc.packages.TetGen',self)
    self.deps = [self.boost]
    return

  def Install(self):
    import sys
    sieveDir = self.getDir()
    self.framework.actions.addArgument('Sieve', 'Install', 'Installed Sieve into '+sieveDir)
    return sieveDir
