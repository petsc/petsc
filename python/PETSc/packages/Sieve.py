import PETSc.package

import os

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    #self.download        = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/sieve.tar.gz']
    self.includes        = ['Mesh.hh']
    self.includedir      = ['include', os.path.join('bmake', 'docsonly'), os.path.join('src', 'dm', 'mesh', 'sieve')]
    self.libdir          = ''
    self.archIndependent = 1
    self.cxx             = 1
    self.complex         = 1
    self.required        = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.petscdir = self.framework.require('PETSc.utilities.petscdir',self)
    self.boost    = self.framework.require('config.packages.Boost',self)
    self.mpi      = self.framework.require('config.packages.MPI',self)
    #self.triangle = self.framework.require('PETSc.packages.Triangle',self)
    #self.tetgen   = self.framework.require('PETSc.packages.TetGen',self)
    self.deps = [self.boost, self.mpi]
    return

  def getSearchDirectories(self):
    return [self.petscdir.dir]

  def Install(self):
    import sys
    sieveDir = self.getDir()
    self.framework.actions.addArgument('Sieve', 'Install', 'Installed Sieve into '+sieveDir)
    return sieveDir

  def configure(self):
    '''Determines if the package should be configured for, then calls the configure'''
    if not self.boost.found:
      self.logPrint('Disabling Sieve since Boost was not located')
      self.framework.argDB['with-'+self.package] = 0
    return PETSc.package.Package.configure(self)
