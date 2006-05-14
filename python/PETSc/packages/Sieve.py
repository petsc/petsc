import PETSc.package

import os

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    #self.download        = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/sieve.tar.gz']
    self.includes        = ['Mesh.hh']
    self.includedir      = ['include', os.path.join('include', 'mpiuni'), os.path.join('bmake', 'docsonly'), os.path.join('src', 'dm', 'mesh', 'sieve')]
    self.libdir          = ''
    self.archIndependent = 1
    self.cxx             = 1
    self.complex         = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.petscdir = self.framework.require('PETSc.utilities.petscdir',self)
    self.boost    = self.framework.require('config.packages.Boost',self)
    #self.boost = self.framework.require('PETSc.packages.Triangle',self)
    #self.boost = self.framework.require('PETSc.packages.TetGen',self)
    self.deps = [self.boost]
    return

  def getSearchDirectories(self):
    return [self.petscdir.dir]

  def Install(self):
    import sys
    sieveDir = self.getDir()
    self.framework.actions.addArgument('Sieve', 'Install', 'Installed Sieve into '+sieveDir)
    return sieveDir
