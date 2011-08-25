# Temporary package file to enable Rosenbrock-W methods. This can be removed along with all reference to PETSC_HAVE_ROSW when the code is stable.

import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions         = 0
    self.includes          = 0
    self.liblist           = 0
    self.archIndependent   = 1
    self.complex           = 1
    self.worksonWindows    = 1
    self.complex           = 1
    self.double            = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps    = []
    return

  def configureLibrary(self):
    self.addDefine('HAVE_ROSW',1)
    self.found = 1
    self.framework.packages.append(self)
