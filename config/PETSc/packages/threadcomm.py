import PETSc.package
import os

# This is a temporary file for defining the flag PETSC_THREADCOMM_ACTIVE.
# It should be deleted when the flag is removed from the code.

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps    = []
    return

  def configureLibrary(self):
    raise RuntimeError('Please use petsc master (git branch) for threadcomm functionality')
    self.addDefine('THREADCOMM_ACTIVE',1)
    return
