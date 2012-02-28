import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions = ['DSC_ErrorDisplay']
    self.includes  = ['dscmain.h']
    self.liblist   = [['dsclibdbl.a']]
    self.noMPIUni  = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = [self.mpi]
    return
