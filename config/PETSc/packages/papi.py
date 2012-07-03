import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions     = ['PAPI_library_init']
    self.includes      = ['papi.h']
    self.liblist       = [['libpapi.a','libperfctr.a']]
