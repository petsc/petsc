import PETSc.package
import sys

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions        = ['clGetPlatformIDs']
    if sys.platform.startswith('darwin'): # Apple requires special care (OpenCL/cl.h)
      self.includes         = ['OpenCL/cl.h']
    else:
      self.includes         = ['CL/cl.h']
    self.liblist          = [['libOpenCL.a'], ['libOpenCL.a', '-framework opencl'], ['libOpenCL.lib']]
    self.double           = 0   # 1 means requires double precision
    self.cxx              = 0
    self.requires32bitint = 0
    self.worksonWindows   = 1

    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = []
    return

  def getSearchDirectories(self):
    yield ''
    return


