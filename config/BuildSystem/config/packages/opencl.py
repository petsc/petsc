import config.package
import sys

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions        = ['clGetPlatformIDs']
    if sys.platform.startswith('darwin'): # Apple requires special care (OpenCL/cl.h)
      self.includes         = ['OpenCL/cl.h']
    else:
      self.includes         = ['CL/cl.h']
    self.liblist          = [['libOpenCL.a'], ['-framework opencl'], ['libOpenCL.lib']]
