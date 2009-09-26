import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions = ['nccreate']
    self.includes  = ['netcdf.h']
    self.liblist   = [['libnetcdf.a']]
    self.complex   = 1
    return
