import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = ['hbw_malloc']
    self.includes          = ['hbwmalloc.h']
    self.liblist           = [['libmemkind.a']]
    self.lookforbydefault  = 0
    self.double            = 0   # 1 means requires double precision
    self.complex           = 1   # 0 means cannot use complex
