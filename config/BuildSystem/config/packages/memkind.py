import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = ['hbw_malloc']
    self.includes          = ['hbwmalloc.h']
    self.liblist           = [['libmemkind.a']]
    self.double            = 0   # 1 means requires double precision
