import config.base
import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions  = ['cg_close']
    self.includes   = ['cgnslib.h']
    self.liblist    = [['libcgns.a'],
                       ['libcgns.a', 'libhdf5.a']] # Sometimes they over-link

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.hdf5 = framework.require('config.packages.hdf5',self)
    return

  def generateLibList(self, framework):
    '''First try library list without compression libraries (zlib) then try with'''
    if self.hdf5.found:
      if self.hdf5.lib:
        self.liblist.append(['libcgns.a']+self.hdf5.lib)
    return config.package.Package.generateLibList(self, framework)
