import config.package
import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit = 'origin/master'
    self.download  = ['git://https://bitbucket.org/saws/saws.git','https://bitbucket.org/saws/saws/get/master.tar.gz']
    self.functions = ['SAWs_Register']
    self.includes  = ['SAWs.h']
    self.liblist   = [['libSAWs.a']]
    self.libdir           = 'lib' # location of libraries in the package directory tree
    self.includedir       = 'include' # location of includes in the package directory tree    return
    self.needsMath        = 1;

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    return


