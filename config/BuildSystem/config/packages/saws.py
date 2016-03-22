import config.package
import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit = '312ccc1698cf6c489c0d1eff6db46f54bd9031b7'
    self.download  = ['git://https://bitbucket.org/saws/saws.git','https://bitbucket.org/saws/saws/get/'+self.gitcommit+'.tar.gz']
    self.functions = ['SAWs_Register']
    self.includes  = ['SAWs.h']
    self.liblist   = [['libSAWs.a']]
    self.libdir           = 'lib' # location of libraries in the package directory tree
    self.includedir       = 'include' # location of includes in the package directory tree    return
    self.needsMath        = 1;

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    return


