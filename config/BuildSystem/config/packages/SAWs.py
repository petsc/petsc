import config.package
import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit   = '028a32f0118f227c9264bebdc282ab71f8eb88db' # May 3, 2026
    self.download    = ['git://https://gitlab.com/petsc/saws.git','https://gitlab.com/petsc/saws/get/'+self.gitcommit+'.tar.gz']
    self.functions   = ['SAWs_Register']
    self.includes    = ['SAWs.h']
    self.liblist     = [['libSAWs.a']]
    self.testoptions = '-saws_port_auto_select -saws_port_auto_select_silent'

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.deps           = [self.mathlib]
    return
