import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = '993719694528eec103be499d263e274696ff58d9'  #devel aug-5-2020
    self.download               = ['git://https://github.com/idaholab/moose.git']
    self.functions              = []
    self.includes               = []
    self.skippackagewithoptions = 1
    self.useddirectly           = 0
    self.linkedbypetsc          = 0
    self.builtafterpetsc        = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    self.libmesh         = framework.require('config.packages.libmesh',self)
    return

  def Install(self):
    import os
    if self.argDB['prefix']:
      self.logPrintBox('MOOSE does not support --prefix installs yet with PETSc')
    else:
      self.logPrintBox('MOOSE is available at '+os.path.join('${PETSC_DIR}',self.arch,'externalpackages','git.moose')+'\nexport PACKAGES_DIR='+os.path.join('${PETSC_DIR}',self.arch,'externalpackages','git.moose')+'\nexport LIBMESH_DIR='+os.path.join('${PETSC_DIR}',self.arch)+'\n')
    return self.installDir
