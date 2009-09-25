import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download = ['http://www.fenics.org/pub/software/ffc/v0.3/ffc-0.3.3.tar.gz']
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.fiat = self.framework.require('config.packages.Fiat', self)
    self.deps = [self.fiat]
    return

  def Install(self):
    import sys
    ffcDir = self.getDir()
    # We could make a check of the md5 of the current configure framework
    self.logPrintBox('FFC needs no installation')
    self.framework.actions.addArgument('FFC', 'Install', 'Installed FFC into '+ffcDir)
    return ffcDir

  def configureLibrary(self):
    '''Find an installation ando check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.log.write('Checking for a functional '+self.name+'\n')

    for location, dir, lib, incl in self.generateGuesses():
      try:
        import FFC
      except ImportError, e:
        self.framework.logPrint('ERROR: Could not import FFC: '+str(e))
    raise RuntimeError('Could not find a functional '+self.name+'\n')
