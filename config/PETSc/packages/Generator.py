import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/Generator.tar.gz']
    self.complex  = 1
    self.double   = 0
    self.worksonWindows    = 1
    self.downloadonWindows = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = []
    return

  def Install(self):
    import os
    generatorDir = self.getDir()
    # We could make a check of the md5 of the current configure framework
    self.logPrintBox('Generator needs no installation')
    self.framework.actions.addArgument('Generator', 'Install', 'Installed Generator into '+generatorDir)
    return generatorDir

  def configureLibrary(self):
    '''Find an installation ando check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    self.framework.log.write('Checking for a functional '+self.name+'\n')

    for location, dir, lib, incl in self.generateGuesses():
      try:
        import sys
        sys.path.insert(0, dir)
        import Cxx
        import CxxHelper
        return
      except ImportError, e:
        self.framework.logPrint('ERROR: Could not import Generator: '+str(e))
        self.framework.logPrint('  from directory '+str(dir))
    raise RuntimeError('Could not find a functional '+self.name+'\n')
