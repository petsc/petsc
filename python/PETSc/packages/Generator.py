import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/Generator.tar.gz']
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.deps = []
    return

  def Install(self):
    generatorDir = self.getDir()
    # We could make a check of the md5 of the current configure framework
    self.logPrintBox('Generator needs no installation')
    self.framework.actions.addArgument('Generator', 'Install', 'Installed Generator into '+generatorDir)
    return generatorDir

  def configureLibrary(self):
    '''Find an installation ando check if it can work with PETSc'''
    import os, sys
    self.framework.log.write('==================================================================================\n')
    self.framework.log.write('Checking for a functional '+self.name+'\n')

    for location, dir, lib, incl in self.generateGuesses():
      try:
        sys.path.insert(0, os.path.dirname(dir))
        import Cxx
        import CxxHelper
        return
      except ImportError, e:
        self.framework.logPrint('ERROR: Could not import Generator: '+str(e))
    raise RuntimeError('Could not find a functional '+self.name+'\n')
