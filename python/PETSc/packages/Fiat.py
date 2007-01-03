import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download = ['http://www.fenics.org/pub/software/fiat/FIAT-0.2.5.tar.gz']
    self.downloadname = self.name.upper()
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.deps = []
    return

  def Install(self):
    fiatDir = self.getDir()
    # We could make a check of the md5 of the current configure framework
    self.logPrintBox('FIAT needs no installation')
    self.framework.actions.addArgument('FIAT', 'Install', 'Installed FIAT into '+fiatDir)
    return fiatDir

  def configureLibrary(self):
    '''Find an installation ando check if it can work with PETSc'''
    import os, sys
    self.framework.log.write('==================================================================================\n')
    self.framework.log.write('Checking for a functional '+self.name+'\n')

    for location, dir, lib, incl in self.generateGuesses():
      try:
        sys.path.insert(0, os.path.dirname(dir))
        print 'Path',sys.path
        import FIAT
        print FIAT.__file__
        import FIAT.shapes
        import FIAT.Lagrange
        import FIAT.quadrature
        return
      except ImportError, e:
        self.framework.logPrint('ERROR: Could not import FIAT: '+str(e))
    raise RuntimeError('Could not find a functional '+self.name+'\n')
