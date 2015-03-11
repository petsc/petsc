import config.package
import os, sys

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/ScientificPythonSimple.tar.gz']
    self.downloadonWindows = 1
    self.liblist           = [['Derivatives.py']]
    self.libdir            = os.path.join('lib', 'python', 'site-packages')
    self.altlibdir         = os.path.join('lib', 'python'+'.'.join(map(str, sys.version_info[0:2])), 'site-packages')
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps = []
    return

  def Install(self):
    import shutil
    # We could make a check of the md5 of the current configure framework
    self.logPrintBox('Installing Scientific Python (PETSc version)')
    # Copy ScientificPython into $PETSC_ARCH/lib/python2.*/site-packages
    installLoc = os.path.join(self.installDir, self.altlibdir)
    initfile   = os.path.join(installLoc, '__init__.py')
    packageDir = os.path.join(installLoc, 'Scientific')
    if self.installSudo:
      self.installDirProvider.printSudoPasswordMessage()
      try:
        output,err,ret  = config.base.Configure.executeShellCommand(self.installSudo+'mkdir -p '+installLoc+' && '+self.installSudo+'rm -rf '+packageDir+'  && '+self.installSudo+'cp -rf '+os.path.join(self.packageDir, 'Scientific')+' '+packageDir, timeout=6000, log = self.log)
      except RuntimeError, e:
        raise RuntimeError('Error copying Scientific Python files from '+os.path.join(self.packageDir, 'Scientific')+' to '+packageDir)
    else:
      if not os.path.isdir(installLoc):
        os.makedirs(installLoc)
      if not os.path.exists(initfile):
        f = file(initfile, 'w')
        f.write('')
        f.close()
      if os.path.exists(packageDir):
        shutil.rmtree(packageDir)
      shutil.copytree(os.path.join(self.packageDir, 'Scientific'), packageDir)
    self.framework.actions.addArgument('Scientific Python', 'Install', 'Installed Scientific Python into '+self.installDir)
    return self.installDir

  def configureLibrary(self):
    '''Find an installation ando check if it can work with PETSc'''
    import sys
    self.log.write('==================================================================================\n')
    self.log.write('Checking for a functional '+self.name+'\n')

    self.checkDependencies()
    for location, rootDir, lib, incDir in self.generateGuesses():
      try:
        libDir = os.path.dirname(lib[0])
        self.logPrint('Checking location '+location)
        self.logPrint('Added directory '+libDir+' to Python path')
        sys.path.insert(0, libDir)
        import Scientific.Functions.Derivatives
        self.found = 1
        return
      except ImportError, e:
        self.logPrint('ERROR: Could not import Scientific Python: '+str(e))
    raise RuntimeError('Could not find a functional '+self.name+'\n')
