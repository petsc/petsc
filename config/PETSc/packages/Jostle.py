import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions = ['pjostle']
    self.includes  = ['jostle.h']
    self.liblist   = [['libjostle.lnx.a']]
    self.noMPIUni  = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = [self.mpi]
    return

  def downloadJostle(self):
    import os
    self.framework.logPrint('Downloading Jostle')
    try:
      jostleDir = self.getDir()  
      self.framework.logPrint('Jostle already downloaded, no need to ftp')
    except RuntimeError:
      import urllib
      packages = self.petscdir.externalPackagesDir 
      try:
        self.logPrintBox('Retrieving Jostle; this may take several minutes')
        urllib.urlretrieve('http://ftp.mcs.anl.gov/pub/petsc/externalpackages/jostle.tar.gz', os.path.join(packages, 'jostle.tar.gz'))
      except Exception, e:
        raise RuntimeError('Error downloading Jostle: '+str(e))
      try:
        PETSc.package.NewPackage.executeShellCommand('cd '+packages+'; gunzip jostle.tar.gz', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping jostle.tar.gz: '+str(e))
      try:
        PETSc.package.NewPackage.executeShellCommand('cd '+packages+'; tar -xf jostle.tar', log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf jostle.tar: '+str(e))
      os.unlink(os.path.join(packages, 'jostle.tar'))
      self.framework.actions.addArgument('Jostle', 'Download', 'Downloaded Jostle into '+self.getDir())
    # Configure and Build Jostle ? Jostle is already configured and build with gcc!
    jostleDir = self.getDir()
    lib     = [[os.path.join(jostleDir, 'libjostle.lnx.a')]] 
    include = [jostleDir]
    return ('Downloaded Jostle', lib, include)
