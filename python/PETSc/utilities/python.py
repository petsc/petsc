#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.usePython = 0
    return

  def __str__(self):
    if not self.usePython:
      return ''
    return '  Using Python\n'

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-python', nargs.ArgBool(None, 0, 'Download and install the Python wrappers'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.sourceControl = framework.require('config.sourceControl',self)
    self.petscdir = framework.require('PETSc.utilities.petscdir', self)
    self.setCompilers = framework.require('config.setCompilers', self)
    self.MPI = framework.require('PETSc.packages.MPI', self)
    if self.framework.argDB['with-python']:
      self.python = framework.require('config.python', None)
    return

  def retrievePackage(self, package, name, urls, packageDir):
    import os
    if not isinstance(urls, list):
      urls = [urls]
    for url in urls:
      import urllib
      tarname = name+'.tar'
      tarnamegz = tarname+'.gz'
      self.framework.log.write('Downloading '+url+' to '+os.path.join(packageDir, package)+'\n')
      try:
        urllib.urlretrieve(url, os.path.join(packageDir, tarnamegz))
      except Exception, e:
        failedmessage = '''Unable to download %s
        You may be off the network. Connect to the internet and run config/configure.py again
        from %s
        ''' % (package, url)
        raise RuntimeError(failedmessage)
      try:
        config.base.Configure.executeShellCommand('cd '+packageDir+'; gunzip '+tarnamegz, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error unzipping '+tarname+': '+str(e))
      try:
        config.base.Configure.executeShellCommand('cd '+packageDir+'; tar -xf '+tarname, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error doing tar -xf '+tarname+': '+str(e))
      os.unlink(os.path.join(packageDir, tarname))
      Dir = None
      for d in os.listdir(packageDir):
        if d.startswith(name) and os.path.isdir(os.path.join(packageDir, d)):
          Dir = d
          break
      self.framework.actions.addArgument(package.upper(), 'Download', 'Downloaded '+package+' into '+str(Dir))
    return

  def getDownloadDir(self, name, packageDir):
    Dir = None
    for d in os.listdir(packageDir):
      if d.startswith(name) and os.path.isdir(os.path.join(packageDir, d)):
        Dir = d
        break
    return Dir

  def retrievePackage(self, package, name, urls, packageDir):
    import install.retrieval

    retriever = install.retrieval.Retriever(self.sourceControl)
    retriever.setup()
    failureMessage = []
    self.framework.log.write('Downloading '+name+'\n')
    for url in urls:
      try:
        retriever.genericRetrieve(url, packageDir, name)
        self.framework.actions.addArgument(package.upper(), 'Download', 'Downloaded '+name+' into '+str(self.getDownloadDir(name, packageDir)))
        return
      except RuntimeError, e:
        failureMessage.append('  Failed to download '+url+'\n'+str(e))
    failureMessage = 'Unable to download '+package+' from locations '+str(urls)+'\n'+'\n'.join(failureMessage)
    raise RuntimeError(failureMessage)

  def configurePythonLanguage(self):
    '''Download the Python bindings into src/python'''
    import os
    if not self.framework.argDB['with-python']:
      return
    if not self.setCompilers.sharedLibraries:
      raise RuntimeError('Python bindings require shared librarary support. This test failed for the specified compilers')
    if not self.setCompilers.dynamicLibraries:
      raise RuntimeError('Python bindings require dynamic library support. This test failed for the specified compilers')
    if not self.MPI.shared and not config.setCompilers.Configure.isDarwin():
      raise RuntimeError('Python bindings require shared MPI library support')
    self.usePython = 1
    if self.petscdir.isClone:
      self.framework.logPrint('PETSc Clone, downloading Python bindings')
      if not os.path.isdir(os.path.join(self.petscdir.dir, 'src', 'python')):
        os.mkdir(os.path.join(self.petscdir.dir, 'src', 'python'))
      if os.path.isdir(os.path.join(self.petscdir.dir, 'src', 'python', 'PETSc')):
        self.logPrint('Python binding source already present')
        return
      try:
        self.retrievePackage('Python Bindings', 'PETScPython', ['ftp://ftp.mcs.anl.gov/pub/petsc/PETScPython.tar.gz'], os.path.join(self.petscdir.dir, 'src', 'python'))
      except:
        self.logPrintBox('Warning: Unable to get the PETSc Python bindings; perhaps you are off the network.\nBuilding without Python bindings')
    else:
      self.framework.logPrint('Not a clone of PETSc, do not need Python bindings')
    return

  def configure(self):
    self.executeTest(self.configurePythonLanguage)
    return
