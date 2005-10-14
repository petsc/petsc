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
    self.petscdir = framework.require('PETSc.utilities.petscdir', self)
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

  def configurePythonLanguage(self):
    '''Download the Python bindings into src/python'''
    import os
    if not self.framework.argDB['with-python']:
      return
    if not self.framework.argDB['with-shared'] and not self.framework.argDB['with-dynamic']:
      raise RuntimeError('Python bindings require both shared and dynamic libraries. Please add --with-shared --with-dynamic to your configure options.')
    if not self.framework.argDB['with-shared']:
      raise RuntimeError('Python bindings require shared libraries. Please add --with-shared to your configure options.')
    if not self.framework.argDB['with-dynamic']:
      raise RuntimeError('Python bindings require dynamic libraries. Please add --with-dynamic to your configure options.')
    self.usePython = 1
    if os.path.isdir(os.path.join(self.petscdir.dir, 'BitKeeper')) or os.path.exists(os.path.join(self.petscdir.dir, 'BK')):
      if not os.path.isdir(os.path.join(self.petscdir.dir, 'src', 'python')):
        os.mkdir(os.path.join(self.petscdir.dir, 'src', 'python'))
      if os.path.isdir(os.path.join(self.petscdir.dir, 'src', 'python', 'PETSc')):
        self.logPrint('Python binding source already present')
        return
      try:
        self.retrievePackage('Python Bindings', 'PETScPython', 'ftp://ftp.mcs.anl.gov/pub/petsc/PETScPython.tar.gz', os.path.join(self.petscdir.dir, 'src', 'python'))
      except:
        self.logPrintBox('Warning: Unable to get the PETSc Python bindings; perhaps you are off the network.\nBuilding without Python bindings')
    return

  def configure(self):
    self.executeTest(self.configurePythonLanguage)
    return
