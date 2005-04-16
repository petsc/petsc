#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.arch         = self.framework.require('PETSc.utilities.arch',self)
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-clanguage=<C or C++>', nargs.Arg(None, 'C', 'Specify C or C++ language'))
    help.addArgument('PETSc', '-with-python', nargs.ArgBool(None, 1, 'Download and install the Python wrappers'))
    help.addArgument('PETSc', '-with-precision=<single,double,matsingle>', nargs.Arg(None, 'double', 'Specify numerical precision'))    
    help.addArgument('PETSc', '-with-scalar-type=<real or complex>', nargs.Arg(None, 'real', 'Specify real or complex numbers'))
    return

  def configureScalarType(self):
    '''Choose between real and complex numbers'''
    self.scalartype = self.framework.argDB['with-scalar-type'].lower()
    if self.scalartype == 'complex':
      self.framework.argDB['with-clanguage'] = 'Cxx'
      self.addDefine('USE_COMPLEX', '1')
    elif not self.scalartype == 'real':
      raise RuntimeError('--with-scalar-type must be real or complex')
    return

  def configurePrecision(self):
    '''Set the default real number precision for PETSc objects'''
    self.precision = self.framework.argDB['with-precision'].lower()
    if self.precision == 'single':
      self.addDefine('USE_SINGLE', '1')
    elif self.precision == 'matsingle':
      self.addDefine('USE_MAT_SINGLE', '1')
    elif not self.precision == 'double':
      raise RuntimeError('--with-precision must be single, double, or matsingle')
    return

  def configureCLanguage(self):
    '''Choose between C and C++ bindings'''
    self.clanguage = self.framework.argDB['with-clanguage'].upper().replace('+','x').replace('X','x')
    if not self.clanguage in ['C', 'Cxx']:
      raise RuntimeError('Invalid C language specified: '+str(self.clanguage))
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
    '''Download the Python bindings'''
    import os
    if os.path.isdir(os.path.join(self.arch.dir,'BitKeeper')):
      try:
        self.retrievePackage('Python Bindings', 'PETScPython', 'ftp://ftp.mcs.anl.gov/pub/petsc/PETScPython.tar.gz', os.path.join(self.arch.dir, 'lib',self.arch.arch))
      except:
        self.logPrintBox('Warning: Unable to get the PETSc Python bindings; perhaps you are off the network.\nBuilding without Python bindings')
    return

  def configureExternC(self):
    '''Protect C bindings from C++ mangling'''
    if self.clanguage == 'C':
      self.addDefine('USE_EXTERN_CXX',' ')
    return

  def configure(self):
    self.executeTest(self.configureScalarType)
    self.executeTest(self.configurePrecision)
    self.executeTest(self.configureCLanguage)
    self.executeTest(self.configurePythonLanguage)
    self.executeTest(self.configureExternC)
    return
