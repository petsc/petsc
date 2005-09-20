#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def __str__(self):
    if not hasattr(self, 'scalartype') or not hasattr(self, 'clanguage'):
      return ''
    return '  Scalar type:' + self.scalartype + '\n  Clanguage: ' + self.clanguage +'\n'
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-clanguage=<C or C++>', nargs.Arg(None, 'C', 'Specify C or C++ language'))
    help.addArgument('PETSc', '-with-c++-support', nargs.Arg(None, 0, 'When building C, compile C++ portions of external libraries (e.g. C++)'))
    help.addArgument('PETSc', '-with-fortran', nargs.ArgBool(None, 1, 'Create and install the Fortran wrappers'))
    help.addArgument('PETSc', '-with-python', nargs.ArgBool(None, 0, 'Download and install the Python wrappers'))
    help.addArgument('PETSc', '-with-precision=<single,double,matsingle>', nargs.Arg(None, 'double', 'Specify numerical precision'))    
    help.addArgument('PETSc', '-with-scalar-type=<real or complex>', nargs.Arg(None, 'real', 'Specify real or complex numbers'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.petscdir = framework.require('PETSc.utilities.petscdir', self)
    if self.framework.argDB['with-python']:
      self.python = framework.require('config.python', None)
    return

  def configureScalarType(self):
    '''Choose between real and complex numbers'''
    self.scalartype = self.framework.argDB['with-scalar-type'].lower()
    if self.scalartype == 'complex':
      self.addDefine('USE_COMPLEX', '1')
    elif not self.scalartype == 'real':
      raise RuntimeError('--with-scalar-type must be real or complex')
    self.framework.logPrint('Scalar type is '+str(self.scalartype))
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
    self.framework.logPrint('Precision is '+str(self.precision))
    return

  def configureCLanguage(self):
    '''Choose between C and C++ bindings'''
    self.clanguage = self.framework.argDB['with-clanguage'].upper().replace('+','x').replace('X','x')
    if not self.clanguage in ['C', 'Cxx']:
      raise RuntimeError('Invalid C language specified: '+str(self.clanguage))
    if self.scalartype == 'complex':
      self.clanguage = 'Cxx'
    if self.clanguage == 'C' and not self.framework.argDB['with-c++-support'] and not self.framework.argDB['download-prometheus']:
      self.framework.argDB['with-cxx'] = '0'
    self.framework.logPrint('C language is '+str(self.clanguage))
    self.addDefine('CLANGUAGE_'+self.clanguage.upper(),'1')
    return

  def configureFortranLanguage(self):
    '''Turn on Fortran bindings'''
    if not self.framework.argDB['with-fortran']:
      self.framework.argDB['with-fc'] = '0'
      self.framework.logPrint('Not using Fortran')
    else:
      self.framework.logPrint('Using Fortran')
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
    self.usePython = 0
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

  def configureExternC(self):
    '''Protect C bindings from C++ mangling'''
    if self.clanguage == 'C':
      self.addDefine('USE_EXTERN_CXX',' ')
    return

  def configure(self):
    self.executeTest(self.configureScalarType)
    self.executeTest(self.configurePrecision)
    self.executeTest(self.configureCLanguage)
    self.executeTest(self.configureFortranLanguage)
    self.executeTest(self.configurePythonLanguage)
    self.executeTest(self.configureExternC)
    return
