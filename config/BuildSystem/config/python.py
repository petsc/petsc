import config.base

import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.include      = None
    self.lib          = None
    self.isShared     = False
    return

  def __str__(self):
    return ''

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    self.headers   = framework.require('config.headers', self)
    self.libraries = framework.require('config.libraries', self)
    return

  def checkInclude(self, includeDir):
    '''Check that Python.h is present'''
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+' '.join([self.headers.getIncludeArgument(inc) for inc in includeDir])
    found = self.checkPreprocess('#include <Python.h>\n')
    self.compilers.CPPFLAGS = oldFlags
    return found

  def checkPythonLink(self, includes, body, cleanup = 1, codeBegin = None, codeEnd = None, shared = 0):
    '''Analogous to checkLink(), but the Python includes and libraries are automatically provided'''
    success  = 0
    oldFlags = self.compilers.CPPFLAGS
    oldLibs  = self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+' '.join([self.headers.getIncludeArgument(inc) for inc in self.include])
    self.compilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.lib])+' '+self.compilers.LIBS
    if self.checkLink(includes, body, cleanup, codeBegin, codeEnd, shared):
      success = 1
    self.compilers.CPPFLAGS = oldFlags
    self.compilers.LIBS = oldLibs
    return success

  def configurePythonLibraries(self):
    import sysconfig

    # Check for Python headers
    inc = [sysconfig.get_path('include'), sysconfig.get_path('platinclude')]
    if not self.checkInclude(inc):
      raise RuntimeError('Unable to locate Python headers')
    self.include = inc
    # Check for Python dynamic library
    dylib = sysconfig.get_config_var('LDLIBRARY')
    if not dylib:
      raise RuntimeError('LDLIBRARY variable is missing from sysconfig database');
    libDirs = [sysconfig.get_config_var('LIBDIR'),
               sysconfig.get_config_var('LIBPL'),
               os.path.join('/System', 'Library', 'Frameworks'),
               os.path.join('/Library', 'Frameworks')]
    lib = None
    for libDir in libDirs:
      if not libDir: continue
      lib = os.path.join(libDir, dylib)
      if os.path.isfile(lib):
        break
    if lib is None:
      raise RuntimeError("Cannot locate Python dynamic libraries");
    # Remove any version numbers from the library name
    if sysconfig.get_config_var('SO'):
      ext = sysconfig.get_config_var('SO')
      if os.path.isfile(lib.split(ext)[0]+ext):
        lib = lib.split(ext)[0]+ext
    # Add any additional libraries needed for the link
    self.lib = [lib]
    if sysconfig.get_config_var('LIBS'):
      flags = sysconfig.get_config_var('LIBS')
      if sysconfig.get_config_var('LDFLAGS'):
        flags = sysconfig.get_config_var('LDFLAGS')+' '+flags
      self.lib.extend(self.splitLibs(flags))
    if sysconfig.get_config_var('SYSLIBS'):
      self.lib.extend(self.splitLibs(sysconfig.get_config_var('SYSLIBS')))
    # Verify that the Python library is a shared library
    try:
      self.isShared = self.libraries.checkShared('#include <Python.h>\n', 'Py_Initialize', 'Py_IsInitialized', 'Py_Finalize', checkLink = self.checkPythonLink, libraries = self.lib, initArgs = '', noCheckArg = 1)
    except RuntimeError as e:
      raise RuntimeError('Python shared library check failed, probably due to inability to link Python libraries or a bad interaction with the shared linker.\nSuggest running with --with-python=0 if you do not need Python. Otherwise send configure.log to petsc-maint@mcs.anl.gov')
    return

  def setOutput(self):
    '''Add defines and substitutions
       - PYTHON_INCLUDE and PYTHON_LIB are command line arguments for the compile and link
       - PYTHON_INCLUDE_DIR is the directory containing mpi.h
       - PYTHON_LIBRARY is the list of Python libraries'''
    if self.include:
      self.addMakeMacro('PYTHON_INCLUDE',     ' '.join(['-I'+inc for inc in self.include]))
      self.addSubstitution('PYTHON_INCLUDE',     ' '.join(['-I'+inc for inc in self.include]))
      self.addSubstitution('PYTHON_INCLUDE_DIR', self.include[0])
    else:
      self.addSubstitution('PYTHON_INCLUDE',     '')
      self.addSubstitution('PYTHON_INCLUDE_DIR', '')
    if self.lib:
      self.addMakeMacro('PYTHON_LIB',     ' '.join(map(self.libraries.getLibArgument, self.lib)))
      self.addSubstitution('PYTHON_LIB',     ' '.join(map(self.libraries.getLibArgument, self.lib)))
      self.addSubstitution('PYTHON_LIBRARY', self.lib)
    else:
      self.addSubstitution('PYTHON_LIB',     '')
      self.addSubstitution('PYTHON_LIBRARY', '')
    return

  def configure(self):
    self.executeTest(self.configurePythonLibraries)
    self.setOutput()
    return
