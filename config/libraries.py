import config.base

import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework, libraries = []):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.libraries    = libraries
    self.compilers    = self.framework.require('config.compilers', self)
    return

  def getDefineName(self, libName):
    return 'HAVE_LIB'+libName.upper()

  def haveLib(self, libName):
    return self.defines.has_key(self.getDefineName(libName))

  def check(self, libName, funcName, libDir = None, otherLibs = '', prototype = '', call = '', fortranMangle = 0):
    '''Checks that the library "libName" contains "funcName", and if it does adds "libName" to $LIBS and defines HAVE_LIB"libName"
       - libDir may be a list of directories
       - libName may be a list of library names'''
    # Handle Fortran mangling
    if fortranMangle:
      funcName = self.compilers.mangleFortranFunction(funcName)
    includes = '/* Override any gcc2 internal prototype to avoid an error. */\n'
    # Handle C++ mangling
    if self.language[-1] == 'C++':
      includes += '''
      #ifdef __cplusplus
      extern "C"
      #endif'''
    # Construct prototype
    if prototype:
      includes += prototype
    else:
      includes += '/* We use char because int might match the return type of a gcc2 builtin and then its argument prototype would still apply. */\n'
      includes += 'char '+funcName+'();\n'
    # Construct function call
    if call:
      body = call
    else:
      body = funcName+'()\n'
    # Setup link line
    oldLibs = self.framework.argDB['LIBS']
    if libDir:
      if not isinstance(libDir, list): libDir = [libDir]
      for dir in libDir:
        self.framework.argDB['LIBS'] += ' -L'+dir
    if not isinstance(libName, list): libName = [libName]
    for lib in libName:
      self.framework.argDB['LIBS'] += ' -l'+lib
    self.framework.argDB['LIBS'] += ' '+otherLibs
    self.pushLanguage(self.language[-1])
    if self.checkLink(includes, body):
      found = 1
      for lib in libName:
        self.framework.argDB['LIBS'] = oldLibs+' -l'+lib
        self.addDefine(self.getDefineName(lib), 1)
    else:
      found = 0
      self.framework.argDB['LIBS'] = oldLibs
    self.popLanguage()
    return found

  def checkMath(self):
    self.check('m', 'sin', prototype = 'double sin(double);', call = 'sin(1.0);\n')
    return

  def configure(self):
    self.framework.argDB['LIBS'] = ''
    map(lambda args: self.executeTest(self.check, list(args)), self.libraries)
    self.executeTest(self.checkMath)
    self.addArgumentSubstitution('LDFLAGS', 'LDFLAGS')
    self.addArgumentSubstitution('LIBS',    'LIBS')
    return
