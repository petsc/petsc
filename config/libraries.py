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

  def getLibArgument(self, libName):
    '''Leave full library path arguments unchanged, otherwise return -l<name> form'''
    if len(libName) > 3 and libName[-4:] == '.lib':
      return libName
    return '-l'+libName

  def getDefineName(self, libName):
    return 'HAVE_LIB'+libName.upper()

  def haveLib(self, libName):
    return self.getDefineName(libName) in self.defines

  def check(self, libName, funcName, libDir = None, otherLibs = '', prototype = '', call = '', fortranMangle = 0):
    '''Checks that the library "libName" contains "funcName", and if it does adds "libName" to $LIBS and defines HAVE_LIB"libName"
       - libDir may be a list of directories
       - libName may be a list of library names'''
    self.framework.log.write('Checking for function '+funcName+' in library '+str(libName)+'\n')
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
      self.framework.argDB['LIBS'] += ' '+self.getLibArgument(lib)
    self.framework.argDB['LIBS'] += ' '+otherLibs
    self.pushLanguage(self.language[-1])
    if self.checkLink(includes, body):
      found = 1
      for lib in libName:
        self.framework.argDB['LIBS'] = oldLibs+' '+self.getLibArgument(lib)
        self.addDefine(self.getDefineName(lib), 1)
    else:
      found = 0
      self.framework.argDB['LIBS'] = oldLibs
    self.popLanguage()
    return found

  def checkMath(self):
    '''Check for sin() in libm, the math library'''
    self.check('m', 'sin', prototype = 'double sin(double);', call = 'sin(1.0);\n')
    return

  def configure(self):
    self.framework.argDB['LIBS'] = ''
    map(lambda args: self.executeTest(self.check, list(args)), self.libraries)
    self.executeTest(self.checkMath)
    self.addArgumentSubstitution('LDFLAGS', 'LDFLAGS')
    self.addArgumentSubstitution('LIBS',    'LIBS')
    return
