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

  def check(self, libName, funcName, libDir = '', otherLibs = '', prototype = '', call = '', fortranMangle = 0):
    '''Checks that the library "libName" contains "funcName", and if it does adds "libName" to $LIBS and defines HAVE_LIB"libName"'''
    if fortranMangle:
      funcName = self.compilers.mangleFortranFunction(funcName)
    includes = '/* Override any gcc2 internal prototype to avoid an error. */\n'
    if self.language[-1] == 'C++':
      includes += '''
      #ifdef __cplusplus
      extern "C"
      #endif'''
    if prototype:
      includes += prototype
    else:
      includes += '/* We use char because int might match the return type of a gcc2 builtin and then its argument prototype would still apply. */\n'
      includes += 'char '+funcName+'();\n'
    if call:
      body = call
    else:
      body = funcName+'()\n'
    oldLibs = self.framework.argDB['LIBS']
    if libDir: self.framework.argDB['LIBS'] += ' -L'+libDir
    self.framework.argDB['LIBS'] += ' -l'+libName+' '+otherLibs
    self.pushLanguage(self.language[-1])
    if self.checkLink(includes, body):
      found = 1
      self.framework.argDB['LIBS'] = oldLibs+' -l'+libName
      self.addDefine(self.getDefineName(libName), 1)
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
