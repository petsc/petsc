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

  def getLibArgument(self, library):
    '''Return the proper link line argument for the given library
       - If the path ends in ".lib" return it unchanged
       - If the path is empty, return it unchanged
       - If the path is absolute and the filename is "lib"<name>, return -L<dir> -l<name>
       - If the filename is "lib"<name>, return -l<name>
       - If the path is absolute, return it unchanged
       - Otherwise return -l<filename>'''
    if len(library) > 3 and library[-4:] == '.lib':
      return library
    if not library:
      return library
    if os.path.basename(library).startswith('lib'):
      name = self.getLibName(library)
      if os.path.isabs(library):
        return '-L'+os.path.dirname(library)+' -l'+name
      else:
        return '-l'+name
    if os.path.isabs(library):
      return library
    return '-l'+library

  def getLibName(self, library):
    if os.path.basename(library).startswith('lib'):
      return os.path.splitext(os.path.basename(library))[0][3:]
    return library

  def getDefineName(self, library):
    return 'HAVE_LIB'+self.getLibName(library).upper()

  def haveLib(self, library):
    return self.getDefineName(library) in self.defines

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
        strippedlib = os.path.splitext(os.path.basename(lib))[0]
        self.addDefine(self.getDefineName(strippedlib), 1)
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
