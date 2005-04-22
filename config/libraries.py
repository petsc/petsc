import config.base

import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework, libraries = []):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.libraries    = libraries
    return

  def setupDependencies(self, framework):
    self.setCompilers = framework.require('config.setCompilers', self)
    self.compilers    = framework.require('config.compilers',    self)
    self.headers      = framework.require('config.headers',      self)
    return

  def getLibArgument(self, library):
    '''Using the method from config.compilers'''
    return self.compilers.getLibArgument(library)

  def getLibName(library):
    if os.path.basename(library).startswith('lib'):
      return os.path.splitext(os.path.basename(library))[0][3:]
    return library
  getLibName = staticmethod(getLibName)

  def getDefineName(self, library):
    return 'HAVE_LIB'+self.getLibName(library).upper().replace('-','_').replace('=','_').replace('+','_')

  def haveLib(self, library):
    return self.getDefineName(library) in self.defines

  def add(self, libName, funcs, libDir = None, otherLibs = [], prototype = '', call = '', fortranMangle = 0):
    '''Checks that the library "libName" contains "funcs", and if it does defines HAVE_LIB"libName AND adds it to $LIBS"
       - libDir may be a list of directories
       - libName may be a list of library names'''
    if not isinstance(libName, list): libName = [libName]
    if self.check(libName, funcs, libDir, otherLibs, prototype, call, fortranMangle):
      self.logPrint('Adding '+str(libName)+' to argDB[LIBS]')
      self.framework.argDB['LIBS'] += ' '+self.toString(libName)
      return 1
    return 0

  def toString(self,libs):
    '''Converts a list of libraries to a string suitable for a linker'''
    return ' '.join([self.getLibArgument(lib) for lib in libs])

  def check(self, libName, funcs, libDir = None, otherLibs = [], prototype = '', call = '', fortranMangle = 0):
    '''Checks that the library "libName" contains "funcs", and if it does defines HAVE_LIB"libName"
       - libDir may be a list of directories
       - libName may be a list of library names'''
    if not isinstance(funcs,list): funcs = [funcs]
    if not isinstance(libName, list): libName = [libName]
    self.framework.logPrint('Checking for functions '+str(funcs)+' in library '+str(libName)+' '+str(otherLibs))
    for funcName in funcs:
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
        # We use char because int might match the return type of a gcc2 builtin and its argument prototype would still apply.
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
      self.framework.argDB['LIBS'] += ' '+self.toString(libName+otherLibs)
      self.pushLanguage(self.language[-1])
      found = 0
      if self.checkLink(includes, body):
        found = 1
        # add to list of found libraries
        for lib in libName:
          if self.haveLib(lib): continue
          if lib.startswith('-L'): continue
          strippedlib = os.path.splitext(os.path.basename(lib))[0]
          if strippedlib: self.addDefine(self.getDefineName(strippedlib), 1)
      self.framework.argDB['LIBS'] = oldLibs
      self.popLanguage()
      if not found: return 0
    return 1

  def checkMath(self):
    '''Check for sin() in libm, the math library'''
    self.math = None
    if self.check('','sin', prototype = 'double sin(double);', call = 'sin(1.0);\n'):
      self.math = []
    elif self.check('m', 'sin', prototype = 'double sin(double);', call = 'sin(1.0);\n'):
      self.math = ['libm.a']
    return

  def checkDynamic(self):
    '''Check for the header and libraries necessary for dynamic library manipulation'''
    self.check(['dl'], 'dlopen')
    self.headers.check('dlfcn.h')
    return

  def checkShared(self, includes, initFunction, checkFunction, finiFunction = None, checkLink = None, libraries = [], initArgs = '&argc, &argv', boolType = 'int', noCheckArg = 0):
    '''Determine whether a library is shared
       - initFunction(int *argc, char *argv[]) is called to initialize some static data
       - checkFunction(int *check) is called to verify that the static data wer set properly
       - finiFunction() is called to finalize the data, and may be omitted
       - checkLink may be given as ana alternative to the one in base.Configure'''
    isShared = 0
    if checkLink is None:
      checkLink = self.checkLink
      configObj = self
    else:
      if hasattr(checkLink, 'im_self'):
        configObj = checkLink.im_self
      else:
        configObj = self

    # Fix these flags
    oldFlags = self.framework.argDB['LIBS']
    self.framework.argDB['LIBS'] += ' '+self.toString(libraries)

    # Make a library which calls initFunction(), and returns checkFunction()
    if noCheckArg:
      checkCode = 'isInitialized = '+checkFunction+'();'
    else:
      checkCode = checkFunction+'(&isInitialized);'
    codeBegin = '''
#ifdef __cplusplus
extern "C"
#endif
int init(int argc,  char *argv[]) {
'''
    body      = '''
  %s isInitialized;

  %s(%s);
  %s
  return (int) isInitialized;
''' % (boolType, initFunction, initArgs, checkCode)
    codeEnd   = '\n}\n'
    if not checkLink(includes, body, cleanup = 0, codeBegin = codeBegin, codeEnd = codeEnd, shared = 1):
      if os.path.isfile(configObj.compilerObj): os.remove(configObj.compilerObj)
      self.framework.argDB['LIBS'] = oldFlags
      raise RuntimeError('Could not complete shared library check')
    if os.path.isfile(configObj.compilerObj): os.remove(configObj.compilerObj)
    os.rename(configObj.linkerObj, 'lib1.so')

    # Make a library which calls checkFunction()
    codeBegin = '''
#ifdef __cplusplus
extern "C"
#endif
int checkInit(void) {
'''
    body      = '''
  %s isInitialized;

  %s
  return (int) isInitialized;
''' % (boolType, checkCode)
    codeEnd   = '\n}\n'
    if not checkLink(includes, body, cleanup = 0, codeBegin = codeBegin, codeEnd = codeEnd, shared = 1):
      if os.path.isfile(configObj.compilerObj): os.remove(configObj.compilerObj)
      self.framework.argDB['LIBS'] = oldFlags
      self.framework.logPrint('Could not complete shared library check')
      return 0
    if os.path.isfile(configObj.compilerObj): os.remove(configObj.compilerObj)
    os.rename(configObj.linkerObj, 'lib2.so')

    self.framework.argDB['LIBS'] = oldFlags

    # Make an executable that dynamically loads and calls both libraries
    #   If the check returns true in the second library, the static data was shared
    guard = self.headers.getDefineName('dlfcn.h')
    if self.headers.headerPrefix:
      guard = self.headers.headerPrefix+'_'+guard
    defaultIncludes = '''
#include <stdio.h>
#include <stdlib.h>
#ifdef %s
  #include <dlfcn.h>
#endif
    ''' % guard
    body = '''
  int   argc    = 1;
  char *argv[1] = {"conftest"};
  void *lib;
  int (*init)(int, char **);
  int (*checkInit)(void);

  lib = dlopen("./lib1.so", RTLD_LAZY);
  if (!lib) {
    fprintf(stderr, "Could not open lib1.so: %s\\n", dlerror());
    exit(1);
  }
  init = (int (*)(int, char **)) dlsym(lib, "init");
  if (!init) {
    fprintf(stderr, "Could not find initialization function\\n");
    exit(1);
  }
  if (!(*init)(argc, argv)) {
    fprintf(stderr, "Could not initialize library\\n");
    exit(1);
  }
  lib = dlopen("./lib2.so", RTLD_LAZY);
  if (!lib) {
    fprintf(stderr, "Could not open lib2.so: %s\\n", dlerror());
    exit(1);
  }
  checkInit = (int (*)(void)) dlsym(lib, "checkInit");
  if (!checkInit) {
    fprintf(stderr, "Could not find initialization check function\\n");
    exit(1);
  }
  if (!(*checkInit)()) {
    fprintf(stderr, "Did not link with shared library\\n");
    exit(2);
  }
    '''
    oldLibs = self.framework.argDB['LIBS']
    if self.haveLib('dl'):
      self.framework.argDB['LIBS'] += ' -ldl'
    if self.checkRun(defaultIncludes, body):
      isShared = 1
    self.framework.argDB['LIBS'] = oldLibs
    if os.path.isfile('lib1.so'): os.remove('lib1.so')
    if os.path.isfile('lib2.so'): os.remove('lib2.so')
    if isShared:
      self.framework.logPrint('Library was shared')
    else:
      self.framework.logPrint('Library was not shared')
    return isShared

  def configure(self):
    self.framework.argDB['LIBS'] = ''
    map(lambda args: self.executeTest(self.check, list(args)), self.libraries)
    self.executeTest(self.checkMath)
    self.executeTest(self.checkDynamic)
    self.addArgumentSubstitution('LDFLAGS', 'LDFLAGS')
    self.addArgumentSubstitution('LIBS',    'LIBS')
    return
