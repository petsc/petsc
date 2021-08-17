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
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers', self)
    self.compilers    = framework.require('config.compilers',    self)
    self.headers      = framework.require('config.headers',      self)
    return

  def getLibArgumentList(self, library, with_rpath=True):
    '''Return the proper link line argument for the given filename library as a list of options
      - If the path is empty, return it unchanged
      - If starts with - then return unchanged
      - If the path ends in ".lib" return it unchanged
      - If the path is absolute and the filename is "lib"<name>, return -L<dir> -l<name> (optionally including rpath flag)
      - If the filename is "lib"<name>, return -l<name>
      - If the path ends in ".so" or ".dylib" return it unchanged
      - If the path ends in ".o" return it unchanged
      - If the path is absolute, return it unchanged
      - Otherwise return -l<library>'''
    if not library:
      return []
    if library.startswith('${CC_LINKER_SLFLAG}'):
      return [library] if with_rpath else []
    if library.startswith('${FC_LINKER_SLFLAG}'):
      return [library] if with_rpath else []
    if library.lstrip()[0] == '-':
      return [library]
    if len(library) > 3 and library[-4:] == '.lib':
      return [library.replace('\\ ',' ').replace(' ', '\\ ').replace('\\(','(').replace('(', '\\(').replace('\\)',')').replace(')', '\\)')]
    if os.path.basename(library).startswith('lib'):
      name = self.getLibName(library)
      if ((len(library) > 2 and library[1] == ':') or os.path.isabs(library)):
        flagName  = self.language[-1]+'SharedLinkerFlag'
        flagSubst = self.language[-1].upper()+'_LINKER_SLFLAG'
        dirname   = os.path.dirname(library).replace('\\ ',' ').replace(' ', '\\ ').replace('\\(','(').replace('(', '\\(').replace('\\)',')').replace(')', '\\)')
        if dirname in ['/usr/lib','/lib','/usr/lib64','/lib64']:
          return [library]
        if with_rpath:
          if hasattr(self.setCompilers, flagName) and not getattr(self.setCompilers, flagName) is None:
            return [getattr(self.setCompilers, flagName)+dirname,'-L'+dirname,'-l'+name]
          if flagSubst in self.argDB:
            return [self.argDB[flagSubst]+dirname,'-L'+dirname,'-l'+name]
        return ['-L'+dirname,'-l'+name]
      else:
        return ['-l'+name]
    if os.path.splitext(library)[1] == '.so' or os.path.splitext(library)[1] == '.o' or os.path.splitext(library)[1] == '.dylib':
      return [library]
    if os.path.isabs(library):
      return [library]
    return ['-l'+library]

  def getLibArgument(self, library):
    '''Same as getLibArgumentList - except it returns a string instead of list.'''
    return  ' '.join(self.getLibArgumentList(library))

  def getLibName(library):
    if os.path.basename(library).startswith('lib'):
      return os.path.splitext(os.path.basename(library))[0][3:]
    return library
  getLibName = staticmethod(getLibName)

  def getDefineName(self, library):
    return 'HAVE_LIB'+self.getLibName(library).upper().replace('-','_').replace('=','_').replace('+','_').replace('.', '_').replace('/','_')

  def getDefineNameFunc(self, funcName):
    return 'HAVE_'+ funcName.upper()

  def haveLib(self, library):
    return self.getDefineName(library) in self.defines

  def add(self, libName, funcs, libDir = None, otherLibs = [], prototype = '', call = '', fortranMangle = 0):
    '''Checks that the library "libName" contains "funcs", and if it does defines HAVE_LIB"libName AND adds it to $LIBS"
       - libDir may be a list of directories
       - libName may be a list of library names'''
    if not isinstance(libName, list): libName = [libName]
    if self.check(libName, funcs, libDir, otherLibs, prototype, call, fortranMangle):
      self.logPrint('Adding '+str(libName)+' to LIBS')
      # Note: this MUST be setCompilers since it can happen before dispatch names is made
      self.setCompilers.LIBS = self.toString(libName)+' '+self.setCompilers.LIBS
      return 1
    return 0

  def toString(self,libs):
    '''Converts a list of libraries to a string suitable for a linker'''
    newlibs = []
    frame = 0
    for lib in libs:
      if frame:
        newlibs += [lib]
        frame   = 0
      elif lib == '-framework':
        newlibs += [lib]
        frame = 1
      else:
        newlibs += self.getLibArgumentList(lib)
    return ' '.join(newlibs)

  def toStringNoDupes(self,libs,with_rpath=True):
    '''Converts a list of libraries to a string suitable for a linker, removes duplicates'''
    '''Moves the flags that can be moved to the beginning of the string but always leaves the libraries and other items that must remain in the same order'''
    newlibs = []
    frame = 0
    for lib in libs:
      if frame:
        newlibs += [lib]
        frame   = 0
      elif lib == '-framework':
        newlibs += [lib]
        frame = 1
      else:
        newlibs += self.getLibArgumentList(lib, with_rpath)
    libs = newlibs
    newldflags = []
    newlibs = []
    frame = 0
    dupflags = ['-L']
    flagName  = self.language[-1]+'SharedLinkerFlag'
    if hasattr(self.setCompilers, flagName) and not getattr(self.setCompilers, flagName) is None:
      dupflags.append(getattr(self.setCompilers, flagName))
    for j in libs:
      # remove duplicate -L, -Wl,-rpath options - and only consecutive -l options
      if j in newldflags and any([j.startswith(flg) for flg in dupflags]): continue
      if newlibs and j == newlibs[-1]: continue
      if j.startswith('-l') or j.endswith('.lib') or j.endswith('.a') or j.endswith('.o') or j == '-Wl,-Bstatic' or j == '-Wl,-Bdynamic' or j == '-Wl,--start-group' or j == '-Wl,--end-group':
        newlibs.append(j)
      else:
        newldflags.append(j)
    liblist = ' '.join(newldflags + newlibs)
    return liblist

  def getShortLibName(self,lib):
    '''returns the short name for the library. Valid names are foo -lfoo or libfoo.[a,so,lib]'''
    if lib.startswith('-l'):
      libname = lib[2:]
      return libname
    if lib.startswith('-'): # must be some compiler options - not a library
      return ''
    if lib.endswith('.a') or lib.endswith('.so') or lib.endswith('.lib'):
      libname = os.path.splitext(os.path.basename(lib))[0]
      if lib.startswith('lib'): libname = libname[3:]
      return libname
    # no match - assuming the given name is already in short notation
    return lib

  def check(self, libName, funcs, libDir = None, otherLibs = [], prototype = '', call = '', fortranMangle = 0, cxxMangle = 0, cxxLink = 0, functionDefine = 0, examineOutput=lambda ret,out,err:None):
    '''Checks that the library "libName" contains "funcs", and if it does defines HAVE_LIB"libName"
       - libDir may be a list of directories
       - libName may be a list of library names'''
    if not isinstance(funcs,list): funcs = [funcs]
    if not isinstance(libName, list): libName = [libName]
    def genPreamble(f, funcName):
      # Construct prototype
      if self.language[-1] == 'FC':
        return ''
      if prototype:
        if isinstance(prototype, str):
          pre = prototype
        else:
          pre = prototype[f]
      else:
        # We use char because int might match the return type of a gcc2 builtin and its argument prototype would still apply.
        pre = 'char '+funcName+'();'
      # Capture the function call in a static function so that any local variables are isolated from
      # calls to other library functions.
      return pre + '\nstatic void _check_%s() { %s }' % (funcName, genCall(f, funcName, pre=True))
    def genCall(f, funcName, pre=False):
      if self.language[-1] != 'FC' and not pre:
        return '_check_' + funcName + '();'
      # Construct function call
      if call:
        if isinstance(call, str):
          body = call
        else:
          body = call[f]
      else:
        body = funcName+'()'
      if self.language[-1] != 'FC':
        body += ';'
      return body
    # Handle Fortran mangling
    if fortranMangle:
      funcs = list(map(self.compilers.mangleFortranFunction, funcs))
    if not funcs:
      self.logPrint('No functions to check for in library '+str(libName)+' '+str(otherLibs))
      return True
    self.logPrint('Checking for functions ['+' '.join(funcs)+'] in library '+str(libName)+' '+str(otherLibs))
    if self.language[-1] == 'FC':
      includes = ''
    else:
      includes = '/* Override any gcc2 internal prototype to avoid an error. */\n'
    # Handle C++ mangling
    if self.language[-1] == 'Cxx' and not cxxMangle:
      includes += '''
#ifdef __cplusplus
extern "C" {
#endif
'''
    includes += '\n'.join([genPreamble(f, fname) for f, fname in enumerate(funcs)])
    # Handle C++ mangling
    if self.language[-1] == 'Cxx' and not cxxMangle:
      includes += '''
#ifdef __cplusplus
}
#endif
'''
    body = '\n'.join([genCall(f, fname) for f, fname in enumerate(funcs)])
    # Setup link line
    oldLibs = self.setCompilers.LIBS
    if libDir:
      if not isinstance(libDir, list): libDir = [libDir]
      for dir in libDir:
        self.setCompilers.LIBS += ' -L'+dir
    # new libs may/will depend on system libs so list new libs first!
    # Matt, do not change this without talking to me
    if libName and otherLibs:
      self.setCompilers.LIBS = ' '+self.toString(libName+otherLibs) +' '+ self.setCompilers.LIBS
    elif otherLibs:
      self.setCompilers.LIBS = ' '+self.toString(otherLibs) +' '+ self.setCompilers.LIBS
    elif libName:
      self.setCompilers.LIBS = ' '+self.toString(libName) +' '+ self.setCompilers.LIBS
    if cxxMangle: compileLang = 'Cxx'
    else:         compileLang = self.language[-1]
    if cxxLink: linklang = 'Cxx'
    else: linklang = self.language[-1]
    self.pushLanguage(compileLang)

    found = 1
    if libName and libName[0].startswith('/'):
      dir = os.path.dirname(libName[0])
      lib = os.path.basename(libName[0])[:-1]
      self.logPrint('Checking directory of requested libraries:'+dir+' for first library:'+lib)
      found = 0
      try:
        files = os.listdir(dir)
      except:
        self.logPrint('Directory of requested libraries '+dir+' does not exist')
      else:
        self.logPrint('Files in directory:'+str(files))
        for i in files:
          if i.startswith(lib):
            found = 1
            break

    if found and self.checkLink(includes, body, linkLanguage=linklang, examineOutput=examineOutput):
      # define the symbol as found
      if functionDefine: [self.addDefine(self.getDefineNameFunc(fname), 1) for f, fname in enumerate(funcs)]
      # add to list of found libraries
      elif libName:
        for lib in libName:
          shortlib = self.getShortLibName(lib)
          if shortlib: self.addDefine(self.getDefineName(shortlib), 1)
    else:
      found = 0
    self.setCompilers.LIBS = oldLibs
    self.popLanguage()
    return found

  def checkClassify(self, libName, funcs, libDir=None, otherLibs=[], prototype='', call='', fortranMangle=0, cxxMangle=0, cxxLink=0):
    '''Recursive decompose to rapidly classify functions as found or missing'''
    import config
    def functional(funcs):
      named = config.NamedInStderr(funcs)
      if self.check(libName, funcs, libDir, otherLibs, prototype, call, fortranMangle, cxxMangle, cxxLink):
        return True
      else:
        return named.named
    found, missing = config.classify(funcs, functional)
    return found, missing

  def checkMath(self):
    '''Check for sin() in libm, the math library'''
    self.math = None
    funcs = ['sin', 'floor', 'log10', 'pow']
    prototypes = ['#include <stdio.h>\ndouble sin(double);',
                  '#include <stdio.h>\ndouble floor(double);',
                  '#include <stdio.h>\ndouble log10(double);',
                  '#include <stdio.h>\ndouble pow(double, double);']
    calls = ['double x,y; scanf("%lf",&x); y = sin(x); printf("%f",y);\n',
             'double x,y; scanf("%lf",&x); y = floor(x); printf("%f",y);\n',
             'double x,y; scanf("%lf",&x); y = log10(x); printf("%f",y);\n',
             'double x,y; scanf("%lf",&x); y = pow(x,x); printf("%f",y);\n']
    if self.check('', funcs, prototype = prototypes, call = calls):
      self.math = []
    elif self.check('m', funcs, prototype = prototypes, call = calls):
      self.math = ['libm.a']
    self.logPrint('CheckMath: using math library '+str(self.math))
    return

  def checkMathErf(self):
    '''Check for erf() in libm, the math library'''
    if not self.math is None and self.check(self.math, ['erf'], prototype = ['#include <math.h>'], call = ['double (*checkErf)(double) = erf;double x = 0,y; y = (*checkErf)(x)']):
      self.logPrint('erf() found')
      self.addDefine('HAVE_ERF', 1)
    else:
      self.logPrint('Warning: erf() not found')
    return

  def checkMathTgamma(self):
    '''Check for tgamma() in libm, the math library'''
    if not self.math is None and self.check(self.math, ['tgamma'], prototype = ['#include <math.h>'], call = ['double (*checkTgamma)(double) = tgamma;double x = 0,y; y = (*checkTgamma)(x)']):
      self.logPrint('tgamma() found')
      self.addDefine('HAVE_TGAMMA', 1)
    else:
      self.logPrint('Warning: tgamma() not found')
    return

  def checkMathLgamma(self):
    '''Check for lgamma() in libm, the math library'''
    if not self.math is None and self.check(self.math, ['lgamma'], prototype = ['#include <math.h>\n#include <stdlib.h>'], call = ['double (*checkLgamma)(double) = lgamma;double x = 1,y; y = (*checkLgamma)(x);if (y != 0.) abort()']):
      self.logPrint('lgamma() found')
      self.addDefine('HAVE_LGAMMA', 1)
    elif not self.math is None and self.check(self.math, ['gamma'], prototype = ['#include <math.h>\n#include <stdlib.h>'], call = ['double (*checkLgamma)(double) = gamma;double x = 1,y; y = (*checkLgamma)(x);if (y != 0.) abort()']):
      self.logPrint('gamma() found')
      self.addDefine('HAVE_LGAMMA', 1)
      self.addDefine('HAVE_LGAMMA_IS_GAMMA', 1)
    else:
      self.logPrint('Warning: lgamma() and gamma() not found')
    return

  def checkMathFenv(self):
    '''Checks if <fenv.h> can be used with FE_DFL_ENV'''
    if not self.math is None and self.check(self.math, ['fesetenv'], prototype = ['#include <fenv.h>'], call = ['fesetenv(FE_DFL_ENV);']):
      self.addDefine('HAVE_FENV_H', 1)
    else:
      self.logPrint('Warning: <fenv.h> with FE_DFL_ENV not found')
    return

  def checkMathLog2(self):
    '''Check for log2() in libm, the math library'''
    if not self.math is None and self.check(self.math, ['log2'], prototype = ['#include <math.h>'], call = ['double (*checkLog2)(double) = log2; double x = 2.5, y = (*checkLog2)(x)']):
      self.logPrint('log2() found')
      self.addDefine('HAVE_LOG2', 1)
    else:
      self.logPrint('Warning: log2() not found')
    return

  def checkRealtime(self):
    '''Check for presence of clock_gettime() in realtime library (POSIX Realtime extensions)'''
    self.rt = None
    funcs = ['clock_gettime']
    prototypes = ['#include <time.h>']
    calls = ['struct timespec tp; clock_gettime(CLOCK_REALTIME,&tp);']
    if self.check('', funcs, prototype=prototypes, call=calls):
      self.logPrint('realtime functions are linked in by default')
      self.rt = []
    elif self.check('rt', funcs, prototype=prototypes, call=calls):
      self.logPrint('Using librt for the realtime library')
      self.rt = ['librt.a']
    else:
      self.logPrint('Warning: No realtime library found')
    return

  def checkDynamic(self):
    '''Check for the header and libraries necessary for dynamic library manipulation'''
    if 'with-dynamic-loading' in self.argDB and not self.argDB['with-dynamic-loading']: return
    self.check(['dl'], 'dlopen')
    self.headers.check('dlfcn.h')
    return

  def checkShared(self, includes, initFunction, checkFunction, finiFunction = None, checkLink = None, libraries = [], initArgs = '&argc, &argv', boolType = 'int', noCheckArg = 0, defaultArg = '', executor = None, timeout = 60):
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
      if hasattr(checkLink, '__self__'):
        configObj = checkLink.__self__
      else:
        configObj = self

    # Fix these flags
    oldFlags = self.setCompilers.LIBS
    self.setCompilers.LIBS = ' '+self.toString(libraries)+' '+self.setCompilers.LIBS

    # Make a library which calls initFunction(), and returns checkFunction()
    lib1Name = os.path.join(self.tmpDir, 'lib1.'+self.setCompilers.sharedLibraryExt)
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
      self.setCompilers.LIBS = oldFlags
      raise RuntimeError('Could not complete shared library check')
    if os.path.isfile(configObj.compilerObj): os.remove(configObj.compilerObj)
    os.rename(configObj.linkerObj, lib1Name)

    # Make a library which calls checkFunction()
    lib2Name = os.path.join(self.tmpDir, 'lib2.'+self.setCompilers.sharedLibraryExt)
    codeBegin = '''
#ifdef __cplusplus
extern "C"
#endif
int checkInit(void) {
'''
    body      = '''
  %s isInitialized;

  %s
''' % (boolType, checkCode)
    if finiFunction:
      body += '  if (isInitialized) '+finiFunction+'();\n'
    body += '  return (int) isInitialized;\n'
    codeEnd   = '\n}\n'
    if not checkLink(includes, body, cleanup = 0, codeBegin = codeBegin, codeEnd = codeEnd, shared = 1):
      if os.path.isfile(configObj.compilerObj): os.remove(configObj.compilerObj)
      self.setCompilers.LIBS = oldFlags
      raise RuntimeError('Could not complete shared library check')
      return 0
    if os.path.isfile(configObj.compilerObj): os.remove(configObj.compilerObj)
    os.rename(configObj.linkerObj, lib2Name)

    self.setCompilers.LIBS = oldFlags

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
  char *argv[2] = {(char *) "conftest", NULL};
  void *lib;
  int (*init)(int, char **);
  int (*checkInit)(void);

  lib = dlopen("'''+lib1Name+'''", RTLD_LAZY);
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
  lib = dlopen("'''+lib2Name+'''", RTLD_LAZY);
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
    oldLibs = self.setCompilers.LIBS
    if self.haveLib('dl'):
      self.setCompilers.LIBS += ' -ldl'
    isShared = 0
    try:
      isShared = self.checkRun(defaultIncludes, body, defaultArg = defaultArg, executor = executor, timeout = timeout)
    except RuntimeError as e:
      if executor and str(e).find('Runaway process exceeded time limit') > -1:
        raise RuntimeError('Timeout: Unable to run MPI program with '+executor+'\n\
    (1) make sure this is the correct program to run MPI jobs\n\
    (2) your network may be misconfigured; see https://petsc.org/release/faq/#mpi-network-misconfigure\n\
    (3) you may have VPN running whose network settings may not play nice with MPI\n')

    self.setCompilers.LIBS = oldLibs
    if os.path.isfile(lib1Name) and self.framework.doCleanup: os.remove(lib1Name)
    if os.path.isfile(lib2Name) and self.framework.doCleanup: os.remove(lib2Name)
    if isShared:
      self.logPrint('Library was shared')
    else:
      self.logPrint('Library was not shared')
    return isShared

  def isBGL(self):
    '''Returns true if compiler is IBM cross compiler for BGL'''
    if not hasattr(self, '_isBGL'):
      self.logPrint('**********Checking if running on BGL/IBM detected')
      if (self.check('', 'bgl_perfctr_void') or self.check('','ADIOI_BGL_Open')) and self.check('', '_xlqadd'):
        self.logPrint('*********BGL/IBM detected')
        self._isBGL = 1
      else:
        self.logPrint('*********BGL/IBM test failure')
        self._isBGL = 0
    return self._isBGL

  def configure(self):
    list(map(lambda args: self.executeTest(self.check, list(args)), self.libraries))
    self.executeTest(self.checkMath)
    self.executeTest(self.checkMathErf)
    self.executeTest(self.checkMathTgamma)
    self.executeTest(self.checkMathLgamma)
    self.executeTest(self.checkMathFenv)
    self.executeTest(self.checkMathLog2)
    self.executeTest(self.checkRealtime)
    self.executeTest(self.checkDynamic)
    return
