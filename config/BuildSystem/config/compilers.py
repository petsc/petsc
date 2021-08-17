import config.base

import re
import os
import shutil

def remove_xcode_verbose(buf):
  retbuf =[]
  for line in buf.splitlines():
    if not line.startswith('ld: warning: text-based stub file'): retbuf.append(line)
  return ('\n').join(retbuf)

class MissingProcessor(AttributeError):
  pass

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.fortranMangling = 'unchanged'
    self.fincs = []
    self.flibs = []
    self.fmainlibs = []
    self.clibs = []
    self.cxxlibs = []
    self.skipdefaultpaths = []
    self.cxxCompileC = False
    self.cRestrict = ' '
    self.cxxRestrict = ' '
    self.cxxdialect = ''
    self.c99flag = None
    return

  def getSkipDefaultPaths(self):
    if len(self.skipdefaultpaths):
      return self.skipdefaultpaths
    else:
      self.skipdefaultpaths = ['/usr/lib','/lib','/usr/lib64','/lib64','/usr/lib/x86_64-linux-gnu','/lib/x86_64-linux-gnu']
      conda_sysrt = os.getenv('CONDA_BUILD_SYSROOT')
      if conda_sysrt:
        conda_sysrt = os.path.abspath(conda_sysrt)
        self.skipdefaultpaths.extend([conda_sysrt+lib for lib in self.skipdefaultpaths])
      return self.skipdefaultpaths

  def setupHelp(self, help):
    import nargs

    help.addArgument('Compilers', '-with-clib-autodetect=<bool>',           nargs.ArgBool(None, 1, 'Autodetect C compiler libraries'))
    help.addArgument('Compilers', '-with-fortranlib-autodetect=<bool>',     nargs.ArgBool(None, 1, 'Autodetect Fortran compiler libraries'))
    help.addArgument('Compilers', '-with-cxxlib-autodetect=<bool>',         nargs.ArgBool(None, 1, 'Autodetect C++ compiler libraries'))
    help.addArgument('Compilers', '-with-dependencies=<bool>',              nargs.ArgBool(None, 1, 'Compile with -MMD or equivalent flag if possible'))
    help.addArgument('Compilers', '-with-cxx-dialect=<dialect>',            nargs.Arg(None, 'auto', 'Dialect under which to compile C++ sources (auto,cxx14,cxx11,0)'))
    help.addArgument('Compilers', '-with-hip-dialect=<dialect>',            nargs.Arg(None, 'auto', 'Dialect under which to compile HIP sources (auto,cxx14,cxx11,0)'))
    help.addArgument('Compilers', '-with-cuda-dialect=<dialect>',           nargs.Arg(None, 'auto', 'Dialect under which to compile CUDA sources (auto,cxx14,cxx11,0)'))
    return

  def getDispatchNames(self):
    '''Return all the attributes which are dispatched from config.setCompilers'''
    names = {}
    names['CC'] = 'No C compiler found.'
    names['CPP'] = 'No C preprocessor found.'
    names['CUDAC'] = 'No CUDA compiler found.'
    names['CUDAPP'] = 'No CUDA preprocessor found.'
    names['HIPC'] = 'No HIP compiler found.'
    names['HIPPP'] = 'No HIP preprocessor found.'
    names['SYCLCXX'] = 'No SYCL compiler found.'
    names['SYCLPP'] = 'No SYCL preprocessor found.'
    names['CXX'] = 'No C++ compiler found.'
    names['CXXPP'] = 'No C++ preprocessor found.'
    names['FC'] = 'No Fortran compiler found.'
    names['AR'] = 'No archiver found.'
    names['RANLIB'] = 'No ranlib found.'
    names['LD_SHARED'] = 'No shared linker found.'
    names['CC_LD'] = 'No C linker found.'
    names['dynamicLinker'] = 'No dynamic linker found.'
    for language in ['C', 'CUDA', 'HIP', 'SYCL', 'Cxx', 'FC']:
      self.pushLanguage(language)
      key = self.getCompilerFlagsName(language, 0)
      names[key] = 'No '+language+' compiler flags found.'
      key = self.getCompilerFlagsName(language, 1)
      names[key] = 'No '+language+' compiler flags found.'
      key = self.getLinkerFlagsName(language)
      names[key] = 'No '+language+' linker flags found.'
      self.popLanguage()
    names['CPPFLAGS'] = 'No preprocessor flags found.'
    names['FPPFLAGS'] = 'No Fortran preprocessor flags found.'
    names['CUDAPPFLAGS'] = 'No CUDA preprocessor flags found.'
    names['HIPPPFLAGS'] = 'No HIP preprocessor flags found.'
    names['SYCLPPFLAGS'] = 'No SYCL preprocessor flags found.'
    names['CXXPPFLAGS'] = 'No C++ preprocessor flags found.'
    names['AR_FLAGS'] = 'No archiver flags found.'
    names['AR_LIB_SUFFIX'] = 'No static library suffix found.'
    names['LIBS'] = 'No extra libraries found.'
    return names

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers', self)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.libraries = framework.require('config.libraries', None)
    self.dispatchNames = self.getDispatchNames()
    return

  def __getattr__(self, name):
    if 'dispatchNames' in self.__dict__:
      if name in self.dispatchNames:
        if not hasattr(self.setCompilers, name):
          raise MissingProcessor(self.dispatchNames[name])
        return getattr(self.setCompilers, name)
      if name in ['CC_LINKER_FLAGS', 'FC_LINKER_FLAGS', 'CXX_LINKER_FLAGS', 'CUDAC_LINKER_FLAGS', 'HIPC_LINKER_FLAGS', 'SYCLCXX_LINKER_FLAGS','sharedLibraryFlags', 'dynamicLibraryFlags']:
        flags = getattr(self.setCompilers, name)
        if not isinstance(flags, list): flags = [flags]
        return ' '.join(flags)
    raise AttributeError('Configure attribute not found: '+name)

  def __setattr__(self, name, value):
    if 'dispatchNames' in self.__dict__:
      if name in self.dispatchNames:
        return setattr(self.setCompilers, name, value)
    config.base.Configure.__setattr__(self, name, value)
    return

  # THIS SHOULD BE REWRITTEN AS checkDeclModifier()
  # checkCInline & checkCxxInline are pretty much the same code right now.
  # but they could be different (later) - and they also check/set different flags - hence
  # code duplication.
  def checkCInline(self):
    '''Check for C inline keyword'''
    self.cInlineKeyword = ' '
    self.pushLanguage('C')
    for kw in ['inline', '__inline', '__inline__']:
      if self.checkCompile('static %s int foo(int a) {return a;}' % kw, 'foo(1);'):
        self.cInlineKeyword = kw
        self.logPrint('Set C Inline keyword to '+self.cInlineKeyword , 4, 'compilers')
        break
    self.popLanguage()
    if self.cInlineKeyword == ' ':
      self.logPrint('No C Inline keyword. Using static function', 4, 'compilers')
    self.addDefine('C_INLINE', self.cInlineKeyword)
    return

  def checkCxxInline(self):
    '''Check for C++ inline keyword'''
    self.cxxInlineKeyword = ' '
    self.pushLanguage('C++')
    for kw in ['inline', '__inline', '__inline__']:
      if self.checkCompile('static %s int foo(int a) {return a;}' % kw, 'foo(1);'):
        self.cxxInlineKeyword = kw
        self.logPrint('Set Cxx Inline keyword to '+self.cxxInlineKeyword , 4, 'compilers')
        break
    self.popLanguage()
    if self.cxxInlineKeyword == ' ':
      self.logPrint('No Cxx Inline keyword. Using static function', 4, 'compilers')
    self.addDefine('CXX_INLINE', self.cxxInlineKeyword)
    return

  def checkRestrict(self,language):
    '''Check for the C/CXX restrict keyword'''
    # Try keywords equivalent to C99 restrict.  Note that many C
    # compilers require special cflags, such as -std=c99 or -restrict to
    # recognize the "restrict" keyword and it is not always practical to
    # expect that every user provides matching options.  Meanwhile,
    # compilers like Intel and MSVC (reportedly) support __restrict
    # without special options.  Glibc uses __restrict, presumably for
    # this reason.  Note that __restrict is not standardized while
    # "restrict" is, but implementation realities favor __restrict.
    if config.setCompilers.Configure.isPGI(self.setCompilers.CC, self.log):
      self.addDefine(language.upper()+'_RESTRICT', ' ')
      self.logPrint('PGI restrict word is broken cannot handle [restrict] '+str(language)+' restrict keyword', 4, 'compilers')
      return
    self.pushLanguage(language)
    for kw in ['__restrict', ' __restrict__', 'restrict']:
      if self.checkCompile('', 'float * '+kw+' x;'):
        if language.lower() == 'c':
          self.cRestrict = kw
        elif language.lower() == 'cxx':
          self.cxxRestrict = kw
        else:
          raise RuntimeError('Unknown Language :' + str(language))
        self.logPrint('Set '+str(language)+' restrict keyword to '+kw, 4, 'compilers')
        # Define to equivalent of C99 restrict keyword, or to nothing if this is not supported.
        self.addDefine(language.upper()+'_RESTRICT', kw)
        self.popLanguage()
        return
    # did not find restrict
    self.addDefine(language.upper()+'_RESTRICT', ' ')
    self.logPrint('No '+str(language)+' restrict keyword', 4, 'compilers')
    self.popLanguage()
    return

  def checkCrossLink(self, func1, func2, language1 = 'C', language2='FC',extraObjs = []):
    '''Compiles C/C++ and Fortran code and tries to link them together; the C and Fortran code are independent so no name mangling is needed
       language1 is used to compile the first code, language2 compiles the second and links them togetether'''
    obj1 = os.path.join(self.tmpDir, 'confc.o')
    found = 0
    # Compile the C test object
    self.pushLanguage(language1)
    if not self.checkCompile(func1, None, cleanup = 0):
      self.logPrint('Cannot compile C function: '+func1, 3, 'compilers')
      self.popLanguage()
      return found
    if not os.path.isfile(self.compilerObj):
      self.logPrint('Cannot locate object file: '+os.path.abspath(self.compilerObj), 3, 'compilers')
      self.popLanguage()
      return found
    os.rename(self.compilerObj, obj1)
    self.popLanguage()
    # Link the test object against a Fortran driver
    self.pushLanguage(language2)
    oldLIBS = self.setCompilers.LIBS
    self.setCompilers.LIBS = obj1+' '+self.setCompilers.LIBS
    if extraObjs:
      self.setCompilers.LIBS = ' '.join(extraObjs)+' '+' '.join([self.libraries.getLibArgument(lib) for lib in self.clibs])+' '+self.setCompilers.LIBS
    found = self.checkLink("", func2,codeBegin = " ", codeEnd = " ")
    self.setCompilers.LIBS = oldLIBS
    self.popLanguage()
    if os.path.isfile(obj1):
      os.remove(obj1)
    return found

  def checkCLibraries(self):
    '''Determines the libraries needed to link with C compiled code'''
    skipclibraries = 1
    if hasattr(self.setCompilers, 'FC'):
      self.setCompilers.saveLog()
      try:
        if self.checkCrossLink('#include <stdio.h>\nvoid asub(void)\n{char s[16];printf("testing %s",s);}\n',"     program main\n      print*,'testing'\n      stop\n      end\n",language1='C',language2='FC'):
          self.logWrite(self.setCompilers.restoreLog())
          self.logPrint('C libraries are not needed when using Fortran linker')
        else:
          self.logWrite(self.setCompilers.restoreLog())
          self.logPrint('C code cannot directly be linked with Fortran linker, therefore will determine needed C libraries')
          skipclibraries = 0
      except RuntimeError as e:
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
        self.logPrint('C code cannot directly be linked with Fortran linker, therefore will determine needed C libraries')
        skipclibraries = 0
    if hasattr(self.setCompilers, 'CXX'):
      self.setCompilers.saveLog()
      try:
        if self.checkCrossLink('#include <stdio.h>\nvoid asub(void)\n{char s[16];printf("testing %s",s);}\n',"int main(int argc,char **args)\n{return 0;}\n",language1='C',language2='C++'):
          self.logWrite(self.setCompilers.restoreLog())
          self.logPrint('C libraries are not needed when using C++ linker')
        else:
          self.logWrite(self.setCompilers.restoreLog())
          self.logPrint('C code cannot directly be linked with C++ linker, therefore will determine needed C libraries')
          skipclibraries = 0
      except RuntimeError as e:
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
        self.logPrint('C code cannot directly be linked with C++ linker, therefore will determine needed C libraries')
        skipclibraries = 0
    if skipclibraries == 1: return

    oldFlags = self.setCompilers.LDFLAGS
    self.setCompilers.LDFLAGS += ' -v'
    self.pushLanguage('C')
    (output, returnCode) = self.outputLink('', '')
    self.setCompilers.LDFLAGS = oldFlags
    self.popLanguage()

    output = remove_xcode_verbose(output)
    # PGI: kill anything enclosed in single quotes
    if output.find('\'') >= 0:
      # Cray has crazy non-matching single quotes so skip the removal
      if not output.count('\'')%2:
        while output.find('\'') >= 0:
          start = output.index('\'')
          end   = output.index('\'', start+1)+1
          output = output.replace(output[start:end], '')

    # The easiest thing to do for xlc output is to replace all the commas
    # with spaces.  Try to only do that if the output is really from xlc,
    # since doing that causes problems on other systems.
    if output.find('XL_CONFIG') >= 0:
      output = output.replace(',', ' ')

    # Parse output
    argIter = iter(output.split())
    clibs = []
    skipdefaultpaths = self.getSkipDefaultPaths()
    lflags  = []
    rpathflags = []
    try:
      while 1:
        arg = next(argIter)
        self.logPrint( 'Checking arg '+arg, 4, 'compilers')

        # Intel compiler sometimes puts " " around an option like "-lsomething"
        if arg.startswith('"') and arg.endswith('"'):
          arg = arg[1:-1]
        # Intel also puts several options together inside a " " so the last one
        # has a stray " at the end
        if arg.endswith('"') and arg[:-1].find('"') == -1:
          arg = arg[:-1]
        # Intel 11 has a bogus -long_double option
        if arg == '-long_double':
          continue
        # if options of type -L foobar
        if arg == '-lto_library':
          lib = next(argIter)
          self.logPrint('Skipping Apple LLVM linker option -lto_library '+lib)
          continue
        if arg == '-L':
          lib = next(argIter)
          self.logPrint('Found -L '+lib, 4, 'compilers')
          clibs.append('-L'+lib)
          continue
        # Check for full library name
        m = re.match(r'^/.*\.a$', arg)
        if m:
          if not arg in lflags:
            lflags.append(arg)
            self.logPrint('Found full library spec: '+arg, 4, 'compilers')
            clibs.append(arg)
          else:
            self.logPrint('Skipping, already in lflags: '+arg, 4, 'compilers')
          continue
        # Check for full dylib library name
        m = re.match(r'^/.*\.dylib$', arg)
        if m:
          if not arg in lflags:
            lflags.append(arg)
            self.logPrint('Found full library spec: '+arg, 4, 'compilers')
            clibs.append(arg)
          else:
            self.logPrint('already in lflags: '+arg, 4, 'compilers')
          continue
        # Check for system libraries
        m = re.match(r'^-l(ang.*|crt[0-9].o|crtbegin.o|c|gcc|gcc_ext(.[0-9]+)*|System|cygwin|xlomp_ser|crt[0-9].[0-9][0-9].[0-9].o)$', arg)
        if m:
          self.logPrint('Skipping system library: '+arg, 4, 'compilers')
          continue
        # Check for special library arguments
        m = re.match(r'^-l.*$', arg)
        if m:
          if not arg in lflags:
            if arg == '-lkernel32':
              continue
            else:
              lflags.append(arg)
            self.logPrint('Found library : '+arg, 4, 'compilers')
            clibs.append(arg)
          continue
        m = re.match(r'^-L.*$', arg)
        if m:
          arg = os.path.abspath(arg[2:])
          if arg in skipdefaultpaths: continue
          arg = '-L'+arg
          lflags.append(arg)
          self.logPrint('Found library directory: '+arg, 4, 'compilers')
          clibs.append(arg)
          continue
        # Check for '-rpath /sharedlibpath/ or -R /sharedlibpath/'
        if arg == '-rpath' or arg == '-R':
          lib = next(argIter)
          if lib.startswith('-'): continue # perhaps the path was striped due to quotes?
          if lib.startswith('"') and lib.endswith('"') and lib.find(' ') == -1: lib = lib[1:-1]
          lib = os.path.abspath(lib)
          if lib in skipdefaultpaths: continue
          if not lib in rpathflags:
            rpathflags.append(lib)
            self.logPrint('Found '+arg+' library: '+lib, 4, 'compilers')
            clibs.append(self.setCompilers.CSharedLinkerFlag+lib)
          else:
            self.logPrint('Already in rpathflags, skipping'+arg, 4, 'compilers')
          continue
        # Check for '-R/sharedlibpath/'
        m = re.match(r'^-R.*$', arg)
        if m:
          lib = os.path.abspath(arg[2:])
          if not lib in rpathflags:
            rpathflags.append(lib)
            self.logPrint('Found -R library: '+lib, 4, 'compilers')
            clibs.append(self.setCompilers.CSharedLinkerFlag+lib)
          else:
            self.logPrint('Already in rpathflags, skipping'+arg, 4, 'compilers')
          continue
        self.logPrint('Unknown arg '+arg, 4, 'compilers')
    except StopIteration:
      pass

    self.clibs = []
    for lib in clibs:
      if not self.setCompilers.staticLibraries and lib.startswith('-L') and not self.setCompilers.CSharedLinkerFlag == '-L':
        self.clibs.append(self.setCompilers.CSharedLinkerFlag+lib[2:])
      self.clibs.append(lib)

    self.logPrint('Libraries needed to link C code with another linker: '+str(self.clibs), 3, 'compilers')

    if hasattr(self.setCompilers, 'FC') or hasattr(self.setCompilers, 'CXX'):
      self.logPrint('Check that C libraries can be used with Fortran as linker', 4, 'compilers')
      oldLibs = self.setCompilers.LIBS
      self.setCompilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.clibs])+' '+self.setCompilers.LIBS
    if hasattr(self.setCompilers, 'FC'):
      self.setCompilers.saveLog()
      try:
        self.setCompilers.checkCompiler('FC')
      except RuntimeError as e:
        self.setCompilers.LIBS = oldLibs
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
        raise RuntimeError('C libraries cannot directly be used with Fortran as linker')
      except OSError as e:
        self.setCompilers.LIBS = oldLibs
        self.logWrite(self.setCompilers.restoreLog())
        raise e
      self.logWrite(self.setCompilers.restoreLog())
    return

  def checkCFormatting(self):
    '''Activate format string checking if using the GNU compilers'''
    '''No checking because we use additional formating conventions'''
    if self.isGCC and 0:
      self.gccFormatChecking = ('PRINTF_FORMAT_CHECK(A,B)', '__attribute__((format (printf, A, B)))')
      self.logPrint('Added gcc printf format checking', 4, 'compilers')
      self.addDefine(self.gccFormatChecking[0], self.gccFormatChecking[1])
    else:
      self.gccFormatChecking = None
    return

  def checkDynamicLoadFlag(self):
    '''Checks that dlopen() takes RTLD_XXX, and defines PETSC_HAVE_RTLD_XXX if it does'''
    if self.setCompilers.dynamicLibraries:
      if self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_LAZY);dlopen(libname, RTLD_NOW);dlopen(libname, RTLD_LOCAL);dlopen(libname, RTLD_GLOBAL);\n'):
        self.addDefine('HAVE_RTLD_LAZY', 1)
        self.addDefine('HAVE_RTLD_NOW', 1)
        self.addDefine('HAVE_RTLD_LOCAL', 1)
        self.addDefine('HAVE_RTLD_GLOBAL', 1)
        return
      if self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_LAZY);\n'):
        self.addDefine('HAVE_RTLD_LAZY', 1)
      if self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_NOW);\n'):
        self.addDefine('HAVE_RTLD_NOW', 1)
      if self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_LOCAL);\n'):
        self.addDefine('HAVE_RTLD_LOCAL', 1)
      if self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_GLOBAL);\n'):
        self.addDefine('HAVE_RTLD_GLOBAL', 1)
    return

  def checkCxxOptionalExtensions(self):
    '''Check whether the C++ compiler (IBM xlC, OSF5) need special flag for .c files which contain C++'''
    self.setCompilers.saveLog()
    self.setCompilers.pushLanguage('Cxx')
    cxxObj = self.framework.getCompilerObject('Cxx')
    oldExt = cxxObj.sourceExtension
    cxxObj.sourceExtension = self.framework.getCompilerObject('C').sourceExtension
    for flag in ['', '-+', '-x cxx -tlocal', '-Kc++']:
      try:
        self.setCompilers.addCompilerFlag(flag, body = 'class somename { int i; };')
        self.cxxCompileC = True
        break
      except RuntimeError:
        pass
    if not self.cxxCompileC:
      for flag in ['-x c++', '-TP','-P']:
        try:
          self.setCompilers.addCompilerFlag(flag, body = 'class somename { int i; };', compilerOnly = 1)
          self.cxxCompileC = True
          break
        except RuntimeError:
          pass
    cxxObj.sourceExtension = oldExt
    self.setCompilers.popLanguage()
    self.logWrite(self.setCompilers.restoreLog())
    return


  def checkCxxDialect(self,language,isGNU):
    """Determine the CXX dialect supported by the compiler(language) [and correspoding compiler option - if any].
    isGNU indicates if the compiler is g++.
    -with-<lang>-dialect can take options:
      auto: use highest dialect configure can determine
      cxx17: [future?]
      cxx14: gnu++14 or c++14
      cxx11: gnu++11 or c++11
      0: disable CxxDialect check and use compiler default
    """
    lang      = language.lower()
    LANG      = language.upper()
    TESTCXX14 = 0
    TESTCXX11 = 0
    with_lang_dialect = self.argDB.get('with-'+lang+'-dialect','').upper().replace('X','+') # configure value
    if with_lang_dialect in ['','0','NONE']: return
    elif with_lang_dialect == 'AUTO':
      TESTCXX14 = 1
      TESTCXX11 = 1
    elif with_lang_dialect == 'C++14':
      TESTCXX14 = 1
    elif with_lang_dialect == 'C++11':
      TESTCXX11 = 1
    else:
      raise RuntimeError('Unknown C++ dialect: with-'+lang+'-dialect=%s' % (self.argDB['with-'+lang+'-dialect']))

    cxxdialect  = '' # tmp var storing test result
    # Test borrowed from Jack Poulson (Elemental)
    includes = """
          #include <random>
          #include <iostream>
          #include <complex>
          template<typename T> constexpr T Cubed( T x ) { return x*x*x; }
          """
    body = """
          std::random_device rd;
          std::mt19937 mt(rd());
          std::normal_distribution<double> dist(0,1);
          const double x = dist(mt);
          std::cout << x;
          """
    body14 = """
          constexpr std::complex<double> I(0.0,1.0);
          auto lambda = [](auto x, auto y) {return x + y;};
          return lambda(3,4) + (int)std::real(I);
          """
    self.setCompilers.saveLog()
    self.setCompilers.pushLanguage(language)
    if TESTCXX14:
      flags_to_try = ['']
      if isGNU: flags_to_try += ['-std=gnu++14']
      else: flags_to_try += ['-std=c++14']
      for flag in flags_to_try:
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('checkCxxDialect: checking CXX14 for '+language+ ' with flag: '+flag)
        self.setCompilers.saveLog()
        if self.setCompilers.checkCompilerFlag(flag, includes, body+body14):
          newflag = getattr(self.setCompilers,LANG+'FLAGS') + ' ' + flag # append flag to the old
          setattr(self.setCompilers,LANG+'FLAGS',newflag)
          newflag = getattr(self.setCompilers,LANG+'PPFLAGS') + ' ' + flag # append flag to the old
          setattr(self.setCompilers,LANG+'PPFLAGS',newflag)
          cxxdialect = 'C++14'
          self.addDefine('HAVE_'+LANG+'_DIALECT_CXX14',1)
          self.addDefine('HAVE_'+LANG+'_DIALECT_CXX11',1)
          break

    if with_lang_dialect == 'C++14' and cxxdialect != 'C++14':
      self.logWrite(self.setCompilers.restoreLog())
      raise RuntimeError('Could not determine compiler flag for with-'+lang+'-dialect=%s,\nIf you know the flag, set it with '+LANG+'FLAGS option')
    elif not cxxdialect and TESTCXX11:
      flags_to_try = ['']
      if isGNU: flags_to_try += ['-std=gnu++11']
      else: flags_to_try += ['-std=c++11','-std=c++0x']
      for flag in flags_to_try:
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('checkCxxDialect: checking CXX11 for '+language+ ' with flag: '+flag)
        self.setCompilers.saveLog()
        if self.setCompilers.checkCompilerFlag(flag, includes, body):
          newflag = getattr(self.setCompilers,LANG+'FLAGS') + ' ' + flag # append flag to the old
          setattr(self.setCompilers,LANG+'FLAGS',newflag)
          newflag = getattr(self.setCompilers,LANG+'PPFLAGS') + ' ' + flag # append flag to the old
          setattr(self.setCompilers,LANG+'PPFLAGS',newflag)
          cxxdialect = 'C++11'
          self.addDefine('HAVE_'+LANG+'_DIALECT_CXX11',1)
          break

    if with_lang_dialect == 'C++11' and cxxdialect != 'C++11':
      self.logWrite(self.setCompilers.restoreLog())
      raise RuntimeError('Could not determine compiler flag for with-'+lang+'-dialect=%s,\nIf you know the flag, set it with '+LANG+'FLAGS option')

    setattr(self,lang+'dialect',cxxdialect) # record the result
    self.setCompilers.popLanguage()
    self.logWrite(self.setCompilers.restoreLog())
    return

  def checkCxxLibraries(self):
    '''Determines the libraries needed to link with C++ from C and Fortran'''
    skipcxxlibraries = 1
    self.setCompilers.saveLog()
    body   = '''#include <iostream>\n#include <vector>\nvoid asub(void)\n{std::vector<int> v;\ntry  { throw 20;  }  catch (int e)  { std::cout << "An exception occurred";  }}'''
    try:
      if self.checkCrossLink(body,"int main(int argc,char **args)\n{return 0;}\n",language1='C++',language2='C'):
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('C++ libraries are not needed when using C linker')
      else:
        skipcxxlibraries = 0
        self.logWrite(self.setCompilers.restoreLog())
        if self.setCompilers.isDarwin(self.log) and config.setCompilers.Configure.isClang(self.getCompiler('C'), self.log):
          oldLibs = self.setCompilers.LIBS
          self.setCompilers.LIBS = '-lc++ '+self.setCompilers.LIBS
          self.setCompilers.saveLog()
          if self.checkCrossLink(body,"int main(int argc,char **args)\n{return 0;}\n",language1='C++',language2='C'):
            self.logWrite(self.setCompilers.restoreLog())
            self.logPrint('C++ requires -lc++ to link with C compiler', 3, 'compilers')
            skipcxxlibraries = 1
          else:
            self.logWrite(self.setCompilers.restoreLog())
            self.setCompilers.LIBS = oldLibs
            self.logPrint('C++ code cannot directly be linked with C linker using -lc++, therefore will determine needed C++ libraries')
            skipcxxlibraries = 0
        if not skipcxxlibraries:
          self.setCompilers.saveLog()
          oldLibs = self.setCompilers.LIBS
          self.setCompilers.LIBS = '-lstdc++ '+self.setCompilers.LIBS
          if self.checkCrossLink(body,"int main(int argc,char **args)\n{return 0;}\n",language1='C++',language2='C'):
            self.logWrite(self.setCompilers.restoreLog())
            self.logPrint('C++ requires -lstdc++ to link with C compiler', 3, 'compilers')
            skipcxxlibraries = 1
          else:
            self.logWrite(self.setCompilers.restoreLog())
            self.setCompilers.LIBS = oldLibs
            self.logPrint('C++ code cannot directly be linked with C linker using -lstdc++, therefore will determine needed C++ libraries')
            skipcxxlibraries = 0
    except RuntimeError as e:
      self.logWrite(self.setCompilers.restoreLog())
      self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
      self.logPrint('C++ code cannot directly be linked with C linker, therefore will determine needed C++ libraries')
      skipcxxlibraries = 0
    if hasattr(self.setCompilers, 'FC'):
      self.setCompilers.saveLog()
      try:
        if self.checkCrossLink(body,"     program main\n      print*,'testing'\n      stop\n      end\n",language1='C++',language2='FC'):
          self.logWrite(self.setCompilers.restoreLog())
          self.logPrint('C++ libraries are not needed when using FC linker')
        else:
          skipcxxlibraries = 0
          if self.setCompilers.isDarwin(self.log) and config.setCompilers.Configure.isClang(self.getCompiler('C'), self.log):
            oldLibs = self.setCompilers.LIBS
            self.setCompilers.LIBS = '-lc++ '+self.setCompilers.LIBS
            self.setCompilers.saveLog()
            if self.checkCrossLink(body,"     program main\n      print*,'testing'\n      stop\n      end\n",language1='C++',language2='FC'):
              self.logWrite(self.setCompilers.restoreLog())
              self.logPrint('C++ requires -lc++ to link with FC compiler', 3, 'compilers')
              skipcxxlibraries = 1
            else:
              self.logWrite(self.setCompilers.restoreLog())
              self.setCompilers.LIBS = oldLibs
              self.logPrint('C++ code cannot directly be linked with C linker using -lc++, therefore will determine needed C++ libraries')
              skipcxxlibraries = 0
          if not skipcxxlibraries:
            self.logWrite(self.setCompilers.restoreLog())
            oldLibs = self.setCompilers.LIBS
            self.setCompilers.LIBS = '-lstdc++ '+self.setCompilers.LIBS
            if self.checkCrossLink(body,"     program main\n      print*,'testing'\n      stop\n      end\n",language1='C++',language2='FC'):
              self.logPrint('C++ requires -lstdc++ to link with FC compiler', 3, 'compilers')
              skipcxxlibraries = 1
            else:
              self.setCompilers.LIBS = oldLibs
              self.logPrint('C++ code cannot directly be linked with FC linker, therefore will determine needed C++ libraries')
              skipcxxlibraries = 0
      except RuntimeError as e:
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
        self.logPrint('C++ code cannot directly be linked with FC linker, therefore will determine needed C++ libraries')
        skipcxxlibraries = 0

    if skipcxxlibraries: return

    oldFlags = self.setCompilers.LDFLAGS
    self.setCompilers.LDFLAGS += ' -v'
    self.pushLanguage('Cxx')
    (output, returnCode) = self.outputLink('', '')
    self.setCompilers.LDFLAGS = oldFlags
    self.popLanguage()

    output = remove_xcode_verbose(output)
    # PGI: kill anything enclosed in single quotes
    if output.find('\'') >= 0:
      if output.count('\'')%2: raise RuntimeError('Mismatched single quotes in C library string')
      while output.find('\'') >= 0:
        start = output.index('\'')
        end   = output.index('\'', start+1)+1
        output = output.replace(output[start:end], '')

    # The easiest thing to do for xlc output is to replace all the commas
    # with spaces.  Try to only do that if the output is really from xlc,
    # since doing that causes problems on other systems.
    if output.find('XL_CONFIG') >= 0:
      output = output.replace(',', ' ')

    # Parse output
    argIter = iter(output.split())
    cxxlibs = []
    skipdefaultpaths = self.getSkipDefaultPaths()
    lflags  = []
    rpathflags = []
    try:
      while 1:
        arg = next(argIter)
        self.logPrint( 'Checking arg '+arg, 4, 'compilers')

        # Intel compiler sometimes puts " " around an option like "-lsomething"
        if arg.startswith('"') and arg.endswith('"'):
          arg = arg[1:-1]
        # Intel also puts several options together inside a " " so the last one
        # has a stray " at the end
        if arg.endswith('"') and arg[:-1].find('"') == -1:
          arg = arg[:-1]
        # Intel 11 has a bogus -long_double option
        if arg == '-long_double':
          continue

        # if options of type -L foobar
        if arg == '-L':
          lib = next(argIter)
          self.logPrint('Found -L '+lib, 4, 'compilers')
          cxxlibs.append('-L'+lib)
          continue
        if arg == '-lto_library':
          lib = next(argIter)
          self.logPrint('Skipping Apple LLVM linker option -lto_library '+lib)
          continue
        # Check for full library name
        m = re.match(r'^/.*\.a$', arg)
        if m:
          if not arg in lflags:
            lflags.append(arg)
            self.logPrint('Found full library spec: '+arg, 4, 'compilers')
            cxxlibs.append(arg)
          else:
            self.logPrint('Already in lflags: '+arg, 4, 'compilers')
          continue
        # Check for full dylib library name
        m = re.match(r'^/.*\.dylib$', arg)
        if m:
          if not arg in lflags and not arg.endswith('LTO.dylib'):
            lflags.append(arg)
            self.logPrint('Found full library spec: '+arg, 4, 'compilers')
            cxxlibs.append(arg)
          else:
            self.logPrint('already in lflags: '+arg, 4, 'compilers')
          continue
        # Check for system libraries
        m = re.match(r'^-l(ang.*|crt[0-9].o|crtbegin.o|c|gcc|gcc_ext(.[0-9]+)*|System|cygwin|xlomp_ser|crt[0-9].[0-9][0-9].[0-9].o)$', arg)
        if m:
          self.logPrint('Skipping system library: '+arg, 4, 'compilers')
          continue
        # Check for special library arguments
        m = re.match(r'^-l.*$', arg)
        if m:
          if not arg in lflags:
            if arg == '-lkernel32':
              continue
            elif arg == '-lLTO' and self.setCompilers.isDarwin(self.log):
              self.logPrint('Skipping -lTO')
            else:
              lflags.append(arg)
            self.logPrint('Found library: '+arg, 4, 'compilers')
            if (arg == '-lLTO' and self.setCompilers.isDarwin(self.log)) or arg in self.clibs:
              self.logPrint('Library already in C list so skipping in C++')
            else:
              cxxlibs.append(arg)
          continue
        m = re.match(r'^-L.*$', arg)
        if m:
          arg = os.path.abspath(arg[2:])
          if arg in skipdefaultpaths: continue
          arg = '-L'+arg
          if not arg in lflags:
            lflags.append(arg)
            self.logPrint('Found library directory: '+arg, 4, 'compilers')
            cxxlibs.append(arg)
          continue
        # Check for '-rpath /sharedlibpath/ or -R /sharedlibpath/'
        if arg == '-rpath' or arg == '-R':
          lib = next(argIter)
          if lib.startswith('-'): continue # perhaps the path was striped due to quotes?
          if lib.startswith('"') and lib.endswith('"') and lib.find(' ') == -1: lib = lib[1:-1]
          lib = os.path.abspath(lib)
          if lib in skipdefaultpaths: continue
          if not lib in rpathflags:
            rpathflags.append(lib)
            self.logPrint('Found '+arg+' library: '+lib, 4, 'compilers')
            cxxlibs.append(self.setCompilers.CSharedLinkerFlag+lib)
          else:
            self.logPrint('Already in rpathflags, skipping:'+arg, 4, 'compilers')
          continue
        # Check for '-R/sharedlibpath/'
        m = re.match(r'^-R.*$', arg)
        if m:
          lib = os.path.abspath(arg[2:])
          if not lib in rpathflags:
            rpathflags.append(lib)
            self.logPrint('Found -R library: '+lib, 4, 'compilers')
            cxxlibs.append(self.setCompilers.CSharedLinkerFlag+lib)
          else:
            self.logPrint('Already in rpathflags, skipping:'+arg, 4, 'compilers')
          continue
        self.logPrint('Unknown arg '+arg, 4, 'compilers')
    except StopIteration:
      pass

    self.cxxlibs = []
    for lib in cxxlibs:
      if not self.setCompilers.staticLibraries and lib.startswith('-L') and not self.setCompilers.CSharedLinkerFlag == '-L':
        self.cxxlibs.append(self.setCompilers.CSharedLinkerFlag+lib[2:])
      self.cxxlibs.append(lib)

    self.logPrint('Libraries needed to link Cxx code with another linker: '+str(self.cxxlibs), 3, 'compilers')

    self.logPrint('Check that Cxx libraries can be used with C as linker', 4, 'compilers')
    oldLibs = self.setCompilers.LIBS
    self.setCompilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.cxxlibs])+' '+self.setCompilers.LIBS
    self.setCompilers.saveLog()
    try:
      self.setCompilers.checkCompiler('C')
    except RuntimeError as e:
      self.logWrite(self.setCompilers.restoreLog())
      self.logPrint('Cxx libraries cannot directly be used with C as linker', 4, 'compilers')
      self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
      raise RuntimeError("Cxx libraries cannot directly be used with C as linker.\n\
If you don't need the C++ compiler to build external packages or for you application you can run\n\
./configure with --with-cxx=0. Otherwise you need a different combination of C and C++ compilers")
    else:
      self.logWrite(self.setCompilers.restoreLog())
    self.setCompilers.LIBS = oldLibs

    if hasattr(self.setCompilers, 'FC'):

      self.logPrint('Check that Cxx libraries can be used with Fortran as linker', 4, 'compilers')
      oldLibs = self.setCompilers.LIBS
      self.setCompilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.cxxlibs])+' '+self.setCompilers.LIBS
      self.setCompilers.saveLog()
      try:
        self.setCompilers.checkCompiler('FC')
      except RuntimeError as e:
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('Cxx libraries cannot directly be used with Fortran as linker', 4, 'compilers')
        self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
        raise RuntimeError("Cxx libraries cannot directly be used with Fortran as linker.\n\
If you don't need the C++ compiler to build external packages or for you application you can run\n\
./configure with --with-cxx=0. If you don't need the Fortran compiler to build external packages\n\
or for you application you can run ./configure with --with-fc=0.\n\
Otherwise you need a different combination of C, C++, and Fortran compilers")
      else:
        self.logWrite(self.setCompilers.restoreLog())
      self.setCompilers.LIBS = oldLibs
    return

  def mangleFortranFunction(self, name):
    if self.fortranMangling == 'underscore':
      if self.fortranManglingDoubleUnderscore and name.find('_') >= 0:
        return name.lower()+'__'
      else:
        return name.lower()+'_'
    elif self.fortranMangling == 'unchanged':
      return name.lower()
    elif self.fortranMangling == 'caps':
      return name.upper()
    elif self.fortranMangling == 'stdcall':
      return name.upper()
    raise RuntimeError('Unknown Fortran name mangling: '+self.fortranMangling)

  def testMangling(self, cfunc, ffunc, clanguage = 'C', extraObjs = []):
    '''Test a certain name mangling'''
    cobj = os.path.join(self.tmpDir, 'confc.o')
    found = 0
    # Compile the C test object
    self.pushLanguage(clanguage)
    if not self.checkCompile(cfunc, None, cleanup = 0):
      self.logPrint('Cannot compile C function: '+cfunc, 3, 'compilers')
      self.popLanguage()
      return found
    if not os.path.isfile(self.compilerObj):
      self.logPrint('Cannot locate object file: '+os.path.abspath(self.compilerObj), 3, 'compilers')
      self.popLanguage()
      return found
    os.rename(self.compilerObj, cobj)
    self.popLanguage()
    # Link the test object against a Fortran driver
    self.pushLanguage('FC')
    oldLIBS = self.setCompilers.LIBS
    self.setCompilers.LIBS = cobj+' '+' '.join([self.libraries.getLibArgument(lib) for lib in self.clibs])+' '+self.setCompilers.LIBS
    if extraObjs:
      self.setCompilers.LIBS = ' '.join(extraObjs)+' '+' '.join([self.libraries.getLibArgument(lib) for lib in self.clibs])+' '+self.setCompilers.LIBS
    found = self.checkLink(None, ffunc)
    self.setCompilers.LIBS = oldLIBS
    self.popLanguage()
    if os.path.isfile(cobj):
      os.remove(cobj)
    return found

  def checkFortranNameMangling(self):
    '''Checks Fortran name mangling, and defines HAVE_FORTRAN_UNDERSCORE, HAVE_FORTRAN_NOUNDERSCORE, HAVE_FORTRAN_CAPS'''
    self.manglerFuncs = {'underscore': ('void d1chk_(void);', 'void d1chk_(void){return;}\n', '       call d1chk()\n'),
                         'unchanged': ('void d1chk(void);', 'void d1chk(void){return;}\n', '       call d1chk()\n'),
                         'caps': ('void D1CHK(void);', 'void D1CHK(void){return;}\n', '       call d1chk()\n'),
                         'stdcall': ('void __stdcall D1CHK(void);', 'void __stdcall D1CHK(void){return;}\n', '       call d1chk()\n'),
                         'double': ('void d1_chk__(void)', 'void d1_chk__(void){return;}\n', '       call d1_chk()\n')}
    #some compilers silently ignore '__stdcall' directive, so do stdcall test last
    # double test is not done here, so its not listed
    key_list = ['underscore','unchanged','caps','stdcall']
    for mangler in key_list:
      cfunc = self.manglerFuncs[mangler][1]
      ffunc = self.manglerFuncs[mangler][2]
      self.logWrite('Testing Fortran mangling type '+mangler+' with code '+cfunc)
      if self.testMangling(cfunc, ffunc):
        self.fortranMangling = mangler
        break
    else:
      if self.setCompilers.isDarwin(self.log):
        mess = '  See https://petsc.org/release/faq/#macos-gfortran'
      else:
        mess = ''
      raise RuntimeError('Unknown Fortran name mangling: Are you sure the C and Fortran compilers are compatible?\n  Perhaps one is 64 bit and one is 32 bit?\n'+mess)
    self.logPrint('Fortran name mangling is '+self.fortranMangling, 4, 'compilers')
    if self.fortranMangling == 'underscore':
      self.addDefine('HAVE_FORTRAN_UNDERSCORE', 1)
    elif self.fortranMangling == 'unchanged':
      self.addDefine('HAVE_FORTRAN_NOUNDERSCORE', 1)
    elif self.fortranMangling == 'caps':
      self.addDefine('HAVE_FORTRAN_CAPS', 1)
    elif self.fortranMangling == 'stdcall':
      raise RuntimeError('Fortran STDCALL compilers are unsupported!\n')
    if config.setCompilers.Configure.isGfortran8plus(self.getCompiler('FC'), self.log):
      self.addDefine('FORTRAN_CHARLEN_T', 'size_t')
    else:
      self.addDefine('FORTRAN_CHARLEN_T', 'int')
    return

  def checkFortranNameManglingDouble(self):
    '''Checks if symbols containing an underscore append an extra underscore, and defines HAVE_FORTRAN_UNDERSCORE_UNDERSCORE if necessary'''
    if self.testMangling(self.manglerFuncs['double'][1], self.manglerFuncs['double'][2]):
      self.logPrint('Fortran appends an extra underscore to names containing underscores', 4, 'compilers')
      self.fortranManglingDoubleUnderscore = 1
      self.addDefine('HAVE_FORTRAN_UNDERSCORE_UNDERSCORE',1)
    else:
      self.fortranManglingDoubleUnderscore = 0
    return

  def checkFortranLibraries(self):
    '''Substitutes for FLIBS the libraries needed to link with Fortran

    This macro is intended to be used in those situations when it is
    necessary to mix, e.g. C++ and Fortran 77, source code into a single
    program or shared library.

    For example, if object files from a C++ and Fortran 77 compiler must
    be linked together, then the C++ compiler/linker must be used for
    linking (since special C++-ish things need to happen at link time
    like calling global constructors, instantiating templates, enabling
    exception support, etc.).

    However, the Fortran 77 intrinsic and run-time libraries must be
    linked in as well, but the C++ compiler/linker does not know how to
    add these Fortran 77 libraries.

    This code was translated from the autoconf macro which was packaged in
    its current form by Matthew D. Langston <langston@SLAC.Stanford.EDU>.
    However, nearly all of this macro came from the OCTAVE_FLIBS macro in
    octave-2.0.13/aclocal.m4, and full credit should go to John W. Eaton
    for writing this extremely useful macro.'''
    if not hasattr(self.setCompilers, 'CC') or not hasattr(self.setCompilers, 'FC'):
      return
    skipfortranlibraries = 1
    self.setCompilers.saveLog()
    asub=self.mangleFortranFunction("asub")
    cbody = "extern void "+asub+"(void);\nint main(int argc,char **args)\n{\n  "+asub+"();\n  return 0;\n}\n";
    self.pushLanguage('FC')
    if self.checkLink(includes='#include <mpif.h>',body='      call MPI_Allreduce()\n'):
      fbody = "      subroutine asub()\n      print*,'testing'\n      call MPI_Allreduce()\n      return\n      end\n"
    else:
      fbody = "      subroutine asub()\n      print*,'testing'\n      return\n      end\n"
    self.popLanguage()
    try:
      if self.checkCrossLink(fbody,cbody,language1='FC',language2='C'):
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('Fortran libraries are not needed when using C linker')
      else:
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('Fortran code cannot directly be linked with C linker, therefore will determine needed Fortran libraries')
        skipfortranlibraries = 0
    except RuntimeError as e:
      self.logWrite(self.setCompilers.restoreLog())
      self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
      self.logPrint('Fortran code cannot directly be linked with C linker, therefore will determine needed Fortran libraries')
      skipfortranlibraries = 0
    if skipfortranlibraries and hasattr(self.setCompilers, 'CXX'):
      self.setCompilers.saveLog()
      try:
        if self.checkCrossLink(fbody,cbody,language1='FC',language2='C++'):
          self.logWrite(self.setCompilers.restoreLog())
          self.logPrint('Fortran libraries are not needed when using C++ linker')
        else:
          self.logWrite(self.setCompilers.restoreLog())
          self.logPrint('Fortran code cannot directly be linked with C++ linker, therefore will determine needed Fortran libraries')
          skipfortranlibraries = 0
      except RuntimeError as e:
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
        self.logPrint('Fortran code cannot directly be linked with CXX linker, therefore will determine needed Fortran libraries')
        skipfortranlibraries = 0

    if skipfortranlibraries == 1: return

    self.pushLanguage('FC')
    oldFlags = self.setCompilers.LDFLAGS
    self.setCompilers.LDFLAGS += ' -v'
    (output, returnCode) = self.outputLink('', '')
    if returnCode: raise RuntimeError('Unable to run linker to determine needed Fortran libraries')
    output = self.filterLinkOutput(output)
    self.setCompilers.LDFLAGS = oldFlags
    self.popLanguage()

    output = remove_xcode_verbose(output)
    # replace \CR that ifc puts in each line of output
    output = output.replace('\\\n', '')

    if output.lower().find('absoft') >= 0:
      loc = output.find(' -lf90math')
      if loc == -1: loc = output.find(' -lf77math')
      if loc >= -1:
        output = output[0:loc]+' -lU77 -lV77 '+output[loc:]

    # PGI/Windows: to properly resolve symbols, we need to list the fortran runtime libraries before -lpgf90
    # PGI Fortran compiler uses PETSC_HAVE_F90_2PTR_ARG which is incompatible with
    # certain PETSc example uses of Fortran (like passing classes) hence we need to define
    # HAVE_PGF90_COMPILER so those examples are not run
    if output.find(' -lpgf90') >= 0 and output.find(' -lkernel32') >= 0:
      loc  = output.find(' -lpgf90')
      loc2 = output.find(' -lpgf90rtl -lpgftnrtl')
      if loc2 >= -1:
        output = output[0:loc] + ' -lpgf90rtl -lpgftnrtl' + output[loc:]
    elif output.find(' -lpgf90rtl -lpgftnrtl') >= 0:
      # somehow doing this hacky thing appears to get rid of error with undefined __hpf_exit
      self.logPrint('Adding -lpgftnrtl before -lpgf90rtl in librarylist')
      output = output.replace(' -lpgf90rtl -lpgftnrtl',' -lpgftnrtl -lpgf90rtl -lpgftnrtl')

    # PGI: kill anything enclosed in single quotes
    if output.find('\'') >= 0:
      if output.count('\'')%2: raise RuntimeError('Mismatched single quotes in Fortran library string')
      while output.find('\'') >= 0:
        start = output.index('\'')
        end   = output.index('\'', start+1)+1
        output = output.replace(output[start:end], '')

    # The easiest thing to do for xlf output is to replace all the commas
    # with spaces.  Try to only do that if the output is really from xlf,
    # since doing that causes problems on other systems.
    if output.find('XL_CONFIG') >= 0:
      output = output.replace(',', ' ')
    # We are only supposed to find LD_RUN_PATH on Solaris systems
    # and the run path should be absolute
    ldRunPath = re.findall(r'^.*LD_RUN_PATH *= *([^ ]*).*', output)
    if ldRunPath: ldRunPath = ldRunPath[0]
    if ldRunPath and ldRunPath[0] == '/':
      if self.isGCC:
        ldRunPath = ['-Xlinker -R -Xlinker '+ldRunPath]
      else:
        ldRunPath = ['-R '+ldRunPath]
    else:
      ldRunPath = []

    # Parse output
    argIter = iter(output.split())
    fincs   = []
    flibs   = []
    skipdefaultpaths = self.getSkipDefaultPaths()
    fmainlibs = []
    lflags  = []
    rpathflags = []
    try:
      while 1:
        arg = next(argIter)
        self.logPrint( 'Checking arg '+arg, 4, 'compilers')
        # Intel compiler sometimes puts " " around an option like "-lsomething"
        if arg.startswith('"') and arg.endswith('"'):
          arg = arg[1:-1]
        # Intel also puts several options together inside a " " so the last one
        # has a stray " at the end
        if arg.endswith('"') and arg[:-1].find('"') == -1:
          arg = arg[:-1]

        if arg == '-lto_library':
          lib = next(argIter)
          self.logPrint('Skipping Apple LLVM linker option -lto_library '+lib)
          continue
        # Check for full library name
        m = re.match(r'^/.*\.a$', arg)
        if m:
          if not arg in lflags:
            lflags.append(arg)
            self.logPrint('Found full library spec: '+arg, 4, 'compilers')
#            # check for Nag Fortran library that must be handled as static because shared version does not have all the symbols
            base = os.path.basename(arg)
            m = re.match(r'libf[1-9][0-9]rts.a', base)
            if m:
              self.logPrint('Detected Nag Fortran compiler library; preserving as static library: '+arg, 4, 'compilers')
              flibs.append(arg)
              flibs.append('-Wl,-Bstatic')
              flibs.append(arg)
              flibs.append('-Wl,-Bdynamic')
            else:
              flibs.append(arg)
          else:
            self.logPrint('already in lflags: '+arg, 4, 'compilers')
          continue
        # Check for full dylib library name
        m = re.match(r'^/.*\.dylib$', arg)
        if m:
          if not arg.endswith('LTO.dylib') and not arg in lflags:
            lflags.append(arg)
            self.logPrint('Found full library spec: '+arg, 4, 'compilers')
            flibs.append(arg)
          else:
            self.logPrint('already in lflags: '+arg, 4, 'compilers')
          continue
        # prevent false positives for include with pathscalr
        if re.match(r'^-INTERNAL.*$', arg): continue
        # Check for special include argument
        # AIX does this for MPI and perhaps other things
        m = re.match(r'^-I.*$', arg)
        if m:
          inc = arg.replace('-I','',1)
          self.logPrint('Found include directory: '+inc, 4, 'compilers')
          fincs.append(inc)
          continue
        # Check for ???
        m = re.match(r'^-bI:.*$', arg)
        if m:
          if not arg in lflags:
            if self.isGCC:
              lflags.append('-Xlinker')
            lflags.append(arg)
            self.logPrint('Found binary include: '+arg, 4, 'compilers')
            flibs.append(arg)
          else:
            self.logPrint('Already in lflags so skipping: '+arg, 4, 'compilers')
          continue
        # Check for system libraries
        m = re.match(r'^-l(ang.*|crt[0-9].o|crtbegin.o|c|gcc|gcc_ext(.[0-9]+)*|System|cygwin|xlomp_ser|crt[0-9].[0-9][0-9].[0-9].o)$', arg)
        if m:
          self.logPrint('Found system library therefore skipping: '+arg, 4, 'compilers')
          continue
        # Check for canonical library argument
        m = re.match(r'^-[lL]$', arg)
        if m:
          lib = arg+next(argIter)
          self.logPrint('Found canonical library: '+lib, 4, 'compilers')
          if not lib == '-LLTO' or not self.setCompilers.isDarwin(self.log):
            flibs.append(lib)
          continue
        # intel windows compilers can use -libpath argument
        if arg.find('-libpath:')>=0:
          self.logPrint('Skipping win32 ifort option: '+arg)
          continue
        # Check for special library arguments
        m = re.match(r'^-l.*$', arg)
        if m:
          # HP Fortran prints these libraries in a very strange way
          if arg == '-l:libU77.a':  arg = '-lU77'
          if arg == '-l:libF90.a':  arg = '-lF90'
          if arg == '-l:libIO77.a': arg = '-lIO77'
          if not arg in lflags:
            if arg == '-lkernel32':
              continue
            elif arg == '-lgfortranbegin':
              fmainlibs.append(arg)
              continue
            elif arg == '-lfrtbegin' and not config.setCompilers.Configure.isCygwin(self.log):
              fmainlibs.append(arg)
              continue
            elif not arg == '-lLTO' or not config.setCompilers.Configure.isDarwin(self.log):
              lflags.append(arg)
            self.logPrint('Found library: '+arg, 4, 'compilers')
            if arg in self.clibs:
              self.logPrint('Library already in C list so skipping in Fortran')
            elif not arg == '-lLTO' or not config.setCompilers.Configure.isDarwin(self.log):
              flibs.append(arg)
          else:
            self.logPrint('Already in lflags: '+arg, 4, 'compilers')
          continue
        m = re.match(r'^-L.*$', arg)
        if m:
          arg = os.path.abspath(arg[2:])
          if arg in skipdefaultpaths: continue
          arg = '-L'+arg
          if not arg in lflags:
            lflags.append(arg)
            self.logPrint('Found library directory: '+arg, 4, 'compilers')
            flibs.append(arg)
          else:
            self.logPrint('Already in lflags so skipping: '+arg, 4, 'compilers')
          continue
        # Check for '-rpath /sharedlibpath/ or -R /sharedlibpath/'
        if arg == '-rpath' or arg == '-R':
          lib = next(argIter)
          if lib == '\\': lib = next(argIter)
          if lib.startswith('-'): continue # perhaps the path was striped due to quotes?
          if lib.startswith('"') and lib.endswith('"') and lib.find(' ') == -1: lib = lib[1:-1]
          lib = os.path.abspath(lib)
          if lib in skipdefaultpaths: continue
          if not lib in rpathflags:
            rpathflags.append(lib)
            self.logPrint('Found '+arg+' library: '+lib, 4, 'compilers')
            flibs.append(self.setCompilers.CSharedLinkerFlag+lib)
          else:
            self.logPrint('Already in rpathflags so skipping: '+arg, 4, 'compilers')
          continue
        # Check for '-R/sharedlibpath/'
        m = re.match(r'^-R.*$', arg)
        if m:
          lib = os.path.abspath(arg[2:])
          if not lib in rpathflags:
            rpathflags.append(lib)
            self.logPrint('Found -R library: '+lib, 4, 'compilers')
            flibs.append(self.setCompilers.CSharedLinkerFlag+lib)
          else:
            self.logPrint('Already in rpathflags so skipping: '+arg, 4, 'compilers')
          continue
        if arg.startswith('-zallextract') or arg.startswith('-zdefaultextract') or arg.startswith('-zweakextract'):
          self.logWrite( 'Found Solaris -z option: '+arg+'\n')
          flibs.append(arg)
          continue
        # Check for ???
        # Should probably try to ensure unique directory options here too.
        # This probably only applies to Solaris systems, and then will only
        # work with gcc...
        if arg == '-Y':
          libs = next(argIter)
          if libs.startswith('"') and libs.endswith('"'):
            libs = libs[1:-1]
          for lib in libs.split(':'):
            #solaris gnu g77 has this extra P, here, not sure why it means
            if lib.startswith('P,'):lib = lib[2:]
            self.logPrint('Handling -Y option: '+lib, 4, 'compilers')
            lib1 = os.path.abspath(lib)
            if lib1 in skipdefaultpaths: continue
            lib1 = '-L'+lib1
            flibs.append(lib1)
          continue
        if arg.startswith('COMPILER_PATH=') or arg.startswith('LIBRARY_PATH='):
          self.logPrint('Skipping arg '+arg, 4, 'compilers')
          continue
        # HPUX lists a bunch of library directories separated by :
        if arg.find(':') >=0:
          founddir = 0
          for l in arg.split(':'):
            if os.path.isdir(l):
              lib1 = os.path.abspath(l)
              if lib1 in skipdefaultpaths: continue
              lib1 = '-L'+lib1
              if not arg in lflags:
                flibs.append(lib1)
                lflags.append(lib1)
                self.logPrint('Handling HPUX list of directories: '+l, 4, 'compilers')
                founddir = 1
          if founddir:
            continue
        if arg.find('f61init.o')>=0 or arg.find('quickfit.o')>=0:
          flibs.append(arg)
          self.logPrint('Found quickfit.o in argument, adding it')
          continue
        # gcc+pgf90 might require pgi.dl
        if arg.find('pgi.ld')>=0:
          flibs.append(arg)
          self.logPrint('Found strange PGI file ending with .ld, adding it')
          continue
        self.logPrint('Unknown arg '+arg, 4, 'compilers')
    except StopIteration:
      pass

    self.fincs = fincs
    self.flibs = []
    for lib in flibs:
      if not self.setCompilers.staticLibraries and lib.startswith('-L') and not self.setCompilers.FCSharedLinkerFlag == '-L':
        self.flibs.append(self.setCompilers.FCSharedLinkerFlag+lib[2:])
      self.flibs.append(lib)
    self.fmainlibs = fmainlibs
    # Append run path
    self.flibs = ldRunPath+self.flibs

    # on OS X, mixing g77 3.4 with gcc-3.3 requires using -lcc_dynamic
    for l in self.flibs:
      if l.find('-L/sw/lib/gcc/powerpc-apple-darwin') >= 0:
        self.logWrite('Detected Apple Mac Fink libraries')
        appleLib = 'libcc_dynamic.so'
        self.libraries.saveLog()
        if self.libraries.check(appleLib, 'foo'):
          self.flibs.append(self.libraries.getLibArgument(appleLib))
          self.logWrite('Adding '+self.libraries.getLibArgument(appleLib)+' so that Fortran can work with C++')
        self.logWrite(self.libraries.restoreLog())
        break

    self.logPrint('Libraries needed to link Fortran code with the C linker: '+str(self.flibs), 3, 'compilers')
    self.logPrint('Libraries needed to link Fortran main with the C linker: '+str(self.fmainlibs), 3, 'compilers')
    # check that these monster libraries can be used with C as the linker
    self.logPrint('Check that Fortran libraries can be used with C as the linker', 4, 'compilers')
    oldLibs = self.setCompilers.LIBS
    self.setCompilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])+' '+self.setCompilers.LIBS
    self.setCompilers.saveLog()
    try:
      self.setCompilers.checkCompiler('C')
    except RuntimeError as e:
      self.logWrite(self.setCompilers.restoreLog())
      self.logPrint('Fortran libraries cannot directly be used with C as the liner, try without -lcrt2.o', 4, 'compilers')
      self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
      # try removing this one
      if '-lcrt2.o' in self.flibs: self.flibs.remove('-lcrt2.o')
      self.setCompilers.LIBS = oldLibs+' '+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
      self.setCompilers.saveLog()
      try:
        self.setCompilers.checkCompiler('C')
      except RuntimeError as e:
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint('Fortran libraries still cannot directly be used with C as the linker, try without pgi.ld files', 4, 'compilers')
        self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
        tmpflibs = self.flibs
        for lib in tmpflibs:
          if lib.find('pgi.ld')>=0:
            self.flibs.remove(lib)
        self.setCompilers.LIBS = oldLibs+' '+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
        self.setCompilers.saveLog()
        try:
          self.setCompilers.checkCompiler('C')
        except:
          self.logWrite(self.setCompilers.restoreLog())
          self.logPrint(str(e), 4, 'compilers')
          raise RuntimeError('Fortran libraries cannot be used with C as linker')
      else:
        self.logWrite(self.setCompilers.restoreLog())
    else:
      self.logWrite(self.setCompilers.restoreLog())

    # check these monster libraries work from C++
    if hasattr(self.setCompilers, 'CXX'):
      self.logPrint('Check that Fortran libraries can be used with C++ as linker', 4, 'compilers')
      self.setCompilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])+' '+oldLibs
      self.setCompilers.saveLog()
      try:
        self.setCompilers.checkCompiler('Cxx')
        self.logPrint('Fortran libraries can be used from C++', 4, 'compilers')
      except RuntimeError as e:
        self.logWrite(self.setCompilers.restoreLog())
        self.logPrint(str(e), 4, 'compilers')
        # try removing this one causes grief with gnu g++ and Intel Fortran
        if '-lintrins' in self.flibs: self.flibs.remove('-lintrins')
        self.setCompilers.LIBS = oldLibs+' '+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
        self.setCompilers.saveLog()
        try:
          self.setCompilers.checkCompiler('Cxx')
        except RuntimeError as e:
          self.logWrite(self.setCompilers.restoreLog())
          self.logPrint(str(e), 4, 'compilers')
          if str(e).find('INTELf90_dclock') >= 0:
            self.logPrint('Intel 7.1 Fortran compiler cannot be used with g++ 3.2!', 2, 'compilers')
        else:
           self.logWrite(self.setCompilers.restoreLog())
        raise RuntimeError('Fortran libraries cannot be used with C++ as linker.\n Run with --with-fc=0 or --with-cxx=0')
      else:
        self.logWrite(self.setCompilers.restoreLog())

    self.setCompilers.LIBS = oldLibs
    return

  def checkFortranLinkingCxx(self):
    '''Check that Fortran can link C++ libraries'''
    link = 0
    cinc, cfunc, ffunc = self.manglerFuncs[self.fortranMangling]
    cinc = 'extern "C" '+cinc+'\n'

    cxxCode = 'void foo(void){'+self.mangleFortranFunction('d1chk')+'();}'
    cxxobj  = os.path.join(self.tmpDir, 'cxxobj.o')
    self.pushLanguage('Cxx')
    if not self.checkCompile(cinc+cxxCode, None, cleanup = 0):
      self.logPrint('Cannot compile Cxx function: '+cfunc, 3, 'compilers')
      raise RuntimeError('Fortran could not successfully link C++ objects')
    if not os.path.isfile(self.compilerObj):
      self.logPrint('Cannot locate object file: '+os.path.abspath(self.compilerObj), 3, 'compilers')
      raise RuntimeError('Fortran could not successfully link C++ objects')
    os.rename(self.compilerObj, cxxobj)
    self.popLanguage()

    if self.testMangling(cinc+cfunc, ffunc, 'Cxx', extraObjs = [cxxobj]):
      self.logPrint('Fortran can link C++ functions', 3, 'compilers')
      link = 1
    else:
      oldLibs = self.setCompilers.LIBS
      self.setCompilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.cxxlibs])+' '+self.setCompilers.LIBS
      if self.testMangling(cinc+cfunc, ffunc, 'Cxx', extraObjs = [cxxobj]):
        self.logPrint('Fortran can link C++ functions using the C++ compiler libraries', 3, 'compilers')
        link = 1
      else:
        self.setCompilers.LIBS = oldLibs
    if os.path.isfile(cxxobj):
      os.remove(cxxobj)
    if not link:
      raise RuntimeError('Fortran could not successfully link C++ objects with Fortran as linker')
    return

  def checkDependencyGenerationFlag(self):
    '''Check if -MMD works for dependency generation, and add it if it does'''
    self.generateDependencies       = {}
    self.dependenciesGenerationFlag = {}
    if not self.argDB['with-dependencies'] :
      self.logPrint("Skip checking dependency compiler options on user request")
      return
    languages = ['C']
    if hasattr(self, 'CXX'):
      languages.append('Cxx')
    # Fortran is handled in compilersfortran.py
    if hasattr(self, 'CUDAC'):
      languages.append('CUDA')
    if hasattr(self, 'HIPC'):
      languages.append('HIP')
    if hasattr(self, 'SYCLCXX'):
      languages.append('SYCL')
    for language in languages:
      self.generateDependencies[language] = 0
      self.setCompilers.saveLog()
      self.setCompilers.pushLanguage(language)
      for testFlag in ['-MMD -MP', # GCC, Intel, Clang, Pathscale
                       '-MMD',     # PGI
                       '-xMMD',    # Sun
                       '-qmakedep=gcc', # xlc
                       '-MD',
                       # Cray only supports -M, which writes to stdout
                     ]:
        try:
          self.logPrint('Trying '+language+' compiler flag '+testFlag)
          if self.setCompilers.checkCompilerFlag(testFlag, compilerOnly = 1):
            depFilename = os.path.splitext(self.setCompilers.compilerObj)[0]+'.d'
            if os.path.isfile(depFilename):
              os.remove(depFilename)
              #self.setCompilers.insertCompilerFlag(testFlag, compilerOnly = 1)
              self.framework.addMakeMacro(language.upper()+'_DEPFLAGS',testFlag)
              self.dependenciesGenerationFlag[language] = testFlag
              self.generateDependencies[language]       = 1
              break
            else:
              self.logPrint('Rejected '+language+' compiler flag '+testFlag+' because no dependency file ('+depFilename+') was generated')
          else:
            self.logPrint('Rejected '+language+' compiler flag '+testFlag)
        except RuntimeError:
          self.logPrint('Rejected '+language+' compiler flag '+testFlag)
      self.setCompilers.popLanguage()
      self.logWrite(self.setCompilers.restoreLog())
    return

  def checkC99Flag(self):
    '''Check for -std=c99 or equivalent flag'''
    includes = "#include <float.h>"
    body = """
    float x[2],y;
    y = FLT_ROUNDS;
    // c++ comment
    int j = 2;
    for (int i=0; i<2; i++){
      x[i] = i*j*y;
    }
    """
    self.setCompilers.pushLanguage('C')
    flags_to_try = ['','-std=c99','-std=gnu99','-std=c11','-std=gnu11','-c99']
    for flag in flags_to_try:
      self.setCompilers.saveLog()
      if self.setCompilers.checkCompilerFlag(flag, includes, body):
        self.logWrite(self.setCompilers.restoreLog())
        self.c99flag = flag
        self.setCompilers.CPPFLAGS += ' ' + flag
        self.framework.logPrint('Accepted C99 compile flag: '+flag)
        break
      else:
        self.logWrite(self.setCompilers.restoreLog())
    self.setCompilers.popLanguage()
    if self.c99flag is None:
      if self.isGCC: additionalErrorMsg = '\nPerhaps you have an Intel compiler environment or module set that is interferring with the GNU compilers.\nTry removing that environment or module and running ./configure again.'
      else: additionalErrorMsg = ''
      raise RuntimeError('PETSc requires c99 compiler! Configure could not determine compatible compiler flag.\nPerhaps you can specify it via CFLAGS.'+additionalErrorMsg)
    return

  def configure(self):
    import config.setCompilers
    if hasattr(self.setCompilers, 'CC'):
      self.isGCC = config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log)
      self.executeTest(self.checkC99Flag)
      self.executeTest(self.checkRestrict,['C'])
      self.executeTest(self.checkCFormatting)
      self.executeTest(self.checkCInline)
      self.executeTest(self.checkDynamicLoadFlag)
      if self.argDB['with-clib-autodetect']:
        self.executeTest(self.checkCLibraries)
      self.executeTest(self.checkDependencyGenerationFlag)
    else:
      self.isGCC = 0
    if hasattr(self.setCompilers, 'CXX'):
      self.isGCXX = config.setCompilers.Configure.isGNU(self.setCompilers.CXX, self.log)
      self.executeTest(self.checkRestrict,['Cxx'])
      self.executeTest(self.checkCxxDialect,['Cxx',self.isGCXX])
      self.executeTest(self.checkCxxOptionalExtensions)
      self.executeTest(self.checkCxxInline)
      if self.argDB['with-cxxlib-autodetect']:
        self.executeTest(self.checkCxxLibraries)
      # To skip Sun C++ compiler warnings/errors
      if config.setCompilers.Configure.isSun(self.setCompilers.CXX, self.log):
        self.addDefine('HAVE_SUN_CXX', 1)
    else:
      self.isGCXX = 0
    if hasattr(self.setCompilers, 'FC'):
      self.executeTest(self.checkFortranNameMangling)
      self.executeTest(self.checkFortranNameManglingDouble)
      if self.argDB['with-fortranlib-autodetect']:
        self.executeTest(self.checkFortranLibraries)
      if hasattr(self.setCompilers, 'CXX'):
        self.executeTest(self.checkFortranLinkingCxx)

    if hasattr(self.setCompilers, 'CUDAC'):
      self.executeTest(self.checkCxxDialect,['CUDA',False]) # Not GNU

    if hasattr(self.setCompilers, 'HIPC'):
      self.executeTest(self.checkCxxDialect,['HIP',False]) # Not GNU
    if hasattr(self.setCompilers, 'SYCL'):
        #Placeholder in case further checks are needed
        pass
    self.no_configure()
    return

  def setupFrameworkCompilers(self):
    if self.framework.compilers is None:
      self.logPrint('Setting framework compilers to this module', 2, 'compilers')
      self.framework.compilers = self
    return

  def no_configure(self):
    self.executeTest(self.setupFrameworkCompilers)
    return
