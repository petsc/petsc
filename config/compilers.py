import config.base

import re
import os

class MissingProcessor(RuntimeError):
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
    self.cRestrict = ' '
    self.cxxRestrict = ' '
    self.f90Guess = None
    return

  def __str__(self):
    if self.f90Guess:
      return 'F90 Interface: ' + self.f90Guess+'\n'
    else:
      return ''

  def setupHelp(self, help):
    import nargs

    help.addArgument('Compilers', '-with-clib-autodetect',           nargs.ArgBool(None, 1, 'Autodetect C compiler libraries'))
    help.addArgument('Compilers', '-with-fortranlib-autodetect',     nargs.ArgBool(None, 1, 'Autodetect Fortran compiler libraries'))
    help.addArgument('Compilers', '-with-cxxlib-autodetect',         nargs.ArgBool(None, 1, 'Autodetect C++ compiler libraries'))

    return

  def getDispatchNames(self):
    '''Return all the attributes which are dispatched from config.setCompilers'''
    names = {}
    names['CC'] = 'No C compiler found.'
    names['CPP'] = 'No C preprocessor found.'
    names['CXX'] = 'No C++ compiler found.'
    names['CXXCPP'] = 'No C++ preprocessor found.'
    names['FC'] = 'No Fortran compiler found.'
    names['AR'] = 'No archiver found.'
    names['RANLIB'] = 'No ranlib found.'
    names['LD_SHARED'] = 'No shared linker found.'
    names['dynamicLinker'] = 'No dynamic linker found.'
    for language in ['C', 'Cxx', 'FC']:
      self.pushLanguage(language)
      key = self.getCompilerFlagsName(language)
      names[key] = 'No '+language+' compiler flags found.'
      key = self.getCompilerFlagsName(language, 1)
      names[key] = 'No '+language+' compiler flags found.'
      key = self.getLinkerFlagsName(language)
      names[key] = 'No '+language+' linker flags found.'
      self.popLanguage()
    names['CPPFLAGS'] = 'No preprocessor flags found.'
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
      if name in ['CC_LINKER_FLAGS', 'FC_LINKER_FLAGS', 'CXX_LINKER_FLAGS', 'sharedLibraryFlags', 'dynamicLibraryFlags']:
        return ' '.join(getattr(self.setCompilers, name))
    raise AttributeError('Configure attribute not found: '+name)

  def __setattr__(self, name, value):
    if 'dispatchNames' in self.__dict__:
      if name in self.dispatchNames:
        return setattr(self.setCompilers, name, value)
    config.base.Configure.__setattr__(self, name, value)
    return

  # THIS SHOULD BE REWRITTEN AS checkDeclModifier()
  # checkCStaticInline & checkCxxStaticInline are pretty much the same code right now.
  # but they could be different (later) - and they also check/set different flags - hence
  # code duplication.
  def checkCStaticInline(self):
    '''Check for C keyword: static inline'''
    self.cStaticInlineKeyword = 'static'
    self.pushLanguage('C')
    for kw in ['static inline', 'static __inline']:
      if self.checkCompile(kw+' int foo(int a) {return a;}','int i = foo(1);'):
        self.cStaticInlineKeyword = kw
        self.logPrint('Set C StaticInline keyword to '+self.cStaticInlineKeyword , 4, 'compilers')
        break
    self.popLanguage()
    if self.cStaticInlineKeyword == 'static':
      self.logPrint('No C StaticInline keyword. using static function', 4, 'compilers')
    self.addDefine('C_STATIC_INLINE', self.cStaticInlineKeyword)
    return
  def checkCxxStaticInline(self):
    '''Check for C++ keyword: static inline'''
    self.cxxStaticInlineKeyword = 'static'
    self.pushLanguage('C++')
    for kw in ['static inline', 'static __inline']:
      if self.checkCompile(kw+' int foo(int a) {return a;}','int i = foo(1);'):
        self.cxxStaticInlineKeyword = kw
        self.logPrint('Set Cxx StaticInline keyword to '+self.cxxStaticInlineKeyword , 4, 'compilers')
        break
    self.popLanguage()
    if self.cxxStaticInlineKeyword == 'static':
      self.logPrint('No Cxx StaticInline keyword. using static function', 4, 'compilers')
    self.addDefine('CXX_STATIC_INLINE', self.cxxStaticInlineKeyword)
    return

  def checkRestrict(self,language):
    '''Check for the C/CXX restrict keyword'''
    self.pushLanguage(language)
    # Try the official restrict keyword, then gcc's __restrict__, then
    # SGI's __restrict.  __restrict has slightly different semantics than
    # restrict (it's a bit stronger, in that __restrict pointers can't
    # overlap even with non __restrict pointers), but I think it should be
    # okay under the circumstances where restrict is normally used.
    for kw in ['restrict', ' __restrict__', '__restrict']:
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

  def checkCLibraries(self):
    '''Determines the libraries needed to link with C'''
    oldFlags = self.setCompilers.LDFLAGS
    self.setCompilers.LDFLAGS += ' -v'
    self.pushLanguage('C')
    (output, returnCode) = self.outputLink('', '')
    self.setCompilers.LDFLAGS = oldFlags
    self.popLanguage()

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
    clibs = []
    lflags  = []
    rpathflags = []
    try:
      while 1:
        arg = argIter.next()
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
          lib = argIter.next()
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
          continue
        # Check for system libraries
        m = re.match(r'^-l(ang.*|crt[0-9].o|crtbegin.o|c|gcc|cygwin|crt[0-9].[0-9][0-9].[0-9].o)$', arg)
        if m: continue
        # Check for special library arguments
        m = re.match(r'^-l.*$', arg)
        if m:
          if not arg in lflags:
            if arg == '-lkernel32':
              continue
            elif arg == '-lm':
              continue
            else:
              lflags.append(arg)
            self.logPrint('Found library : '+arg, 4, 'compilers')
            clibs.append(arg)
          continue
        m = re.match(r'^-L.*$', arg)
        if m:
          arg = '-L'+os.path.abspath(arg[2:])
          if arg in ['-L/usr/lib','-L/lib','-L/usr/lib64','-L/lib64']: continue
          lflags.append(arg)
          self.logPrint('Found library directory: '+arg, 4, 'compilers')
          clibs.append(arg)
          continue
        # Check for '-rpath /sharedlibpath/ or -R /sharedlibpath/'
        if arg == '-rpath' or arg == '-R':
          lib = argIter.next()
          if lib.startswith('"') and lib.endswith('"') and lib.find(' ') == -1: lib = lib[1:-1]
          lib = os.path.abspath(lib)
          if lib in ['/usr/lib','/lib','/usr/lib64','/lib64']: continue          
          if not lib in rpathflags:          
            rpathflags.append(lib)
            self.logPrint('Found '+arg+' library: '+lib, 4, 'compilers')
            clibs.append(self.setCompilers.CSharedLinkerFlag+lib)
          continue
        # Check for '-R/sharedlibpath/'
        m = re.match(r'^-R.*$', arg)
        if m:
          lib = os.path.abspath(arg[2:])
          if not lib in rpathflags:          
            rpathflags.append(lib)
            self.logPrint('Found -R library: '+lib, 4, 'compilers')
            clibs.append(self.setCompilers.CSharedLinkerFlag+lib)
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

    if hasattr(self.setCompilers, 'FC'):
      self.logPrint('Check that C libraries can be used from Fortran', 4, 'compilers')
      oldLibs = self.setCompilers.LIBS
      self.setCompilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.clibs])+' '+self.setCompilers.LIBS
      try:
        self.setCompilers.checkCompiler('FC')
      except RuntimeError, e:
        self.setCompilers.LIBS = oldLibs
        self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
        raise RuntimeError('C libraries cannot directly be used from Fortran')
    return

  def checkCFormatting(self):
    '''Activate format string checking if using the GNU compilers'''
    if self.isGCC:
      self.gccFormatChecking = ('PRINTF_FORMAT_CHECK(A,B)', '__attribute__((format (printf, A, B)))')
      self.logPrint('Added gcc printf format checking', 4, 'compilers')
      self.addDefine(self.gccFormatChecking[0], self.gccFormatChecking[1])
    else:
      self.gccFormatChecking = None
    return

  def checkDynamicLoadFlag(self):
    '''Checks that dlopen() takes RTLD_XXX, and defines PETSC_HAVE_RTLD_XXX if it does'''
    if self.setCompilers.dynamicLibraries:
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
    self.setCompilers.pushLanguage('Cxx')
    cxxObj = self.framework.getCompilerObject('Cxx')
    oldExt = cxxObj.sourceExtension
    cxxObj.sourceExtension = self.framework.getCompilerObject('C').sourceExtension
    success=0
    for flag in ['', '-+', '-x cxx -tlocal', '-Kc++']:
      try:
        self.setCompilers.addCompilerFlag(flag, body = 'class somename { int i; };')
        success=1
        break
      except RuntimeError:
        pass
    if success==0:
      for flag in ['-TP','-P']:
        try:
          self.setCompilers.addCompilerFlag(flag, body = 'class somename { int i; };', compilerOnly = 1)
          break
        except RuntimeError:
          pass
    cxxObj.sourceExtension = oldExt
    self.setCompilers.popLanguage()
    return

  def checkCxxNamespace(self):
    '''Checks that C++ compiler supports namespaces, and if it does defines HAVE_CXX_NAMESPACE'''
    self.pushLanguage('C++')
    self.cxxNamespace = 0
    if self.checkCompile('namespace petsc {int dummy;}'):
      if self.checkCompile('template <class dummy> struct a {};\nnamespace trouble{\ntemplate <class dummy> struct a : public ::a<dummy> {};\n}\ntrouble::a<int> uugh;\n'):
        self.cxxNamespace = 1
    self.popLanguage()
    if self.cxxNamespace:
      self.logPrint('C++ has namespaces', 4, 'compilers')
      self.addDefine('HAVE_CXX_NAMESPACE', 1)
    else:
      self.logPrint('C++ does not have namespaces', 4, 'compilers')
    return

  def checkCxxLibraries(self):
    '''Determines the libraries needed to link with C++'''
    oldFlags = self.setCompilers.LDFLAGS
    self.setCompilers.LDFLAGS += ' -v'
    self.pushLanguage('Cxx')
    (output, returnCode) = self.outputLink('', '')
    self.setCompilers.LDFLAGS = oldFlags
    self.popLanguage()

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
    lflags  = []
    rpathflags = []
    try:
      while 1:
        arg = argIter.next()
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
          lib = argIter.next()
          self.logPrint('Found -L '+lib, 4, 'compilers')
          cxxlibs.append('-L'+lib)
          continue
        # Check for full library name
        m = re.match(r'^/.*\.a$', arg)
        if m:
          if not arg in lflags:
            lflags.append(arg)
            self.logPrint('Found full library spec: '+arg, 4, 'compilers')
            cxxlibs.append(arg)
          continue
        # Check for system libraries
        m = re.match(r'^-l(ang.*|crt[0-9].o|crtbegin.o|c|gcc|cygwin|crt[0-9].[0-9][0-9].[0-9].o)$', arg)
        if m: continue
        # Check for special library arguments
        m = re.match(r'^-l.*$', arg)
        if m:
          if not arg in lflags:
            if arg == '-lkernel32':
              continue
            elif arg == '-lm':
              continue
            else:
              lflags.append(arg)
            self.logPrint('Found library: '+arg, 4, 'compilers')
            if arg in self.clibs:
              self.logPrint('Library already in C list so skipping in C++')
            else:
              cxxlibs.append(arg)
          continue
        m = re.match(r'^-L.*$', arg)
        if m:
          arg = '-L'+os.path.abspath(arg[2:])
          if arg in ['-L/usr/lib','-L/lib','-L/usr/lib64','-L/lib64']: continue                    
          if not arg in lflags:
            lflags.append(arg)
            self.logPrint('Found library directory: '+arg, 4, 'compilers')
            if arg in self.clibs:
              self.logPrint('Library directory already in C list so skipping in C++')
            else:
              cxxlibs.append(arg)
          continue
        # Check for '-rpath /sharedlibpath/ or -R /sharedlibpath/'
        if arg == '-rpath' or arg == '-R':
          lib = argIter.next()
          if lib.startswith('"') and lib.endswith('"') and lib.find(' ') == -1: lib = lib[1:-1]
          lib = os.path.abspath(lib)
          if lib in ['/usr/lib','/lib','/usr/lib64','/lib64']: continue
          if not lib in rpathflags:          
            rpathflags.append(lib)
            self.logPrint('Found '+arg+' library: '+lib, 4, 'compilers')
            cxxlibs.append(self.setCompilers.CSharedLinkerFlag+lib)
          continue
        # Check for '-R/sharedlibpath/'
        m = re.match(r'^-R.*$', arg)
        if m:
          lib = os.path.abspath(arg[2:])
          if not lib in rpathflags:
            rpathflags.append(lib)
            self.logPrint('Found -R library: '+lib, 4, 'compilers')
            cxxlibs.append(self.setCompilers.CSharedLinkerFlag+lib)
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

    self.logPrint('Check that Cxx libraries can be used from C', 4, 'compilers')
    oldLibs = self.setCompilers.LIBS
    self.setCompilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.cxxlibs])+' '+self.setCompilers.LIBS
    try:
      self.setCompilers.checkCompiler('C')
    except RuntimeError, e:
      self.logPrint('Cxx libraries cannot directly be used from C', 4, 'compilers')
      self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
    self.setCompilers.LIBS = oldLibs

    if hasattr(self.setCompilers, 'FC'):
      self.logPrint('Check that Cxx libraries can be used from Fortran', 4, 'compilers')
      oldLibs = self.setCompilers.LIBS
      self.setCompilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.cxxlibs])+' '+self.setCompilers.LIBS
      try:
        self.setCompilers.checkCompiler('FC')
      except RuntimeError, e:
        self.logPrint('Cxx libraries cannot directly be used from Fortran', 4, 'compilers')
        self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
      self.setCompilers.LIBS = oldLibs
    return

  def checkFortranTypeSizes(self):
    '''Check whether real*8 is supported and suggest flags which will allow support'''
    self.pushLanguage('FC')
    # Check whether the compiler (ifc) bitches about real*8, if so try using -w90 -w to eliminate bitch
    (output, error, returnCode) = self.outputCompile('', '      real*8 variable', 1)
    if (output+error).find('Type size specifiers are an extension to standard Fortran 95') >= 0:
      oldFlags = self.setCompilers.FFLAGS
      self.setCompilers.FFLAGS += ' -w90 -w'
      (output, error, returnCode) = self.outputCompile('', '      real*8 variable', 1)
      if returnCode or (output+error).find('Type size specifiers are an extension to standard Fortran 95') >= 0:
        self.setCompilers.FFLAGS = oldFlags
      else:
        self.logPrint('Looks like ifc compiler, adding -w90 -w flags to avoid warnings about real*8 etc', 4, 'compilers')
    self.popLanguage()
    return

  def mangleFortranFunction(self, name):
    if self.fortranMangling == 'underscore':
      if self.fortranManglingDoubleUnderscore and name.find('_') >= 0:
        return name+'__'
      else:
        return name+'_'
    elif self.fortranMangling == 'unchanged':
      return name
    elif self.fortranMangling == 'caps':
      return name.upper()
    elif self.fortranMangling == 'stdcall':
      return name.upper()
    raise RuntimeError('Unknown Fortran name mangling: '+self.fortranMangling)

  def testMangling(self, cfunc, ffunc, clanguage = 'C', extraObjs = []):
    '''Test a certain name mangling'''
    cobj = 'confc.o'
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
    '''Checks Fortran name mangling, and defines HAVE_FORTRAN_UNDERSCORE, HAVE_FORTRAN_NOUNDERSCORE, HAVE_FORTRAN_CAPS, or HAVE_FORTRAN_STDCALL'''
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
      self.framework.log.write('Testing Fortran mangling type '+mangler+' with code '+cfunc)
      if self.testMangling(cfunc, ffunc):
        self.fortranMangling = mangler
        break
    else:
      raise RuntimeError('Unknown Fortran name mangling')
    self.logPrint('Fortran name mangling is '+self.fortranMangling, 4, 'compilers')
    if self.fortranMangling == 'underscore':
      self.addDefine('HAVE_FORTRAN_UNDERSCORE', 1)
    elif self.fortranMangling == 'unchanged':
      self.addDefine('HAVE_FORTRAN_NOUNDERSCORE', 1)
    elif self.fortranMangling == 'caps':
      self.addDefine('HAVE_FORTRAN_CAPS', 1)
    elif self.fortranMangling == 'stdcall':
      self.addDefine('HAVE_FORTRAN_STDCALL', 1)
      self.addDefine('STDCALL', '__stdcall')
      self.addDefine('HAVE_FORTRAN_CAPS', 1)
      self.addDefine('HAVE_FORTRAN_MIXED_STR_ARG', 1)
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

  def checkFortranPreprocessor(self):
    '''Determine if Fortran handles preprocessing properly'''
    self.setCompilers.pushLanguage('FC')
    # Does Fortran compiler need special flag for using CPP
    for flag in ['', '-cpp', '-xpp=cpp', '-F', '-Cpp', '-fpp', '-fpp:-m']:
      try:
        flagsArg = self.setCompilers.getCompilerFlagsArg()
        oldFlags = getattr(self.setCompilers, flagsArg)
        self.setCompilers.addCompilerFlag(flag, body = '#define dummy \n           dummy\n#ifndef dummy\n       fooey\n#endif')
        setattr(self.setCompilers, flagsArg, oldFlags+' '+flag)
        self.fortranPreprocess = 1
        self.setCompilers.popLanguage()
        self.logPrint('Fortran uses CPP preprocessor', 3, 'compilers')
        return
      except RuntimeError:
        setattr(self.setCompilers, flagsArg, oldFlags)
    self.setCompilers.popLanguage()
    self.fortranPreprocess = 0
    self.logPrint('Fortran does NOT use CPP preprocessor', 3, 'compilers')
    return

  def checkFortranDefineCompilerOption(self):
    '''Check if -WF,-Dfoobar or -Dfoobar is the compiler option to define a macro'''
    self.FortranDefineCompilerOption = 0
    if not self.fortranPreprocess:
      return
    self.setCompilers.pushLanguage('FC')
    for flag in ['-D', '-WF,-D']:
      if self.setCompilers.checkCompilerFlag(flag+'Testing', body = '#define dummy \n           dummy\n#ifndef Testing\n       fooey\n#endif'):
        self.FortranDefineCompilerOption = flag
        self.setCompilers.popLanguage()
        self.logPrint('Fortran uses '+flag+' for defining macro', 3, 'compilers')
        return
    self.setCompilers.popLanguage()
    self.logPrint('Fortran does not support defining macro', 3, 'compilers')
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
    self.pushLanguage('FC')
    oldFlags = self.setCompilers.LDFLAGS
    if config.setCompilers.Configure.isNAG(self.getCompiler()):
      self.setCompilers.LDFLAGS += ' --verbose'
    else:
      self.setCompilers.LDFLAGS += ' -v'
    (output, returnCode) = self.outputLink('', '')
    self.setCompilers.LDFLAGS = oldFlags
    self.popLanguage()

    # replace \CR that ifc puts in each line of output
    output = output.replace('\\\n', '')

    if output.lower().find('absoft') >= 0:
      loc = output.find(' -lf90math')
      if loc == -1: loc = output.find(' -lf77math')
      if loc >= -1:
        output = output[0:loc]+' -lU77 -lV77 '+output[loc:]

    # PGI/Windows: to properly resolve symbols, we need to list the fortran runtime libraries before -lpgf90
    if output.find(' -lpgf90') >= 0 and output.find(' -lkernel32') >= 0:
      loc  = output.find(' -lpgf90')
      loc2 = output.find(' -lpgf90rtl -lpgftnrtl')
      if loc2 >= -1:
        output = output[0:loc] + ' -lpgf90rtl -lpgftnrtl' + output[loc:]
    elif output.find(' -lpgf90rtl -lpgftnrtl') >= 0:
      # somehow doing this hacky thing appears to get rid of error with undefined __hpf_exit
      self.logPrint('Adding -lpgftnrtl before -lpgf90rtl in library list')
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
    fmainlibs = []
    lflags  = []
    rpathflags = []
    try:
      while 1:
        arg = argIter.next()
        self.logPrint( 'Checking arg '+arg, 4, 'compilers')
        # Intel compiler sometimes puts " " around an option like "-lsomething"
        if arg.startswith('"') and arg.endswith('"'):
          arg = arg[1:-1]
        # Intel also puts several options together inside a " " so the last one
        # has a stray " at the end
        if arg.endswith('"') and arg[:-1].find('"') == -1:
          arg = arg[:-1]

        # Check for full library name
        m = re.match(r'^/.*\.a$', arg)
        if m:
          if not arg in lflags:
            lflags.append(arg)
            self.logPrint('Found full library spec: '+arg, 4, 'compilers')
            flibs.append(arg)
          continue
        # prevent false positives for include with pathscalr
        if re.match(r'^-INTERNAL.*$', arg): continue
        # Check for special include argument
        # AIX does this for MPI and perhaps other things
        m = re.match(r'^-I.*$', arg)
        if m:
          inc = arg.replace('-I','')
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
          continue
        # Check for system libraries
        m = re.match(r'^-l(ang.*|crt[0-9].o|crtbegin.o|c|gcc|cygwin|crt[0-9].[0-9][0-9].[0-9].o)$', arg)
        if m: continue
        # Check for canonical library argument
        m = re.match(r'^-[lL]$', arg)
        if m:
          lib = arg+argIter.next()
          self.logPrint('Found canonical library: '+lib, 4, 'compilers')
          flibs.append(lib)
          continue
        # Check for special library arguments
        m = re.match(r'^-l.*$', arg)
        if m:
          # HP Fortran prints these libraries in a very strange way
          if arg == '-l:libU77.a':  arg = '-lU77'
          if arg == '-l:libF90.a':  arg = '-lF90'
          if arg == '-l:libIO77.a': arg = '-lIO77'                      
          if not arg in lflags:
            
            #TODO: if arg == '-lkernel32' and host_os.startswith('cygwin'):
            if arg == '-lkernel32':
              continue
            elif arg == '-lm':
              pass
            elif arg == '-lgfortranbegin':
              fmainlibs.append(arg)
              continue
            elif arg == '-lfrtbegin' and not config.setCompilers.Configure.isCygwin():
              fmainlibs.append(arg)
              continue
            else:
              lflags.append(arg)
            self.logPrint('Found library: '+arg, 4, 'compilers')
            flibs.append(arg)
          continue
        m = re.match(r'^-L.*$', arg)
        if m:
          arg = '-L'+os.path.abspath(arg[2:])
          if arg in ['-L/usr/lib','-L/lib','-L/usr/lib64','-L/lib64']: continue          
          if not arg in lflags:
            
            #TODO: if arg == '-lkernel32' and host_os.startswith('cygwin'):
            lflags.append(arg)
            self.logPrint('Found library directory: '+arg, 4, 'compilers')
            flibs.append(arg)
          continue
        # Check for '-rpath /sharedlibpath/ or -R /sharedlibpath/'
        if arg == '-rpath' or arg == '-R':
          lib = argIter.next()
          if lib == '\\': lib = argIter.next()
          if lib.startswith('"') and lib.endswith('"') and lib.find(' ') == -1: lib = lib[1:-1]
          lib = os.path.abspath(lib)
          if lib in ['/usr/lib','/lib','/usr/lib64','/lib64']: continue
          if not lib in rpathflags:
            rpathflags.append(lib)
            self.logPrint('Found '+arg+' library: '+lib, 4, 'compilers')
            flibs.append(self.setCompilers.CSharedLinkerFlag+lib)
          continue
        # Check for '-R/sharedlibpath/'
        m = re.match(r'^-R.*$', arg)
        if m:
          lib = os.path.abspath(arg[2:])
          if not lib in rpathflags:
            rpathflags.append(lib)
            self.logPrint('Found -R library: '+lib, 4, 'compilers')
            flibs.append(self.setCompilers.CSharedLinkerFlag+lib)
          continue
        if arg.startswith('-zallextract') or arg.startswith('-zdefaultextract') or arg.startswith('-zweakextract'):
          self.framework.log.write( 'Found Solaris -z option: '+arg+'\n')
          flibs.append(arg)
          continue
        # Check for ???
        # Should probably try to ensure unique directory options here too.
        # This probably only applies to Solaris systems, and then will only
        # work with gcc...
        if arg == '-Y':
          libs = argIter.next()
          if libs.startswith('"') and libs.endswith('"'):
            libs = libs[1:-1]
          for lib in libs.split(':'):
            #solaris gnu g77 has this extra P, here, not sure why it means
            if lib.startswith('P,'):lib = lib[2:]
            self.logPrint('Handling -Y option: '+lib, 4, 'compilers')
            lib1 = '-L'+os.path.abspath(lib)
            if lib1 in ['-L/usr/lib','-L/lib','-L/usr/lib64','-L/lib64']: continue
            flibs.append(lib1)
          continue
        if arg.startswith('COMPILER_PATH=') or arg.startswith('LIBRARY_PATH='):
          self.logPrint('Skipping arg '+arg, 4, 'compilers')
          continue
        # HPUX lists a bunch of library directories seperated by :
        if arg.find(':') >=0:
          founddir = 0
          for l in arg.split(':'):
            if os.path.isdir(l):
              lib1 = '-L'+os.path.abspath(l)
              if lib1 in ['-L/usr/lib','-L/lib','-L/usr/lib64','-L/lib64']: continue
              if not arg in lflags:              
                flibs.append(lib1)
                lflags.append(lib1)
                self.logPrint('Handling HPUX list of directories: '+l, 4, 'compilers')
                founddir = 1
          if founddir:
            continue
        if arg.find('quickfit.o')>=0:
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
        self.framework.log.write('Detected Apple Mac Fink libraries')
        appleLib = 'libcc_dynamic.so'
        if self.libraries.check(appleLib, 'foo'):
          self.flibs.append(self.libraries.getLibArgument(appleLib))
          self.framework.log.write('Adding '+self.libraries.getLibArgument(appleLib)+' so that Fortran can work with C++')
        break

    self.logPrint('Libraries needed to link Fortran code with the C linker: '+str(self.flibs), 3, 'compilers')
    self.logPrint('Libraries needed to link Fortran main with the C linker: '+str(self.fmainlibs), 3, 'compilers')
    # check that these monster libraries can be used from C
    self.logPrint('Check that Fortran libraries can be used from C', 4, 'compilers')
    oldLibs = self.setCompilers.LIBS
    self.setCompilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])+' '+self.setCompilers.LIBS
    try:
      self.setCompilers.checkCompiler('C')
    except RuntimeError, e:
      self.logPrint('Fortran libraries cannot directly be used from C, try without -lcrt2.o', 4, 'compilers')
      self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
      # try removing this one
      if '-lcrt2.o' in self.flibs: self.flibs.remove('-lcrt2.o')
      self.setCompilers.LIBS = oldLibs+' '+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
      try:
        self.setCompilers.checkCompiler('C')
      except RuntimeError, e:
        self.logPrint('Fortran libraries still cannot directly be used from C, try without pgi.ld files', 4, 'compilers')
        self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
        tmpflibs = self.flibs
        for lib in tmpflibs:
          if lib.find('pgi.ld')>=0:
            self.flibs.remove(lib)
        self.setCompilers.LIBS = oldLibs+' '+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
        try:
          self.setCompilers.checkCompiler('C')
        except:
          self.logPrint(str(e), 4, 'compilers')
          raise RuntimeError('Fortran libraries cannot be used with C compiler')

    # check if Intel library exists (that is not linked by default but has iargc_ in it :-(
    self.logPrint('Check for Intel PEPCF90 library', 4, 'compilers')
    self.setCompilers.LIBS = oldLibs+' -lPEPCF90 '+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
    try:
      self.setCompilers.checkCompiler('C')
      self.flibs = [' -lPEPCF90']+self.flibs
      self.logPrint('Intel PEPCF90 library exists', 4, 'compilers')
    except RuntimeError, e:
      self.logPrint('Intel PEPCF90 library does not exist', 4, 'compilers')
      self.setCompilers.LIBS = oldLibs+' '+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])

    # check these monster libraries work from C++
    if hasattr(self.setCompilers, 'CXX'):
      self.logPrint('Check that Fortran libraries can be used from C++', 4, 'compilers')
      self.setCompilers.LIBS = ' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])+' '+oldLibs
      try:
        self.setCompilers.checkCompiler('Cxx')
        self.logPrint('Fortran libraries can be used from C++', 4, 'compilers')
      except RuntimeError, e:
        self.logPrint(str(e), 4, 'compilers')
        # try removing this one causes grief with gnu g++ and Intel Fortran
        if '-lintrins' in self.flibs: self.flibs.remove('-lintrins')
        self.setCompilers.LIBS = oldLibs+' '+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
        try:
          self.setCompilers.checkCompiler('Cxx')
        except RuntimeError, e:
          self.logPrint(str(e), 4, 'compilers')
          if str(e).find('INTELf90_dclock') >= 0:
            self.logPrint('Intel 7.1 Fortran compiler cannot be used with g++ 3.2!', 2, 'compilers')
          raise RuntimeError('Fortran libraries cannot be used with C++ compiler.\n Run with --with-fc=0 or --with-cxx=0')

    self.setCompilers.LIBS = oldLibs
    return

  def checkFortranLinkingCxx(self):
    '''Check that Fortran can be linked against C++'''
    link = 0
    cinc, cfunc, ffunc = self.manglerFuncs[self.fortranMangling]
    cinc = 'extern "C" '+cinc+'\n'

    cxxCode = 'void foo(void){'+self.mangleFortranFunction('d1chk')+'();}'
    cxxobj = 'cxxobj.o'
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
      raise RuntimeError('Fortran could not successfully link C++ objects')
    return

  def checkFortran90(self):
    '''Determine whether the Fortran compiler handles F90'''
    self.pushLanguage('FC')
    if self.checkLink(body = '      INTEGER, PARAMETER :: int = SELECTED_INT_KIND(8)\n      INTEGER (KIND=int) :: ierr\n\n      ierr = 1'):
      self.addDefine('USING_F90', 1)
      self.fortranIsF90 = 1
      self.logPrint('Fortran compiler supports F90')
    else:
      self.fortranIsF90 = 0
      self.logPrint('Fortran compiler does not support F90')
    self.popLanguage()
    return

  def checkFortran90Array(self):
    '''Check for F90 array interfaces'''
    if not self.fortranIsF90:
      self.logPrint('Not a Fortran90 compiler - hence skipping f90-array test')
      return
    # do an apporximate test when batch mode is used, as we cannot run the proper test..
    if self.framework.argDB['with-batch']:
      if config.setCompilers.Configure.isPGI(self.setCompilers.FC):
        self.addDefine('HAVE_F90_2PTR_ARG', 1)
        self.logPrint('PGI F90 compiler detected & using --with-batch', 3, 'compilers')
      return
    # do not check on windows - as it pops up the annoying debugger
    if config.setCompilers.Configure.isCygwin():
      self.logPrint('Cygwin detected: ignoring HAVE_F90_2PTR_ARG test')
      return

    # Compile the C test object
    cinc  = '#include<stdio.h>\n#include <stdlib.h>\n'
    ccode = 'void '+self.mangleFortranFunction('f90arraytest')+'''(void* a1, void* a2,void* a3, void* i)
{
  printf("arrays [%p %p %p]\\n",a1,a2,a3);
  fflush(stdout);
  return;
}
''' + 'void '+self.mangleFortranFunction('f90ptrtest')+'''(void* a1, void* a2,void* a3, void* i, void* p1 ,void* p2, void* p3)
{
  printf("arrays [%p %p %p]\\n",a1,a2,a3);
  if ((p1 == p3) && (p1 != p2)) {
    printf("pointers match! [%p %p] [%p]\\n",p1,p3,p2);
    fflush(stdout);
  } else {
    printf("pointers do not match! [%p %p] [%p]\\n",p1,p3,p2);
    fflush(stdout);
    abort();
  }
  return;
}\n'''
    cobj = 'fooobj.o'
    self.pushLanguage('C')
    if not self.checkCompile(cinc+ccode, None, cleanup = 0):
      self.logPrint('Cannot compile C function: f90ptrtest', 3, 'compilers')
      raise RuntimeError('Could not check Fortran pointer arguments')
    if not os.path.isfile(self.compilerObj):
      self.logPrint('Cannot locate object file: '+os.path.abspath(self.compilerObj), 3, 'compilers')
      raise RuntimeError('Could not check Fortran pointer arguments')
    os.rename(self.compilerObj, cobj)
    self.popLanguage()
    # Link the test object against a Fortran driver
    self.pushLanguage('FC')
    oldLIBS = self.setCompilers.LIBS
    self.setCompilers.LIBS = cobj+' '+self.setCompilers.LIBS
    fcode = '''\
      Interface
         Subroutine f90ptrtest(p1,p2,p3,i)
         integer, pointer :: p1(:,:)
         integer, pointer :: p2(:,:)
         integer, pointer :: p3(:,:)
         integer i
         End Subroutine
      End Interface

      integer, pointer :: ptr1(:,:),ptr2(:,:)
      integer, target  :: array(6:8,9:21)
      integer  in

      in   = 25
      ptr1 => array
      ptr2 => array

      call f90arraytest(ptr1,ptr2,ptr1,in)
      call f90ptrtest(ptr1,ptr2,ptr1,in)\n'''

    found = self.checkRun(None, fcode, defaultArg = 'f90-2ptr-arg')
    self.setCompilers.LIBS = oldLIBS
    self.popLanguage()
    # Cleanup
    if os.path.isfile(cobj):
      os.remove(cobj)
    if found:
      self.addDefine('HAVE_F90_2PTR_ARG', 1)
      self.logPrint('F90 compiler uses two arguments for array pointers', 3, 'compilers')
    else:
      self.logPrint('F90 uses a single argument for array pointers', 3, 'compilers')
    return

  def checkFortranModuleInclude(self):
    '''Figures out what flag is used to specify the include path for Fortran modules'''
    self.setCompilers.fortranModuleIncludeFlag = None
    if not self.fortranIsF90:
      self.logPrint('Not a Fortran90 compiler - hence skipping module include test')
      return
    found   = False
    testdir = 'confdir'
    modobj  = 'configtest.o'
    modcode = '''\
      module configtest
      integer testint
      parameter (testint = 42)
      end module configtest\n'''
    # Compile the Fortran test module
    self.pushLanguage('FC')
    if not self.checkCompile(modcode, None, cleanup = 0):
      self.logPrint('Cannot compile Fortran module', 3, 'compilers')
      self.popLanguage()
      raise RuntimeError('Cannot determine Fortran module include flag')
    if not os.path.isfile(self.compilerObj):
      self.logPrint('Cannot locate object file: '+os.path.abspath(self.compilerObj), 3, 'compilers')
      self.popLanguage()
      raise RuntimeError('Cannot determine Fortran module include flag')
    if not os.path.isdir(testdir):
      os.mkdir(testdir)
    os.rename(self.compilerObj, modobj)
    if os.path.isfile('configtest.mod'):
      modname = 'configtest.mod'
    elif os.path.isfile('CONFIGTEST.mod'):
      modname = 'CONFIGTEST.mod'
    else:
      raise RuntimeError('Fortran module was not created during the compile. configtest.mod/CONFIGTEST.mod not found')
    os.rename(modname, os.path.join(testdir, modname))
    fcode = '''\
      use configtest

      write(*,*) testint\n'''
    self.pushLanguage('FC')
    oldFLAGS = self.setCompilers.FFLAGS
    oldLIBS  = self.setCompilers.LIBS
    for flag in ['-I', '-p', '-M']:
      self.setCompilers.FFLAGS = flag+testdir+' '+self.setCompilers.FFLAGS
      self.setCompilers.LIBS   = modobj+' '+self.setCompilers.LIBS
      if not self.checkLink(None, fcode):
        self.logPrint('Fortran module include flag '+flag+' failed', 3, 'compilers')
      else:
        self.logPrint('Fortran module include flag '+flag+' found', 3, 'compilers')        
        self.setCompilers.fortranModuleIncludeFlag = flag
        found = 1
      self.setCompilers.LIBS   = oldLIBS
      self.setCompilers.FFLAGS = oldFLAGS
      if found: break
    self.popLanguage()
    if os.path.isfile(modobj):
      os.remove(modobj)
    os.remove(os.path.join(testdir, modname))
    os.rmdir(testdir)
    if not found:
      raise RuntimeError('Cannot determine Fortran module include flag')
    return

  def checkFortranModuleOutput(self):
    '''Figures out what flag is used to specify the include path for Fortran modules'''
    self.setCompilers.fortranModuleOutputFlag = None
    if not self.fortranIsF90:
      self.logPrint('Not a Fortran90 compiler - hence skipping module include test')
      return
    found   = False
    testdir = 'confdir'
    modobj  = 'configtest.o'
    modcode = '''\
      module configtest
      integer testint
      parameter (testint = 42)
      end module configtest\n'''
    modname = None
    # Compile the Fortran test module
    if not os.path.isdir(testdir):
      os.mkdir(testdir)
    self.pushLanguage('FC')
    oldFLAGS = self.setCompilers.FFLAGS
    oldLIBS  = self.setCompilers.LIBS
    for flag in ['-module ', '-module:', '-fmod=', '-J', '-M', '-p', '-moddir=']:
      self.setCompilers.FFLAGS = flag+testdir+' '+self.setCompilers.FFLAGS
      self.setCompilers.LIBS   = modobj+' '+self.setCompilers.LIBS
      if not self.checkCompile(modcode, None, cleanup = 0):
        self.logPrint('Fortran module output flag '+flag+' compile failed', 3, 'compilers')
      elif os.path.isfile(os.path.join(testdir, 'configtest.mod')) or os.path.isfile(os.path.join(testdir, 'CONFIGTEST.mod')):
        if os.path.isfile(os.path.join(testdir, 'configtest.mod')): modname = 'configtest.mod'
        if os.path.isfile(os.path.join(testdir, 'CONFIGTEST.mod')): modname = 'CONFIGTEST.mod'
        self.logPrint('Fortran module output flag '+flag+' found', 3, 'compilers')        
        self.setCompilers.fortranModuleOutputFlag = flag
        found = 1
      else:
        self.logPrint('Fortran module output flag '+flag+' failed', 3, 'compilers')
      self.setCompilers.LIBS   = oldLIBS
      self.setCompilers.FFLAGS = oldFLAGS
      if found: break
    self.popLanguage()
    if modname: os.remove(os.path.join(testdir, modname))
    os.rmdir(testdir)
    # Flag not used by PETSc - do do not flag a runtime error
    #if not found:
    #  raise RuntimeError('Cannot determine Fortran module output flag')
    return

  def configure(self):
    import config.setCompilers
    if hasattr(self.setCompilers, 'CC'):
      self.isGCC = config.setCompilers.Configure.isGNU(self.setCompilers.CC)
      self.executeTest(self.checkRestrict,['C'])
      self.executeTest(self.checkCFormatting)
      self.executeTest(self.checkCStaticInline)
      self.executeTest(self.checkDynamicLoadFlag)
      if self.framework.argDB['with-clib-autodetect']:
        self.executeTest(self.checkCLibraries)      
    else:
      self.isGCC = 0
    if hasattr(self.setCompilers, 'CXX'):
      self.isGCXX = config.setCompilers.Configure.isGNU(self.setCompilers.CXX)
      self.executeTest(self.checkRestrict,['Cxx'])
      self.executeTest(self.checkCxxNamespace)
      self.executeTest(self.checkCxxOptionalExtensions)
      self.executeTest(self.checkCxxStaticInline)
      if self.framework.argDB['with-cxxlib-autodetect']:
        self.executeTest(self.checkCxxLibraries)
    else:
      self.isGCXX = 0
    if hasattr(self.setCompilers, 'FC'):
      self.executeTest(self.checkFortranTypeSizes)
      self.executeTest(self.checkFortranNameMangling)
      self.executeTest(self.checkFortranNameManglingDouble)
      self.executeTest(self.checkFortranPreprocessor)
      self.executeTest(self.checkFortranDefineCompilerOption)
      if self.framework.argDB['with-fortranlib-autodetect']:
        self.executeTest(self.checkFortranLibraries)
      if hasattr(self.setCompilers, 'CXX'):
        self.executeTest(self.checkFortranLinkingCxx)
      self.executeTest(self.checkFortran90)
      self.executeTest(self.checkFortran90Array)
      self.executeTest(self.checkFortranModuleInclude)
      self.executeTest(self.checkFortranModuleOutput)
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
