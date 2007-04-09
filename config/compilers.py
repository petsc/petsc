import config.base

import re
import os

try:
  import sets
except ImportError:
  import config.setsBackport as sets

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

    help.addArgument('Compilers', '-with-f90-interface=<type>', nargs.Arg(None, None, 'Specify  compiler type for eg: intel8,solaris,rs6000,IRIX,win32,absoft,t3e,alpha,cray_x1,hpux,lahey'))
    return

  def getDispatchNames(self):
    '''Return all the attributes which are dispatched from config.setCompilers'''
    names = sets.Set()
    errors = {}
    names.update(['CC', 'CPP', 'CXX', 'CXXCPP', 'FC'])
    errors['CC'] = 'No C compiler found.'
    errors['CPP'] = 'No C preprocessor found.'
    errors['CXX'] = 'No C++ compiler found.'
    errors['CXXCPP'] = 'No C++ preprocessor found.'
    errors['FC'] = 'No Fortran compiler found.'
    names.update(['AR', 'RANLIB', 'LD_SHARED', 'dynamicLinker'])
    errors['AR'] = 'No archiver found.'
    errors['RANLIB'] = 'No ranlib found.'
    errors['LD_SHARED'] = 'No shared linker found.'
    errors['dynamicLinker'] = 'No dynamic linker found.'
    for language in ['C', 'Cxx', 'FC']:
      self.pushLanguage(language)
      names.update([self.getCompilerFlagsName(language), self.getCompilerFlagsName(language, 1), self.getLinkerFlagsName(language)])
      errors[self.getCompilerFlagsName(language)] = 'No '+language+' compiler flags found.'
      errors[self.getLinkerFlagsName(language)] = 'No '+language+' linker flags found.'
      self.popLanguage()
    names.update(['CPPFLAGS'])
    errors['CPPFLAGS'] = 'No preprocessor flags found.'
    names.update(['AR_FLAGS', 'AR_LIB_SUFFIX'])
    errors['AR_FLAGS'] = 'No archiver flags found.'
    errors['AR_LIB_SUFFIX'] = 'No static library suffix found.'
    names.update(['LIBS'])
    errors['LIBS'] = 'No extra libraries found.'
    return names, errors

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers', self)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.libraries = framework.require('config.libraries', None)
    self.dispatchNames, self.dispatchErrors = self.getDispatchNames()
    return

  def __getattr__(self, name):
    if 'dispatchNames' in self.__dict__:
      if name in self.dispatchNames:
        if not hasattr(self.setCompilers, name):
          raise MissingProcessor(errors[name])
        return getattr(self.setCompilers, name)
      if name in ['executableFlags', 'sharedLibraryFlags', 'dynamicLibraryFlags']:
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

    # The easiest thing to do for xlc output is to replace all the commas
    # with spaces.  Try to only do that if the output is really from xlc,
    # since doing that causes problems on other systems.
    if output.find('XL_CONFIG') >= 0:
      output = output.replace(',', ' ')
      
    # Parse output
    argIter = iter(output.split())
    clibs = []
    lflags  = []
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
        m = re.match(r'^-l(ang.*|crt0.o|crt1.o|crt2.o|crtbegin.o|c|gcc)$', arg)
        if m: continue
        # Check for special library arguments
        m = re.match(r'^-[lLR].*$', arg)
        if m:
          if not arg in lflags:
            if arg == '-lkernel32':
              continue
            elif arg == '-lm':
              continue
            else:
              lflags.append(arg)
            self.logPrint('Found library or library directory: '+arg, 4, 'compilers')
            clibs.append(arg)
          continue
        if arg == '-rpath':
          lib = argIter.next()
          self.logPrint('Found -rpath library: '+lib, 4, 'compilers')
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
        self.logPrint('C libraries cannot directly be used from Fortran', 4, 'compilers')
        self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
        self.setCompilers.LIBS = oldLibs
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
    '''Checks that dlopen() takes RTLD_GLOBAL, and defines PETSC_HAVE_RTLD_GLOBAL if it does'''
    if self.setCompilers.dynamicLibraries:
      if self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);\n'):
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

    # The easiest thing to do for xlc output is to replace all the commas
    # with spaces.  Try to only do that if the output is really from xlc,
    # since doing that causes problems on other systems.
    if output.find('XL_CONFIG') >= 0:
      output = output.replace(',', ' ')
      
    # Parse output
    argIter = iter(output.split())
    cxxlibs = []
    lflags  = []
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
        m = re.match(r'^-l(ang.*|crt0.o|crt1.o|crt2.o|crtbegin.o|c|gcc)$', arg)
        if m: continue
        # Check for special library arguments
        m = re.match(r'^-[lLR].*$', arg)
        if m:
          if not arg in lflags:
            if arg == '-lkernel32':
              continue
            elif arg == '-lm':
              continue
            else:
              lflags.append(arg)
            self.logPrint('Found library or library directory: '+arg, 4, 'compilers')
            if arg in self.clibs:
              self.logPrint('Library already in C list so skipping in C++')
            else:
              cxxlibs.append(arg)
          continue
        if arg == '-rpath':
          lib = argIter.next()
          self.logPrint('Found -rpath library: '+lib, 4, 'compilers')
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
    elif self.fortranMangling == 'capitalize':
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
    self.setCompilers.LIBS = cobj+' '+self.setCompilers.LIBS
    if extraObjs:
      self.setCompilers.LIBS = ' '.join(extraObjs)+' '+self.setCompilers.LIBS
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
                         'capitalize': ('void D1CHK(void);', 'void D1CHK(void){return;}\n', '       call d1chk()\n'),
                         'stdcall': ('void __stdcall D1CHK(void);', 'void __stdcall D1CHK(void){return;}\n', '       call d1chk()\n'),
                         'double': ('void d1_chk__(void)', 'void d1_chk__(void){return;}\n', '       call d1_chk()\n')}
    #some compilers silently ignore '__stdcall' directive, so do stdcall test last
    # double test is not done here, so its not listed
    key_list = ['underscore','unchanged','capitalize','stdcall']
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
    elif self.fortranMangling == 'capitalize':
      self.addDefine('HAVE_FORTRAN_CAPS', 1)
    elif self.fortranMangling == 'stdcall':
      self.addDefine('HAVE_FORTRAN_STDCALL', 1)
      self.addDefine('STDCALL', '__stdcall')
      self.addDefine('HAVE_FORTRAN_CAPS', 1)
      self.addDefine('HAVE_FORTRAN_MIXED_STR_ARG', 1)
    return

  def checkFortranNameManglingDouble(self):
    '''Checks if symbols containing and underscore append and extra underscore, and defines HAVE_FORTRAN_UNDERSCORE_UNDERSCORE if necessary'''
    if self.testMangling(self.manglerFuncs['double'][1], self.manglerFuncs['double'][2]):
      self.logPrint('Fortran appends and extra underscore to names containing underscores', 4, 'compilers')
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
        setattr(self.setCompilers, flagsArg, oldFlags+' '+'-DPTesting')
        self.setCompilers.addCompilerFlag(flag, body = '#define dummy \n           dummy\n#ifndef PTesting\n       fooey\n#endif')
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
        m = re.match(r'^-l(ang.*|crt0.o|crt1.o|crt2.o|crtbegin.o|c|gcc)$', arg)
        if m: continue
        # Check for canonical library argument
        m = re.match(r'^-[lLR]$', arg)
        if m:
          lib = arg+argIter.next()
          self.logPrint('Found canonical library: '+lib, 4, 'compilers')
          flibs.append(lib)
          continue
        # Check for special library arguments
        m = re.match(r'^-[lLR].*$', arg)
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
            elif arg == '-lfrtbegin' and not config.setCompilers.Configure.isCygwin():
              fmainlibs.append(arg)
              continue
            else:
              lflags.append(arg)
            self.logPrint('Found library or library directory: '+arg, 4, 'compilers')
            flibs.append(arg)
          continue
        if arg == '-rpath':
          lib = argIter.next()
          self.logPrint('Found -rpath library: '+lib, 4, 'compilers')
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
          for lib in argIter.next().split(':'):
            #solaris gnu g77 has this extra P, here, not sure why it means
            if lib.startswith('P,'):lib = lib[2:]
            self.logPrint('Handling -Y option: '+lib, 4, 'compilers')
            flibs.append('-L'+lib)
          continue
        # HPUX lists a bunch of library directories seperated by :
        if arg.find(':') >=0:
          founddir = 0
          for l in arg.split(':'):
            if os.path.isdir(l):
              flibs.append('-L'+l)
              self.logPrint('Handling HPUX list of directories: '+l, 4, 'compilers')
              founddir = 1
          if founddir:
            continue
        if arg.find('quickfit.o')>=0:
          flibs.append(arg)
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

  def stripquotes(self,str):
    if str[0] =='"': str = str[1:]
    if str[-1] =='"': str = str[:-1]
    return str

  def getFortran90SourceGuesses(self):
    f90Guess = None
    if config.setCompilers.Configure.isG95(self.setCompilers.FC):
      f90Guess = 'g95'
    elif config.setCompilers.Configure.isIntel(self.setCompilers.FC):
      f90Guess = 'intel8'
    elif config.setCompilers.Configure.isCompaqF90(self.setCompilers.FC):
      f90Guess = 'win32'
    elif config.setCompilers.Configure.isNAG(self.setCompilers.FC):
      f90Guess = 'nag'
    elif config.setCompilers.Configure.isNAG(self.setCompilers.FC):
      f90Guess = 'cray_x1'
    elif config.setCompilers.Configure.isSUN(self.setCompilers.FC):
      f90Guess = 'solaris'
    elif self.setCompilers.vendor:
      if self.setCompilers.vendor == 'absoft':
        f90Guess = 'absoft'
      elif self.setCompilers.vendor == 'cray':
        f90Guess = 't3e'
        #f90Guess = 'cray_x1'
      elif self.setCompilers.vendor == 'dec':
        f90Guess = 'alpha'
      elif self.setCompilers.vendor == 'hp':
        f90Guess = 'hpux'
      elif self.setCompilers.vendor == 'ibm':
        f90Guess = 'rs6000'
      elif self.setCompilers.vendor == 'intel':
        #headerGuess = 'f90_intel.h'
        f90Guess = 'intel8'
      elif self.setCompilers.vendor == 'nag':
        f90Guess = 'nag'
      elif self.setCompilers.vendor == 'lahaye':
        f90Guess = 'lahaye'        
##    This interface is not finished
##      elif self.setCompilers.vendor == 'portland':
##        f90Guess = 'pgi'
      elif self.setCompilers.vendor == 'sgi':
        f90Guess = 'IRIX'
      elif self.setCompilers.vendor == 'solaris':
        f90Guess = 'solaris'
    return f90Guess

  def checkFortran90Interface(self):
    '''Check for custom F90 interfaces, such as that provided by PETSc'''
    if not self.fortranIsF90:
      self.logPrint('Not a Fortran90 compiler - hence skipping f90-interface test')
      return
    self.f90Guess = self.getFortran90SourceGuesses()
    if 'with-f90-interface' in self.framework.argDB:
      self.f90Guess = self.stripquotes(self.framework.argDB['with-f90-interface'])

    if not self.f90Guess :
      return

    headerPath = os.path.join('src', 'sys','f90', 'f90_'+self.f90Guess+'.h')
    sourcePath = os.path.join('src', 'sys','f90','f90_'+self.f90Guess+'.c')

    if os.path.isfile(headerPath):
      self.f90HeaderPath = headerPath
    else:
      self.logPrint('Invalid F90 header: '+str(headerPath), 2, 'compilers')
      
    if os.path.isfile(sourcePath):
      self.f90SourcePath = sourcePath
    else:
      self.logPrint('Invalid F90 source: '+str(sourcePath), 2, 'compilers')
      
    if hasattr(self, 'f90HeaderPath') and hasattr(self, 'f90SourcePath'):
      self.addDefine('HAVE_F90_H', '"'+self.f90HeaderPath+'"')
      self.addDefine('HAVE_F90_C', '"'+self.f90SourcePath+'"')
    else:
      raise RuntimeError('Perhaps incorrect with-f90-interface specified. Could not confiure f90 interface for : '+self.f90Guess)      
    return

  def configure(self):
    import config.setCompilers
    if hasattr(self.setCompilers, 'CC'):
      self.isGCC = config.setCompilers.Configure.isGNU(self.setCompilers.CC)
      self.executeTest(self.checkRestrict,['C'])
      self.executeTest(self.checkCFormatting)
      self.executeTest(self.checkCStaticInline)
      self.executeTest(self.checkDynamicLoadFlag)
      self.executeTest(self.checkCLibraries)      
    else:
      self.isGCC = 0
    if hasattr(self.setCompilers, 'CXX'):
      self.isGCXX = config.setCompilers.Configure.isGNU(self.setCompilers.CXX)
      self.executeTest(self.checkRestrict,['Cxx'])
      self.executeTest(self.checkCxxNamespace)
      self.executeTest(self.checkCxxOptionalExtensions)
      self.executeTest(self.checkCxxStaticInline)
      self.executeTest(self.checkCxxLibraries)
    else:
      self.isGCXX = 0
    if hasattr(self.setCompilers, 'FC'):
      self.executeTest(self.checkFortranTypeSizes)
      self.executeTest(self.checkFortranNameMangling)
      self.executeTest(self.checkFortranNameManglingDouble)
      self.executeTest(self.checkFortranPreprocessor)
      self.executeTest(self.checkFortranLibraries)
      if hasattr(self.setCompilers, 'CXX'):
        self.executeTest(self.checkFortranLinkingCxx)
      self.executeTest(self.checkFortran90)
      self.executeTest(self.checkFortran90Interface)
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
