import config.base

import re
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.setCompilers = self.framework.require('config.setCompilers', self)
    self.libraries = self.framework.require('config.libraries', self)
    self.fortranMangling = 'unchanged'
    self.flibs = []
    return

  def __str__(self):
    return ''

  def setupHelp(self, help):
    import nargs

    help.addArgument('Compilers', '-with-f90-header=<file>', nargs.Arg(None, None, 'Specify the C header for the F90 interface, e.g. f90impl/f90_intel.h'))
    help.addArgument('Compilers', '-with-f90-source=<file>', nargs.Arg(None, None, 'Specify the C source for the F90 interface, e.g. src/sys/src/f90/f90_intel.c'))
    return

  def checkCRestrict(self):
    '''Check for the C restrict keyword'''
    self.restrictKeyword = ' '
    self.pushLanguage('C')
    # Try the official restrict keyword, then gcc's __restrict__, then
    # SGI's __restrict.  __restrict has slightly different semantics than
    # restrict (it's a bit stronger, in that __restrict pointers can't
    # overlap even with non __restrict pointers), but I think it should be
    # okay under the circumstances where restrict is normally used.
    for kw in ['restrict', ' __restrict__', '__restrict']:
      if self.checkCompile('', 'floater * '+kw+' x;'):
        self.restrictKeyword = kw
        self.logPrint('Set C restrict keyword to '+self.restrictKeyword, 4, 'compilers')
        break
    self.popLanguage()
    if not self.restrictKeyword:
      self.logPrint('No C restrict keyword', 4, 'compilers')
    return

  def checkCFormatting(self):
    '''Activate format string checking if using the GNU compilers'''
    if self.isGCC:
      self.gccFormatChecking = ('PRINTF_FORMAT_CHECK(A,B)', '__attribute__((format (printf, A, B)))')
      self.logPrint('Added gcc printf format checking', 4, 'compilers')
    else:
      self.gccFormatChecking = None
    return

  def checkCxxOptionalExtensions(self):
    '''Check whether the C++ compiler (IBM xlC, OSF5) need special flag for .c files which contain C++'''
    self.pushLanguage('C++')
    cxxObj = self.framework.getCompilerObject('C++')
    oldExt = cxxObj.sourceExtension
    cxxObj.sourceExtension = self.framework.getCompilerObject('C').sourceExtension
    success=0
    for flag in ['', '-+', '-x cxx -tlocal', '-Kc++']:
      try:
        self.addCompilerFlag(flag, body = 'class somename { int i; };')
        success=1
        break
      except RuntimeError:
        pass
    if success==0:
      for flag in ['-TP','-P']:
        try:
          self.addCompilerFlag(flag, body = 'class somename { int i; };', compilerOnly = 1)
          break
        except RuntimeError:
          pass
    cxxObj.sourceExtension = oldExt
    self.popLanguage()
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
    else:
      self.logPrint('C++ does not have namespaces', 4, 'compilers')
    return

  def checkFortranTypeSizes(self):
    '''Check whether real*8 is supported and suggest flags which will allow support'''
    self.pushLanguage('FC')
    # Check whether the compiler (ifc) bitches about real*8, if so try using -w90 -w to eliminate bitch
    (output, error, returnCode) = self.outputCompile('', '      real*8 variable', 1)
    if (output+error).find('Type size specifiers are an extension to standard Fortran 95') >= 0:
      oldFlags = self.framework.argDB['FFLAGS']
      self.framework.argDB['FFLAGS'] += ' -w90 -w'
      (output, error, returnCode) = self.outputCompile('', '      real*8 variable', 1)
      if returnCode or (output+error).find('Type size specifiers are an extension to standard Fortran 95') >= 0:
        self.framework.argDB['FFLAGS'] = oldFlags
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

  def checkFortranNameMangling(self):
    '''Checks Fortran name mangling, and defines HAVE_FORTRAN_UNDERSCORE, HAVE_FORTRAN_NOUNDERSCORE, HAVE_FORTRAN_CAPS, or HAVE_FORTRAN_STDCALL
    Also checks weird g77 behavior, and defines HAVE_FORTRAN_UNDERSCORE_UNDERSCORE if necessary'''
    oldLIBS = self.framework.argDB['LIBS']

    # Define known manglings and write tests in C
    cobjs     = ['confc0.o','confc1.o','confc2.o','confc3.o']
    cfuncs    = ['void d1chk_(void){return;}\n','void d1chk(void){return;}\n','void __stdcall D1CHK(void){return;}\n','void D1CHK(void){return;}\n']
    manglers  = ['underscore','unchanged','stdcall','capitalize']
    found     = 0

    for cfunc, cobj, mangler in zip(cfuncs, cobjs, manglers):
      self.framework.log.write('Testing Fortran mangling type '+mangler+' with code '+cfunc)
      # Compile each of the C test objects
      self.pushLanguage('C')
      if not self.checkCompile(cfunc,None,cleanup = 0):
        self.logPrint('Cannot compile C function: '+cfunc, 3, 'compilers')
        continue
      if not os.path.isfile(self.compilerObj):
        self.logPrint('Cannot locate object file: '+os.path.abspath(self.compilerObj), 3, 'compilers')
        continue
      os.rename(self.compilerObj,cobj)
      self.popLanguage()

      # Link each test object against Fortran driver.  If successful, then mangling found.
      self.pushLanguage('FC')
      self.framework.argDB['LIBS'] += ' '+cobj
      if self.checkLink(None,'       call d1chk()\n'):
        self.fortranMangling = mangler
        self.framework.argDB['LIBS'] = oldLIBS
        found = 1
      self.framework.argDB['LIBS'] = oldLIBS
      self.popLanguage()
      if found:
        break
    else:
      raise RuntimeError('Unknown Fortran name mangling')
    self.logPrint('Fortran name mangling is '+self.fortranMangling, 4, 'compilers')

    # Clean up C test objects
    for cobj in cobjs:
      if os.path.isfile(cobj): os.remove(cobj)

    # Check double trailing underscore
    #
    #   Create C test object
    self.pushLanguage('C')
    if not self.checkCompile('void d1_chk__(void) {return;}\n',None,cleanup = 0):
      raise RuntimeError('Cannot compile C function: double underscore test')
    if not os.path.isfile(self.compilerObj):
      raise RuntimeError('Cannot locate object file: '+os.path.abspath(self.compilerObj))
    os.rename(self.compilerObj, cobjs[0])
    self.framework.argDB['LIBS'] += ' '+cobjs[0]
    self.popLanguage()

    #   Test against driver
    self.pushLanguage('FC')
    if self.checkLink(None,'       call d1_chk()\n'):
      self.fortranManglingDoubleUnderscore = 1
      self.logPrint('Fortran has funny g77 name mangling with double underscores', 4, 'compilers')
    else:
      self.fortranManglingDoubleUnderscore = 0

    #   Cleanup
    if os.path.isfile(cobjs[0]): os.remove(cobjs[0])
    self.framework.argDB['LIBS'] = oldLIBS
    self.popLanguage()
    return

  def checkFortranPreprocessor(self):
    '''Determine if Fortran handles preprocessing properly'''
    self.pushLanguage('FC')
    # Does Fortran compiler need special flag for using CPP
    for flag in ['', '-cpp', '-xpp=cpp', '-F', '-Cpp', '-fpp', '-fpp:-m']:
      try:
        flagsArg = self.getCompilerFlagsArg()
        oldFlags = self.framework.argDB[flagsArg]
        self.framework.argDB[flagsArg] = self.framework.argDB[flagsArg]+' '+'-DPTesting'
        self.addCompilerFlag(flag, body = '#define dummy \n           dummy\n#ifndef PTesting\n       fooey\n#endif')
        self.framework.argDB[flagsArg] = oldFlags + ' ' + flag
        self.fortranPreprocess = 1
        self.popLanguage()
        self.logPrint('Fortran uses CPP preprocessor', 3, 'compilers')
        return
      except RuntimeError:
        self.framework.argDB[flagsArg] = oldFlags
    self.popLanguage()
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
    if not 'CC' in self.framework.argDB or not 'FC' in self.framework.argDB: 
      return
    oldFlags = self.framework.argDB['LDFLAGS']
    self.framework.argDB['LDFLAGS'] += ' -v'
    self.pushLanguage('FC')
    (output, returnCode) = self.outputLink('', '')
    self.framework.argDB['LDFLAGS'] = oldFlags
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
        
    # The easiest thing to do for xlf output is to replace all the commas
    # with spaces.  Try to only do that if the output is really from xlf,
    # since doing that causes problems on other systems.
    if output.find('xlf') >= 0:
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
    flibs   = []
    lflags  = []
    try:
      while 1:
        arg = argIter.next()
        self.logPrint( 'Checking arg '+arg, 4, 'compilers')
        # Check for full library name
        m = re.match(r'^/.*\.a$', arg)
        if m:
          if not arg in lflags:
            lflags.append(arg)
            self.logPrint('Found full library spec: '+arg, 4, 'compilers')
            flibs.append(arg)
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
        m = re.match(r'^-l(ang.*|crt0.o|crt1.o|crtbegin.o|c|gcc)$', arg)
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
            else:
              lflags.append(arg)
            self.logPrint('Found special library: '+arg, 4, 'compilers')
            flibs.append(arg)
          continue
        if arg == '-rpath':
          lib = argIter.next()
          self.logPrint('Found -rpath library: '+lib, 4, 'compilers')
          flibs.append(self.setCompilers.CSharedLinkerFlag+lib)
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
        self.logPrint('Unknown arg '+arg, 4, 'compilers')
    except StopIteration:
      pass

    # Change to string
    self.flibs = []
    for lib in flibs:
      if 'FC_LINKER_SLFLAG' in self.framework.argDB and lib.startswith('-L'):
        self.flibs.append(self.framework.argDB['FC_LINKER_SLFLAG']+lib[2:])
      self.flibs.append(lib)
    # Append run path
    self.flibs = ldRunPath+self.flibs

    # on OS X, mixing g77 3.4 with gcc-3.3 requires using -lcc_dynamic
    for l in self.flibs:
      if l.find('-L/sw/lib/gcc/powerpc-apple-darwin') >= 0:
        self.flibs.append('-lcc_dynamic')
        self.framework.log.write('Detected Apple Mac Fink libraries used; adding -lcc_dynamic so Fortran can work with C++')
        break

    self.logPrint('Libraries needed to link against Fortran compiler'+str(self.flibs), 3, 'compilers')
    # check that these monster libraries can be used from C
    self.logPrint('Check that Fortran libraries can be used from C', 4, 'compilers')
    oldLibs = self.framework.argDB['LIBS']
    self.framework.argDB['LIBS'] += ' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
    try:
      self.setCompilers.checkCompiler('C')
    except RuntimeError, e:
      self.logPrint('Fortran libraries cannot directly be used from C, try without -lcrt2.o', 4, 'compilers')
      self.logPrint('Error message from compiling {'+str(e)+'}', 4, 'compilers')
      # try removing this one
      if '-lcrt2.o' in self.flibs: self.flibs.remove('-lcrt2.o')
      self.framework.argDB['LIBS'] = oldLibs+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
      try:
        self.setCompilers.checkCompiler('C')
      except RuntimeError, e:
        self.logPrint(str(e), 4, 'compilers')
        raise RuntimeError('Fortran libraries cannot be used with C compiler')

    # check if Intel library exists (that is not linked by default but has iargc_ in it :-(
    self.logPrint('Check for Intel PEPCF90 library', 4, 'compilers')
    self.framework.argDB['LIBS'] = oldLibs+' -lPEPCF90 '+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
    try:
      self.setCompilers.checkCompiler('C')
      self.flibs = [' -lPEPCF90']+self.flibs
      self.logPrint('Intel PEPCF90 library exists', 4, 'compilers')
    except RuntimeError, e:
      self.logPrint('Intel PEPCF90 library does not exist', 4, 'compilers')
      self.framework.argDB['LIBS'] = oldLibs+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])

    # check these monster libraries work from C++
    if 'CXX' in self.framework.argDB:
      self.logPrint('Check that Fortran libraries can be used from C++', 4, 'compilers')
      self.framework.argDB['LIBS'] = oldLibs+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
      try:
        self.setCompilers.checkCompiler('C++')
        self.logPrint('Fortran libraries can be used from C++', 4, 'compilers')
      except RuntimeError, e:
        self.logPrint(str(e), 4, 'compilers')
        # try removing this one causes grief with gnu g++ and Intel Fortran
        if '-lintrins' in self.flibs: self.flibs.remove('-lintrins')
        self.framework.argDB['LIBS'] = oldLibs+' '.join([self.libraries.getLibArgument(lib) for lib in self.flibs])
        try:
          self.setCompilers.checkCompiler('C++')
        except RuntimeError, e:
          self.logPrint(str(e), 4, 'compilers')
          if str(e).find('INTELf90_dclock') >= 0:
            self.logPrint('Intel 7.1 Fortran compiler cannot be used with g++ 3.2!', 2, 'compilers')
          raise RuntimeError('Fortran libraries cannot be used with C++ compiler.\n Run with --with-fc=0 or --with-cxx=0')

    self.framework.argDB['LIBS'] = oldLibs
    return

  def checkFortran90(self):
    '''Determine whether the Fortran compiler handles F90'''
    self.fortranIsF90 = 0
    if self.framework.argDB.has_key('with-f90-header') and self.framework.argDB.has_key('with-f90-source'):
      self.fortranIsF90 = 1
    return

  def stripquotes(self,str):
    if str[0] =='"': str = str[1:]
    if str[-1] =='"': str = str[:-1]
    return str

  def getFortran90SourceGuesses(self):
    headerGuess = None
    sourceGuess = None
    if self.framework.argDB['with-vendor-compilers'] and not self.framework.argDB['with-vendor-compilers'] == '0':
      if self.setCompilers.FC.startswith('win32fe'):
        headerGuess = 'f90_win32.h'
        sourceGuess = 'f90_win32.c'
      if self.framework.argDB['with-vendor-compilers'] == 'absoft':
        headerGuess = 'f90_absoft.h'
        sourceGuess = 'f90_absoft.c'
      elif self.framework.argDB['with-vendor-compilers'] == 'cray':
        headerGuess = 'f90_t3e.h'
        sourceGuess = 'f90_t3e.c'
        #headerGuess = 'f90_cray_x1.h'
        #sourceGuess = 'f90_cray_x1.c'
      elif self.framework.argDB['with-vendor-compilers'] == 'dec':
        headerGuess = 'f90_alpha.h'
        sourceGuess = 'f90_alpha.c'
      elif self.framework.argDB['with-vendor-compilers'] == 'hp':
        headerGuess = 'f90_hpux.h'
        sourceGuess = 'f90_hpux.c'
      elif self.framework.argDB['with-vendor-compilers'] == 'ibm':
        headerGuess = 'f90_rs6000.h'
        sourceGuess = 'f90_rs6000.c'
      elif self.framework.argDB['with-vendor-compilers'] == 'intel':
        #headerGuess = 'f90_intel.h'
        #sourceGuess = 'f90_intel.c'
        headerGuess = 'f90_intel8.h'
        sourceGuess = 'f90_intel8.c'
      elif self.framework.argDB['with-vendor-compilers'] == 'lahaye':
        headerGuess = 'f90_nag.h'
        sourceGuess = 'f90_nag.c'
      elif self.framework.argDB['with-vendor-compilers'] == 'portland':
        headerGuess = 'f90_pgi.h'
        sourceGuess = 'f90_pgi.c'
      elif self.framework.argDB['with-vendor-compilers'] == 'sgi':
        headerGuess = 'f90_IRIX.h'
        sourceGuess = 'f90_IRIX.c'
      elif self.framework.argDB['with-vendor-compilers'] == 'solaris':
        headerGuess = 'f90_solaris.h'
        sourceGuess = 'f90_solaris.c'
    return (headerGuess, sourceGuess)

  def checkFortran90Interface(self):
    '''Check for custom F90 interfaces, such as that provided by PETSc'''
    if not self.fortranIsF90:
      return
    (headerGuess, sourceGuess) = self.getFortran90SourceGuesses()
    if 'with-f90-header' in self.framework.argDB:
      headerGuess = self.stripquotes(self.framework.argDB['with-f90-header'])
    if 'with-f90-source' in self.framework.argDB.has_key():
      sourceGuess = self.stripquotes(self.framework.argDB['with-f90-source'])
    if headerGuess:
      headerPath = os.path.abspath(headerGuess)
      if not os.path.isfile(headerPath):
        headerPath = os.path.abspath(os.path.join('include', headerGuess))
        if not os.path.isfile(headerPath):
          raise RuntimeError('Invalid F90 header: '+str(headerPath))
      self.f90HeaderPath = headerPath
    if sourceGuess:
      sourcePath = os.path.abspath(sourceGuess)
      if not os.path.isfile(sourcePath):
        raise RuntimeError('Invalid F90 source: '+str(sourcePath))
      self.f90SourcePath = sourcePath
    return

  def output(self):
    '''Output module data as defines and substitutions'''
    if 'CC' in self.framework.argDB:
      self.pushLanguage('C')
      setattr(self, 'CC', self.argDB['CC'])
      if 'CPP' in self.framework.argDB:
        setattr(self, 'CPP', self.argDB['CPP'])
      setattr(self, self.getCompilerFlagsArg(), self.argDB[self.getCompilerFlagsArg()])
      setattr(self, 'CPPFLAGS', self.argDB['CPPFLAGS'])
      setattr(self, self.getLinkerFlagsArg(), self.argDB[self.getLinkerFlagsArg()])
      self.popLanguage()
      # Define to equivalent of C99 restrict keyword, or to nothing if this is not supported.  Do not define if restrict is supported directly.
      if not self.restrictKeyword == 'restrict':
        self.addDefine('RESTRICT', self.restrictKeyword)
      if self.gccFormatChecking:
        self.addDefine(self.gccFormatChecking[0], self.gccFormatChecking[1])
    if 'CXX' in self.framework.argDB:
      self.pushLanguage('C++')
      setattr(self, 'CXX', self.argDB['CXX'])
      if 'CXXCPP' in self.framework.argDB:
        setattr(self, 'CXXCPP', self.argDB['CXXCPP'])
      setattr(self, self.getCompilerFlagsArg(), self.argDB[self.getCompilerFlagsArg()])
      setattr(self, self.getCompilerFlagsArg(1), self.argDB[self.getCompilerFlagsArg(1)])
      setattr(self, self.getLinkerFlagsArg(), self.argDB[self.getLinkerFlagsArg()])
      self.popLanguage()
      if self.cxxNamespace:
        self.addDefine('HAVE_CXX_NAMESPACE', 1)
    if 'FC' in self.framework.argDB:
      self.pushLanguage('FC')
      setattr(self, 'FC', self.argDB['FC'])
      setattr(self, self.getCompilerFlagsArg(), self.argDB[self.getCompilerFlagsArg()])
      setattr(self, self.getLinkerFlagsArg(), self.argDB[self.getLinkerFlagsArg()])
      self.popLanguage()
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
      if self.fortranManglingDoubleUnderscore:
        self.addDefine('HAVE_FORTRAN_UNDERSCORE_UNDERSCORE',1)
    self.AR = self.setCompilers.AR
    self.AR_FLAGS = self.setCompilers.AR_FLAGS
    self.RANLIB = self.setCompilers.RANLIB
    self.LD_SHARED = self.setCompilers.LD_SHARED
    if hasattr(self, 'f90HeaderPath'):
      self.addDefine('HAVE_F90_H', '"'+self.f90HeaderPath+'"')
    if hasattr(self, 'f90SourcePath'):
      self.addDefine('HAVE_F90_C', '"'+self.f90SourcePath+'"')
    return

  def configure(self):
    if 'CC' in self.framework.argDB:
      import config.setCompilers

      self.isGCC  = config.setCompilers.Configure.isGNU(self.framework.argDB['CC'])
      self.executeTest(self.checkCRestrict)
      self.executeTest(self.checkCFormatting)
    else:
      self.isGCC  = 0
    if 'CXX' in self.framework.argDB:
      self.isGCXX = config.setCompilers.Configure.isGNU(self.framework.argDB['CXX'])
      self.executeTest(self.checkCxxNamespace)
      self.executeTest(self.checkCxxOptionalExtensions)
    else:
      self.isGCXX = 0
    if 'FC' in self.framework.argDB:
      self.executeTest(self.checkFortranTypeSizes)
      self.executeTest(self.checkFortranNameMangling)
      self.executeTest(self.checkFortranPreprocessor)
      self.executeTest(self.checkFortranLibraries)
      self.executeTest(self.checkFortran90Interface)
    self.executeTest(self.output)
    if self.framework.compilers is None:
      self.logPrint('Setting framework compilers to this module', 2, 'compilers')
      self.framework.compilers = self
    return
