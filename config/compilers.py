import config.base

import re
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.setCompilers = self.framework.require('config.setCompilers', self)
    return

  def __str__(self):
    return ''

  def configureHelp(self, help):
    import nargs

    help.addArgument('Compilers', '-with-f90-header=<file>', nargs.Arg(None, None, 'Specify the C header for the F90 interface'))
    help.addArgument('Compilers', '-with-f90-source=<file>', nargs.Arg(None, None, 'Specify the C source for the F90 interface'))
    return

  def checkCRestrict(self):
    '''Check for the C restrict keyword'''
    keyword = 'unsupported'
    self.pushLanguage('C')
    # Try the official restrict keyword, then gcc's __restrict__, then
    # SGI's __restrict.  __restrict has slightly different semantics than
    # restrict (it's a bit stronger, in that __restrict pointers can't
    # overlap even with non __restrict pointers), but I think it should be
    # okay under the circumstances where restrict is normally used.
    for kw in ['restrict', ' __restrict__', '__restrict']:
      if self.checkCompile('', 'float * '+kw+' x;'):
        keyword = kw
        break
    self.popLanguage()
    # Define to equivalent of C99 restrict keyword, or to nothing if this is not supported.  Do not define if restrict is supported directly.
    if not keyword == 'restrict':
      if keyword == 'unsupported':
        keyword = ''
      self.framework.addDefine('restrict', keyword)
    return

  def checkCFormatting(self):
    '''Activate format string checking if using the GNU compilers'''
    if self.isGCC:
      self.addDefine('PRINTF_FORMAT_CHECK(A,B)', '__attribute__((format (printf, A, B)))')
    return

  def checkCxxOptionalExtensions(self):
    '''Check whether the C++ compiler (IBM xlC, OSF5) need special flag for .c files which contain C++'''
    self.pushLanguage('C++')
    self.sourceExtension = '.c'
    success=0
    for flag in ['', '-+', '-x cxx -tlocal']:
      try:
        self.addCompilerFlag(flag, body = 'class somename { int i; };')
        success=1
        break
      except RuntimeError:
        pass
    if success==0:
      for flag in ['-TP']:
        try:
          self.addCompilerFlag(flag, body = 'class somename { int i; };', compilerOnly = 1)
          break
        except RuntimeError:
          pass
    self.popLanguage()
    return

  def checkCxxNamespace(self):
    '''Checks that C++ compiler supports namespaces, and if it does defines HAVE_CXX_NAMESPACE'''
    self.pushLanguage('C++')
    if self.checkCompile('namespace petsc {int dummy;}'):
      self.addDefine('HAVE_CXX_NAMESPACE', 1)
    self.popLanguage()
    return

  def checkFortranTypeSizes(self):
    '''Check whether real*8 is supported and suggest flags which will allow support'''
    self.pushLanguage('F77')
    # Check whether the compiler (ifc) bitches about real*8, if so try using -w90 -w to eliminate bitch
    (output, error, returnCode) = self.outputCompile('', '      real*8 variable', 1)
    if output.find('Type size specifiers are an extension to standard Fortran 95') >= 0:
      oldFlags = self.framework.argDB['FFLAGS']
      self.framework.argDB['FFLAGS'] += ' -w90 -w'
      (output, error, returnCode) = self.outputCompile('', '      real*8 variable', 1)
      if returnCode or output.find('Type size specifiers are an extension to standard Fortran 95') >= 0:
        self.framework.argDB['FFLAGS'] = oldFlags
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
    Also checks wierd g77 behavior, and defines HAVE_FORTRAN_UNDERSCORE_UNDERSCORE if necessary'''
    oldLIBS = self.framework.argDB['LIBS']

    # Define known manglings and write tests in C
    cobjs     = ['confc0.o','confc1.o','confc2.o','confc3.o']
    cfuncs    = ['void d1chk_(void){return;}\n','void d1chk(void){return;}\n','void D1CHK(void){return;}\n','void __stdcall D1CHK(void){return;}\n']
    manglers  = ['underscore','unchanged','capitalize','stdcall']
    manglDEFs = ['HAVE_FORTRAN_UNDERSCORE','HAVE_FORTRAN_NOUNDERSCORE','HAVE_FORTRAN_CAPS','HAVE_FORTRAN_STDCALL']
    found     = 0

    for cfunc, cobj, mangler, manglDEF in zip(cfuncs, cobjs, manglers, manglDEFs):
      # Compile each of the C test objects
      self.pushLanguage('C')
      if not self.checkCompile(cfunc,None,cleanup = 0):
        self.framework.log.write('Cannot compile C function: '+cfunc)
        continue
      if not os.path.isfile(self.compilerObj):
        self.framework.log.write('Cannot locate object file: '+os.path.abspath(self.compilerObj))
        continue
      os.rename(self.compilerObj,cobj)
      self.popLanguage()

      # Link each test object against F77 driver.  If successful, then mangling found.
      self.pushLanguage('F77')
      self.framework.argDB['LIBS'] += ' '+cobj
      if self.checkLink(None,'       call d1chk()\n'):
        self.fortranMangling = mangler
        self.addDefine(manglDEF, 1)
        self.framework.argDB['LIBS'] = oldLIBS
        if mangler == 'stdcall':
          self.addDefine('STDCALL', '__stdcall')
          self.addDefine('HAVE_FORTRAN_CAPS', 1)
          self.addDefine('HAVE_FORTRAN_MIXED_STR_ARG', 1)
        found = 1
      self.framework.argDB['LIBS'] = oldLIBS
      self.popLanguage()
      if found:
        break
    else:
      raise RuntimeError('Unknown Fortran name mangling')

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
    self.pushLanguage('F77')
    if self.checkLink(None,'       call d1_chk()\n'):
      self.fortranManglingDoubleUnderscore = 1
      self.addDefine('HAVE_FORTRAN_UNDERSCORE_UNDERSCORE',1)
    else:
      self.fortranManglingDoubleUnderscore = 0

    #   Cleanup
    if os.path.isfile(cobjs[0]): os.remove(cobjs[0])
    self.framework.argDB['LIBS'] = oldLIBS
    self.popLanguage()
    return

  def checkFortranPreprocessor(self):
    '''Determine if Fortran handles preprocessing properly'''
    self.pushLanguage('F77')
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
        self.framework.log.write('Fortran uses CPP preprocessor\n')
        return
      except RuntimeError:
        self.framework.argDB[flagsArg] = oldFlags
    self.popLanguage()
    self.fortranPreprocess = 0
    self.framework.log.write('Fortran does NOT use CPP preprocessor')
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
    add these Fortran 77 libraries.  Hence, the macro
    AC_F77_LIBRARY_LDFLAGS was created to determine these Fortran 77
    libraries.

    This macro was packaged in its current form by Matthew D. Langston
    <langston@SLAC.Stanford.EDU>.  However, nearly all of this macro
    came from the OCTAVE_FLIBS macro in octave-2.0.13/aclocal.m4,
    and full credit should go to John W. Eaton for writing this
    extremely useful macro.  Thank you John.'''
    if not 'CC' in self.framework.argDB or not 'FC' in self.framework.argDB: 
      self.flibs = ''
      self.addSubstitution('FLIBS', '')
      # this is not the correct place for the next 2 lines, but I'm to lazy to figure out where to put them
      self.addSubstitution('FC', '')
      self.addSubstitution('FC_SHARED_OPT', '')
      return
    oldFlags = self.framework.argDB['LDFLAGS']
    self.framework.argDB['LDFLAGS'] += ' -v'
    self.pushLanguage('F77')
    (output, returnCode) = self.outputLink('', '')
    self.framework.argDB['LDFLAGS'] = oldFlags
    self.popLanguage()

    # The easiest thing to do for xlf output is to replace all the commas
    # with spaces.  Try to only do that if the output is really from xlf,
    # since doing that causes problems on other systems.
    if output.find('xlfentry') >= 0:
      output = output.replace(',', ' ')
    # We are only supposed to find LD_RUN_PATH on Solaris systems
    # and the run path should be absolute
    ldRunPath = re.findall(r'^.*LD_RUN_PATH *= *([^ ]*).*', output)
    if ldRunPath: ldRunPath = ldRunPath[0]
    if ldRunPath and ldRunPath[0] == '/':
      if self.isGCC:
        ldRunPath = '-Xlinker -R -Xlinker '+ldRunPath
      else:
        ldRunPath = '-R '+ldRunPath
    else:
      ldRunPath = ''
    # Parse output
    argIter = iter(output.split())
    flibs   = []
    lflags  = []
    try:
      while 1:
        arg = argIter.next()
        # Check for full library name
        m = re.match(r'^/.*\.a$', arg)
        if m:
          if not arg in lflags:
            lflags.append(arg)
            #print 'Found full library spec: '+arg
            flibs.append(arg)
          continue
        # Check for ???
        m = re.match(r'^-bI:.*$', arg)
        if m:
          if not arg in lflags:
            if self.isGCC:
              lflags.append('-Xlinker')
            lflags.append(arg)
            #print 'Found binary include: '+arg
            flibs.append(arg)
          continue
        # Check for system libraries
        m = re.match(r'^-l(ang.*|crt0.o|crt1.o|crtbegin.o|c|gcc)$', arg)
        if m: continue
        # Check for canonical library argument
        m = re.match(r'^-[lLR]$', arg)
        if m:
          lib = arg+argIter.next()
          #print 'Found canonical library: '+lib
          flibs.append(lib)
          continue
        # Check for special library arguments
        m = re.match(r'^-[lLR].*$', arg)
        if m:
          if not arg in lflags:
            #TODO: if arg == '-lkernel32' and host_os.startswith('cygwin'):
            if arg == '-lkernel32':
              continue
            elif arg == '-lm':
              pass
            else:
              lflags.append(arg)
            #print 'Found special library: '+arg
            flibs.append(arg)
          continue
        # Check for ???
##        This breaks for the Intel Fortran compiler
##        if arg == '-u':
##          lib = arg+' '+argIter.next()
##          #print 'Found u library: '+lib
##          flibs.append(lib)
##          continue
        # Check for ???
        # Should probably try to ensure unique directory options here too.
        # This probably only applies to Solaris systems, and then will only
        # work with gcc...
        if arg == '-Y':
          for lib in argIter.next().split(':'):
            flibs.append('-L'+lib)
          continue
    except StopIteration:
      pass

    # Change to string
    self.flibs = ''
    for lib in flibs:
      if self.setCompilers.slpath and lib.startswith('-L'):
        self.flibs += ' '+self.setCompilers.slpath+lib[2:]
      self.flibs += ' '+lib
    # Append run path
    if ldRunPath: self.flibs = ldRunPath+self.flibs
    
    # check that these monster libraries can be used from C
    oldLibs = self.framework.argDB['LIBS']
    self.framework.argDB['LIBS'] += ' '+self.flibs
    try:
      self.setCompilers.checkCompiler('C')
    except RuntimeError, e:
      self.framework.log.write(str(e)+'\n')
      # try removing this one
      self.flibs = re.sub('-lcrt2.o','',self.flibs)
      self.framework.argDB['LIBS'] = oldLibs+self.flibs
      try:
        self.setCompilers.checkCompiler('C')
      except RuntimeError:
        self.framework.log.write(str(e)+'\n')
        raise RuntimeError('Fortran libraries cannot be used with C compiler')

    # check these monster libraries work from C++
    if 'CXX' in self.framework.argDB:
      self.framework.argDB['LIBS'] += oldLibs+self.flibs
      try:
        self.setCompilers.checkCompiler('C++')
      except RuntimeError:
        self.framework.log.write(str(e)+'\n')
        # try removing this one causes grief with gnu g++ and Intel Fortran
        self.flibs = re.sub('-lintrins','',self.flibs)
        self.framework.argDB['LIBS'] = oldLibs+self.flibs
        try:
          self.setCompilers.checkCompiler('C++')
        except RuntimeError:
          self.framework.log.write(str(e)+'\n')
          raise RuntimeError('Fortran libraries cannot be used with C++ compiler.\n Run with --with-fc=0 or --with-cxx=0')


    self.framework.argDB['LIBS'] = oldLibs
    self.addSubstitution('FLIBS', self.flibs)
    return

  def checkFortran90Interface(self):
    '''Check for custom F90 interfaces, such as that provided by PETSc'''
    if self.framework.argDB.has_key('with-f90-header'):
      self.addDefine('PETSC_HAVE_F90_H', self.framework.argDB['with-f90-header'])
    if self.framework.argDB.has_key('with-f90-source'):
      self.addDefine('PETSC_HAVE_F90_C', self.framework.argDB['with-f90-source'])
    return

  def configure(self):
    if 'CC' in self.framework.argDB:
      import config.setCompilers

      self.isGCC = config.setCompilers.Configure.isGNU(self.framework.argDB['CC'])
      self.executeTest(self.checkCRestrict)
      self.executeTest(self.checkCFormatting)
    if 'CXX' in self.framework.argDB:
      self.executeTest(self.checkCxxNamespace)
      self.executeTest(self.checkCxxOptionalExtensions)
    if 'FC' in self.framework.argDB:
      self.executeTest(self.checkFortranTypeSizes)
      self.executeTest(self.checkFortranNameMangling)
      self.executeTest(self.checkFortranPreprocessor)
    self.executeTest(self.checkFortranLibraries)
    self.executeTest(self.checkFortran90Interface)
    return
