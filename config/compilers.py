import config.base

import re
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def configureHelp(self, help):
    import nargs

    help.addArgument('Compilers', '-with-cpp=<prog>', nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-with-cc=<prog>',  nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-with-cxx=<prog>', nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-with-fc=<prog>',  nargs.Arg(None, None, 'Specify the Fortran compiler'))
    #help.addArgument('Compilers', '-with-f90=<prog>',       nargs.Arg(None, None, 'Specify the Fortran 90 compiler'))
    help.addArgument('Compilers', '-with-f90-header=<file>', nargs.Arg(None, None, 'Specify the C header for the F90 interface'))
    help.addArgument('Compilers', '-with-f90-source=<file>', nargs.Arg(None, None, 'Specify the C source for the F90 interface'))
    help.addArgument('Compilers', '-with-ld=<prog>',         nargs.Arg(None, None, 'Specify the linker'))

    help.addArgument('Compilers', '-CPP=<prog>',        nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-CPPFLAGS=<string>', nargs.Arg(None, '',   'Specify the C preprocessor options'))
    help.addArgument('Compilers', '-CXXPP=<prog>',      nargs.Arg(None, None, 'Specify the C++ preprocessor'))
    help.addArgument('Compilers', '-CC=<prog>',         nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-CFLAGS=<string>',   nargs.Arg(None, '',   'Specify the C compiler options'))
    help.addArgument('Compilers', '-CXX=<prog>',        nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-CXXFLAGS=<string>', nargs.Arg(None, '',   'Specify the C++ compiler options'))
    help.addArgument('Compilers', '-FC=<prog>',         nargs.Arg(None, None, 'Specify the Fortran compiler'))
    help.addArgument('Compilers', '-FFLAGS=<string>',   nargs.Arg(None, '',   'Specify the Fortran compiler options'))

    help.addArgument('Compilers', '-LD=<prog>',         nargs.Arg(None, None, 'Specify the default linker'))
    help.addArgument('Compilers', '-LD_CC=<prog>',      nargs.Arg(None, None, 'Specify the linker for C only'))
    help.addArgument('Compilers', '-LD_CXX=<prog>',     nargs.Arg(None, None, 'Specify the linker for C++ only'))
    help.addArgument('Compilers', '-LD_FC=<prog>',      nargs.Arg(None, None, 'Specify the linker for Fortran only'))
    help.addArgument('Compilers', '-LDFLAGS=<string>',  nargs.Arg(None, '',   'Specify the linker options'))
    return

  def isGNU(compiler):
    '''Returns true if the compiler is a GNU compiler'''
    try:
      import commands

      (status, output) = commands.getstatusoutput(compiler+' --help')
      if not status and output.find('www.gnu.org') >= 0:
        return 1
    except Exception:
      pass
    return 0
  isGNU = staticmethod(isGNU)

  def checkCCompiler(self):
    '''Determine the C compiler using --with-cc, then CC, then a search
    - Also determines the preprocessor from --with-cpp, then CPP, then the C compiler'''
    if self.framework.argDB.has_key('with-cc'):
      compilers = self.framework.argDB['with-cc']
    elif self.framework.argDB.has_key('CC'):
      compilers = self.framework.argDB['CC']
    else:
      compilers = ['gcc', 'cc', 'xlC', 'xlc', 'pgcc']
    if not isinstance(compilers, list): compilers = [compilers]
    if self.getExecutables(compilers, resultName = 'CC'):
      self.framework.argDB['CC'] = self.CC
      self.addSubstitution('CC', self.CC)
      self.isGCC = Configure.isGNU(self.framework.argDB['CC'])

      if self.framework.argDB.has_key('with-cpp'):
        preprocessors = self.framework.argDB['with-cpp']
      elif self.framework.argDB.has_key('CPP'):
        preprocessors = self.framework.argDB['CPP']
      else:
        preprocessors = [self.framework.argDB['CC']+' -E']
      if not isinstance(preprocessors, list): preprocessors = [preprocessors]
      if self.getExecutables(preprocessors, resultName = 'CPP'):
        self.framework.argDB['CPP'] = self.CPP
        self.addSubstitution('CPP', self.CPP)
    return

  def checkCRestrict(self):
    '''Check for the C restrict keyword'''
    if not self.framework.argDB.has_key('CC'): return
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
    if not self.framework.argDB.has_key('CC'): return
    if self.isGCC:
      self.addDefine('PRINTF_FORMAT_CHECK(A,B)', '__attribute__((format (printf, A, B)))')
    return

  def checkCxxCompiler(self):
    '''Determine the C++ compiler using --with-cxx, then CXX, then a search
    - Also determines the preprocessor from --with-cxxcpp, then CXXCPP, then the C++ compiler'''
    if self.framework.argDB.has_key('with-cxx'):
      compilers = self.framework.argDB['with-cxx']
    elif self.framework.argDB.has_key('CXX'):
      compilers = self.framework.argDB['CXX']
    else:
      compilers = ['g++', 'c++', 'CC', 'xlC', 'pgCC', 'cxx', 'cc++', 'cl']
    if not isinstance(compilers, list): compilers = [compilers]
    if self.getExecutables(compilers, resultName = 'CXX'):
      self.framework.argDB['CXX'] = self.CXX
      self.addSubstitution('CXX', self.CXX)
      # Check for g++
      self.isGCXX = 0
      if self.framework.argDB['CXX'].endswith('g++'):
        self.isGCXX = 1
      else:
        try:
          import commands
          (status, output) = commands.getstatusoutput(self.framework.argDB['CXX']+' --help')
          if not status and output.find('www.gnu.org') >= 0:
            self.isGCXX = 1
        except Exception, e: pass

      if self.framework.argDB.has_key('with-cxxcpp'):
        preprocessors = self.framework.argDB['with-cxxcpp']
      elif self.framework.argDB.has_key('CXXCPP'):
        preprocessors = self.framework.argDB['CXXCPP']
      else:
        preprocessors = [self.CXX+' -E']
      if not isinstance(preprocessors, list): preprocessors = [preprocessors]
      if self.getExecutables(preprocessors, resultName = 'CXXCPP'):
        self.framework.argDB['CXXCPP'] = self.CXXCPP
        self.addSubstitution('CXXCPP', self.CXXCPP)
    return

  def checkCxxNamespace(self):
    '''Checks that C++ compiler supports namespaces, and if it does defines HAVE_CXX_NAMESPACE'''
    if not self.framework.argDB.has_key('CXX'): return
    self.pushLanguage('C++')
    if self.checkCompile('namespace petsc {int dummy;}'):
      self.addDefine('HAVE_CXX_NAMESPACE', 1)
    self.popLanguage()
    return

  def checkFortranCompiler(self):
    '''Determine the Fortran compiler using --with-fc, then FC, then a search'''
    if self.framework.argDB.has_key('with-fc'):
      compilers = self.framework.argDB['with-fc']
    elif self.framework.argDB.has_key('FC'):
      compilers = self.framework.argDB['FC']
    else:
      compilers = ['g77', 'f77', 'pgf77']
    if not isinstance(compilers, list): compilers = [compilers]
    if self.getExecutables(compilers, resultName = 'FC'):
      self.framework.argDB['FC'] = self.FC
      self.addSubstitution('FC', self.FC)
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
    raise RuntimeError('Unknown Fortran name mangling: '+self.fortranMangling)

  def checkFortranNameMangling(self):
    '''Checks Fortran name mangling, and defines HAVE_FORTRAN_UNDERSCORE, HAVE_FORTRAN_NOUNDERSCORE, or HAVE_FORTRAN_CAPS
    Also checks wierd g77 behavior, and defines HAVE_FORTRAN_UNDERSCORE_UNDERSCORE if necessary'''
    if not self.framework.argDB.has_key('FC'): return
    oldLIBS = self.framework.argDB['LIBS']

    # Define known manglings and write tests in C
    numtest = [0,1,2]
    cobj    = ['confc0.o','confc1.o','confc2.o']
    cfunc   = ['void d1chk_(void){return;}\n','void d1chk(void){return;}\n','void D1CHK(void){return;}\n']
    mangler = ['underscore','unchanged','capitalize']
    manglDEF= ['HAVE_FORTRAN_UNDERSCORE','HAVE_FORTRAN_NOUNDERSCORE','HAVE_FORTRAN_CAPS']

    # Compile each of the C test objects
    self.pushLanguage('C')
    for i in numtest:
      if not self.checkCompile(cfunc[i],None,cleanup = 0):
        raise RuntimeError('Cannot compile C function: '+cfunc[i])
      if not os.path.isfile(self.compilerObj):
        raise RuntimeError('Cannot locate object file: '+os.path.abspath(self.compilerObj))
      os.rename(self.compilerObj,cobj[i])
    self.popLanguage()

    # Link each test object against F77 driver.  If successful, then mangling found.
    self.pushLanguage('F77')
    for i in numtest:
      self.framework.argDB['LIBS'] += ' '+cobj[i]
      if self.checkLink(None,'       call d1chk()\n'):
        self.fortranMangling = mangler[i]
        self.addDefine(manglDEF[i],1)
        self.framework.argDB['LIBS'] = oldLIBS
        break
      self.framework.argDB['LIBS'] = oldLIBS
    else:
      raise RuntimeError('Unknown Fortran name mangling')
    self.popLanguage()

    # Clean up C test objects
    for i in numtest:
      if os.path.isfile(cobj[i]): os.remove(cobj[i])

    # Check double trailing underscore
    #
    #   Create C test object
    self.pushLanguage('C')
    if not self.checkCompile('void d1_chk__(void) {return;}\n',None,cleanup = 0):
      raise RuntimeError('Cannot compile C function: double underscore test')
    if not os.path.isfile(self.compilerObj):
      raise RuntimeError('Cannot locate object file: '+os.path.abspath(self.compilerObj))
    os.rename(self.compilerObj,cobj[0])
    self.framework.argDB['LIBS'] += ' '+cobj[0]
    self.popLanguage()

    #   Test against driver
    self.pushLanguage('F77')
    if self.checkLink(None,'       call d1_chk()\n'):
      self.fortranManglingDoubleUnderscore = 1
      self.addDefine('HAVE_FORTRAN_UNDERSCORE_UNDERSCORE',1)
    else:
      self.fortranManglingDoubleUnderscore = 0

    #   Cleanup
    if os.path.isfile(cobj[0]): os.remove(cobj[0])
    self.framework.argDB['LIBS'] = oldLIBS
    self.popLanguage()
    return


  def checkFortran90Compiler(self):
    '''Determine the Fortran 90 compiler using --with-f90, then F90, then a search'''
    if self.framework.argDB.has_key('with-f90'):
      compilers = self.framework.argDB['with-f90']
    elif self.framework.argDB.has_key('F90'):
      compilers = self.framework.argDB['F90']
    else:
      compilers = ['f90', 'pgf90', 'ifc']
    if not isinstance(compilers, list): compilers = [compilers]
    if self.getExecutables(compilers, resultName = 'F90'):
      self.framework.argDB['F90'] = self.F90
      self.addSubstitution('F90', self.F90)
    return

  def checkFortran90Interface(self):
    '''Check for custom F90 interfaces, such as that provided by PETSc'''
    if self.framework.argDB.has_key('with-f90-header'):
      self.addDefine('PETSC_HAVE_F90_H', self.framework.argDB['with-f90-header'])
    if self.framework.argDB.has_key('with-f90-source'):
      self.addDefine('PETSC_HAVE_F90_C', self.framework.argDB['with-f90-source'])
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
    if not self.framework.argDB.has_key('CC') or not self.framework.argDB.has_key('FC'): 
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
          lib = arg+' '+argIter.next()
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
    for lib in flibs: self.flibs += ' '+lib
    # Append run path
    if ldRunPath: self.flibs = ldRunPath+self.flibs
    self.addSubstitution('FLIBS', self.flibs)
    return

  def checkSharedLinkerFlag(self):
    '''Determine what flags are necessary for dynamic library creation'''
    flag                 = '-shared'
    (output, status) = self.outputLink('', '')
    if status or output.find('unrecognized option') >= 0:
      flag                 = '-dylib'
      (output, status) = self.outputLink('', '')
      if status or output.find('unrecognized option') >= 0:
        flag = ''
    self.addSubstitution('SHARED_LIBRARY_FLAG', flag)
    return

  def checkSharedLinkerPaths(self):
    '''Determine whether the linker accepts the -rpath'''
    flag                 = '-Wl,-rpath,'
    (output, status) = self.outputLink('', '')
    if status or output.find('unknown flag') >= 0:
      flag = ''
    self.addSubstitution('RPATH', flag)
    return

  def configure(self):
    self.executeTest(self.checkCCompiler)
    self.executeTest(self.checkCRestrict)
    self.executeTest(self.checkCFormatting)
    self.executeTest(self.checkCxxCompiler)
    self.executeTest(self.checkCxxNamespace)
    self.executeTest(self.checkFortranCompiler)
    if 'FC' in self.framework.argDB:
      self.executeTest(self.checkFortranNameMangling)
    self.executeTest(self.checkFortranLibraries)
    self.executeTest(self.checkFortran90Compiler)
    self.executeTest(self.checkFortran90Interface)
    self.executeTest(self.checkSharedLinkerFlag)
    self.executeTest(self.checkSharedLinkerPaths)
    return
