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
    help.addOption('Compilers', '-with-cpp=<prog>', 'Specify the C preprocessor')
    help.addOption('Compilers', '-with-cc=<prog>', 'Specify the C compiler')
    help.addOption('Compilers', '-with-cxx=<prog>', 'Specify the C++ compiler')
    help.addOption('Compilers', '-with-fc=<prog>', 'Specify the Fortran compiler')
    #help.addOption('Compilers', '-with-f90=<prog>', 'Specify the Fortran 90 compiler')
    help.addOption('Compilers', '-with-f90-header=<file>', 'Specify the C header for the F90 interface')
    help.addOption('Compilers', '-with-f90-source=<file>', 'Specify the C source for the F90 interface')

    help.addOption('Compilers', '-CPP=<prog>', 'Specify the C preprocessor')
    help.addOption('Compilers', '-CPPFLAGS=<string>', 'Specify the C preprocessor options')
    help.addOption('Compilers', '-CXXPP=<prog>', 'Specify the C++ preprocessor')
    help.addOption('Compilers', '-CC=<prog>', 'Specify the C compiler')
    help.addOption('Compilers', '-CFLAGS=<string>', 'Specify the C compiler options')
    help.addOption('Compilers', '-CXX=<prog>', 'Specify the C++ compiler')
    help.addOption('Compilers', '-CXXFLAGS=<string>', 'Specify the C++ compiler options')
    help.addOption('Compilers', '-FC=<prog>', 'Specify the Fortran compiler')
    help.addOption('Compilers', '-FFLAGS=<string>', 'Specify the Fortran compiler options')
    help.addOption('Compilers', '-LDFLAGS=<string>', 'Specify the linker options')

    self.framework.argDB['CPPFLAGS'] = ''
    self.framework.argDB['CFLAGS']   = ''
    self.framework.argDB['CXXFLAGS'] = ''
    self.framework.argDB['FFLAGS']   = ''
    self.framework.argDB['LDFLAGS']  = ''
    return

  def checkCCompiler(self):
    '''Determine the C compiler using --with-cc, then CC, then a search
    - Also determines the preprocessor from --with-cpp, then CPP, then the C compiler'''
    if self.framework.argDB.has_key('with-cc'):
      self.CC = self.framework.argDB['with-cc']
    elif self.framework.argDB.has_key('CC'):
      self.CC = self.framework.argDB['CC']
    else:
      if not self.getExecutables(['gcc', 'cc', 'xlC', 'xlc', 'pgcc'], resultName = 'CC'):
        raise RuntimeError('Could not find a C compiler. Please set with the option --with-cc or -CC')
    self.framework.argDB['CC'] = self.CC
    self.addSubstitution('CC', self.CC)

    if self.framework.argDB.has_key('with-cpp'):
      self.CPP = self.framework.argDB['with-cpp']
    elif self.framework.argDB.has_key('CPP'):
      self.CPP = self.framework.argDB['CPP']
    else:
      self.CPP = self.CC+' -E'
    self.framework.argDB['CPP'] = self.CPP
    self.addSubstitution('CPP', self.CPP)
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
    if self.CC  == "gcc":
      self.addDefine('PRINTF_FORMAT_CHECK(A,B)', '__attribute__((format (printf, A, B)))')
    return

  def checkCxxCompiler(self):
    '''Determine the C++ compiler using --with-cxx, then CXX, then a search
    - Also determines the preprocessor from --with-cxxcpp, then CXXCPP, then the C++ compiler'''
    if self.framework.argDB.has_key('with-cxx'):
      self.CXX = self.framework.argDB['with-cxx']
    elif self.framework.argDB.has_key('CXX'):
      self.CXX = self.framework.argDB['CXX']
    else:
      if not self.getExecutables(['g++', 'c++', 'CC', 'xlC', 'pgCC', 'cxx', 'cc++', 'cl'], resultName = 'CXX'):
        raise RuntimeError('Could not find a C++ compiler. Please set with the option --with-cxx or -CXX')
    self.framework.argDB['CXX'] = self.CXX
    self.addSubstitution('CXX', self.CXX)

    if self.framework.argDB.has_key('with-cxxcpp'):
      self.CXXCPP = self.framework.argDB['with-cxxcpp']
    elif self.framework.argDB.has_key('CXXCPP'):
      self.CXXCPP = self.framework.argDB['CXXCPP']
    else:
      self.CXXCPP = self.CXX+' -E'
    self.framework.argDB['CXXCPP'] = self.CXXCPP
    self.addSubstitution('CXXCPP', self.CXXCPP)
    return

  def checkCxxNamespace(self):
    '''Checks that C++ compiler supports namespaces, and if it does defines HAVE_CXX_NAMESPACE'''
    self.pushLanguage('C++')
    if self.checkCompile('namespace petsc {int dummy;}'):
      self.addDefine('HAVE_CXX_NAMESPACE', 1)
    self.popLanguage()
    return

  def checkFortranCompiler(self):
    '''Determine the Fortran compiler using --with-fc, then FC, then a search'''
    if self.framework.argDB.has_key('with-fc'):
      self.FC = self.framework.argDB['with-fc']
    elif self.framework.argDB.has_key('FC'):
      self.FC = self.framework.argDB['FC']
    else:
      if not self.getExecutables(['g77', 'f77', 'pgf77'], resultName = 'FC'):
        raise RuntimeError('Could not find a Fortran 77 compiler. Please set with the option --with-fc or -FC')
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
      self.F90 = self.framework.argDB['with-f90']
    elif self.framework.argDB.has_key('F90'):
      self.F90 = self.framework.argDB['F90']
    else:
      if not self.getExecutables(['f90', 'pgf90', 'ifc'], resultName = 'F90'):
        #raise RuntimeError('Could not find a Fortran 90 compiler. Please set with the option --with-f90 or -F90')
        self.F90 = 'f90'
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
    isGCC = self.framework.argDB['CC'] == 'gcc'
    oldFlags = self.framework.argDB['LDFLAGS']
    self.framework.argDB['LDFLAGS'] += ' -v'
    self.pushLanguage('F77')
    output = self.outputLink('', '')
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
      if isGCC:
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
            if isGCC:
              lflags.append('-Xlinker')
            lflags.append(arg)
            #print 'Found binary include: '+arg
            flibs.append(arg)
          continue
        # Check for system libraries
        m = re.match(r'^-l(ang.*|crt0.o|c|gcc)$', arg)
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
            lib = arg+' '+argIter.next()
            #print 'Found special library: '+arg
            flibs.append(arg)
          continue
        # Check for ???
        if arg == '-u':
          lib = arg+' '+argIter.next()
          #print 'Found u library: '+lib
          flibs.append(lib)
          continue
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

  def configure(self):
    self.executeTest(self.checkCCompiler)
    self.executeTest(self.checkCRestrict)
    self.executeTest(self.checkCFormatting)
    self.executeTest(self.checkCxxCompiler)
    self.executeTest(self.checkCxxNamespace)
    self.executeTest(self.checkFortranCompiler)
    self.executeTest(self.checkFortranNameMangling)
    self.executeTest(self.checkFortran90Compiler)
    self.executeTest(self.checkFortran90Interface)
    self.executeTest(self.checkFortranLibraries)
    return
