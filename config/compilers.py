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
    help.addOption('Compilers', '-with-cc=<prog>', 'Specify the C compiler')
    help.addOption('Compilers', '-with-cxx=<prog>', 'Specify the C++ compiler')
    help.addOption('Compilers', '-with-fc=<prog>', 'Specify the Fortran compiler')
    help.addOption('Compilers', '-with-f90=<prog>', 'Specify the Fortran 90 compiler')
    help.addOption('Compilers', '-with-f90-header=<file>', 'Specify the C header for the F90 interface')
    help.addOption('Compilers', '-with-f90-source=<file>', 'Specify the C source for the F90 interface')
    return

  def checkCCompiler(self):
    if self.framework.argDB.has_key('with-cc'):
      self.CC = self.framework.argDB['with-cc']
    elif self.framework.argDB.has_key('CC'):
      self.CC = self.framework.argDB['CC']
    else:
      self.CC = 'gcc'
    self.addSubstitution('CC', self.CC, comment = 'C compiler')
    return

  def checkFormatting(self):
    if self.CC  == "gcc":
      self.addDefine('PRINTF_FORMAT_CHECK(A,B)', '__attribute__((format (printf, A, B)))')
    return

  def checkCxxCompiler(self):
    if self.framework.argDB.has_key('with-cxx'):
      self.CXX = self.framework.argDB['with-cxx']
    elif self.framework.argDB.has_key('CXX'):
      self.CXX = self.framework.argDB['CXX']
    else:
      self.CXX = 'g++'
    self.addSubstitution('CXX', self.CXX, comment = 'C++ compiler')
    return

  def checkCxxNamespace(self):
    '''Checks that C++ compiler supports namespaces, and if it does defines HAVE_CXX_NAMESPACE'''
    self.pushLanguage('C++')
    if self.checkCompile('namespace petsc {int dummy;}'):
      self.addDefine('HAVE_CXX_NAMESPACE', 1)
    self.popLanguage()
    return

  def checkFortranCompiler(self):
    if self.framework.argDB.has_key('with-fc'):
      self.FC = self.framework.argDB['with-fc']
    elif self.framework.argDB.has_key('FC'):
      self.FC = self.framework.argDB['FC']
    else:
      self.FC = 'g77'
    self.addSubstitution('FC', self.FC, comment = 'Fortran compiler')
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
    fobj    = 'conff.o'
    oldLibs = self.framework.argDB['LIBS']
    self.framework.argDB['LIBS'] += ' '+fobj

    self.pushLanguage('F77')
    if not self.checkCompile('        subroutine d1chk()\n        return\n        end\n', None, cleanup = 0):
      raise RuntimeError('Cannot compile Fortran program')
    self.popLanguage()
    if not os.path.isfile(self.compilerObj):
      raise RuntimeError('Cannot locate object file: '+os.path.abspath(self.compilerObj))
    os.rename(self.compilerObj, fobj)

    # Check single trailing underscore
    if self.checkLink('void d1chk_(void);\n', 'd1chk_();\nreturn 0;\n'):
      self.fortranMangling = 'underscore'
      self.addDefine('HAVE_FORTRAN_UNDERSCORE', 1)
    # Check no change
    elif self.checkLink('void d1chk(void);\n', 'd1chk();\nreturn 0;\n'):
      self.fortranMangling = 'unchanged'
      self.addDefine('HAVE_FORTRAN_NOUNDERSCORE', 1)
    # Check capitalization
    elif self.checkLink('void D1CHK(void);\n', 'D1CHK();\nreturn 0;\n'):
      self.fortranMangling = 'capitalize'
      self.addDefine('HAVE_FORTRAN_CAPS', 1)
    else:
      raise RuntimeError('Unknown Fortran name mangling')
    if os.path.isfile(fobj): os.remove(fobj)

    self.pushLanguage('F77')
    if not self.checkCompile('        subroutine d1_chk()\n        return\n        end\n', None, cleanup = 0):
      raise RuntimeError('Cannot compile Fortran program')
    self.popLanguage()
    os.rename(self.compilerObj, fobj)

    # Check double trailing underscore
    if self.checkLink('void d1_chk__(void);\n', 'd1_chk__();\nreturn 0;\n'):
      self.fortranManglingDoubleUnderscore = 1
      self.addDefine('HAVE_FORTRAN_UNDERSCORE_UNDERSCORE', 1)
    else:
      self.fortranManglingDoubleUnderscore = 0
    if os.path.isfile(fobj): os.remove(fobj)

    self.framework.argDB['LIBS'] = oldLibs
    return

  def checkFortran90Compiler(self):
    if self.framework.argDB.has_key('with-f90'):
      self.F90 = self.framework.argDB['with-f90']
    elif self.framework.argDB.has_key('F90'):
      self.F90 = self.framework.argDB['F90']
    else:
      self.F90 = 'f90'
    self.addSubstitution('F90', self.F90, comment = 'Fortran 90 compiler')
    return

  def checkFortran90Interface(self):
    if self.framework.argDB.has_key('with-f90-header'):
      self.addDefine('PETSC_HAVE_F90_H', self.framework.argDB['with-f90-header'])
    if self.framework.argDB.has_key('with-f90-source'):
      self.addDefine('PETSC_HAVE_F90_C', self.framework.argDB['with-f90-source'])
    return

  def checkFortranLibraries(self):
    '''This macro is intended to be used in those situations when it is
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
    self.addSubstitution('FLIBS', self.flibs, 'Libraries needed for linking with Fortran')
    return

  def configure(self):
    self.executeTest(self.checkCCompiler)
    self.executeTest(self.checkFormatting)
    self.executeTest(self.checkCxxCompiler)
    self.executeTest(self.checkCxxNamespace)
    self.executeTest(self.checkFortranCompiler)
    self.executeTest(self.checkFortranNameMangling)
    self.executeTest(self.checkFortran90Compiler)
    self.executeTest(self.checkFortran90Interface)
    self.executeTest(self.checkFortranLibraries)
    return
