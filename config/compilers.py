from __future__ import generators
import config.base

import re
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def __str__(self):
    desc = ['Compilers:']
    if 'CC' in self.framework.argDB:
      self.pushLanguage('C')
      desc.append('  C Compiler:         '+self.getCompiler())
      desc.append('  C Compiler Flags:   '+self.compilerFlags)
      desc.append('  C Linker:           '+self.getLinker())
      desc.append('  C Linker Flags:     '+self.linkerFlags)
      self.popLanguage()
    if 'CXX' in self.framework.argDB and self.framework.argDB['CXX']:
      self.pushLanguage('Cxx')
      desc.append('  C++ Compiler:       '+self.getCompiler())
      desc.append('  C++ Compiler Flags: '+self.compilerFlags)
      desc.append('  C++ Linker:         '+self.getLinker())
      desc.append('  C++ Linker Flags:   '+self.linkerFlags)
      self.popLanguage()
    if 'FC' in self.framework.argDB:
      self.pushLanguage('F77')
      desc.append('  Fortran Compiler:       '+self.getCompiler())
      desc.append('  Fortran Compiler Flags: '+self.compilerFlags)
      desc.append('  Fortran Linker:         '+self.getLinker())
      desc.append('  Fortran Linker Flags:   '+self.linkerFlags)
      self.popLanguage()
    return '\n'.join(desc)+'\n'

  def configureHelp(self, help):
    import nargs

    help.addArgument('Compilers', '-with-cpp=<prog>', nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-with-cc=<prog>',  nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-with-cxx=<prog>', nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-with-fc=<prog>',  nargs.Arg(None, None, 'Specify the Fortran compiler'))
    help.addArgument('Compilers', '-with-f90-header=<file>', nargs.Arg(None, None, 'Specify the C header for the F90 interface'))
    help.addArgument('Compilers', '-with-f90-source=<file>', nargs.Arg(None, None, 'Specify the C source for the F90 interface'))
    help.addArgument('Compilers', '-with-ld=<prog>',         nargs.Arg(None, None, 'Specify the linker'))

    help.addArgument('Compilers', '-with-gnu-compilers',             nargs.ArgBool(None, 1, 'Try to use GNU compilers'))
    help.addArgument('Compilers', '-with-vendor-compilers=<vendor>', nargs.Arg(None, '', 'Try to use vendor compilers (no argument means all vendors)'))

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
      if not status and (output.find('www.gnu.org') >= 0 or output.find('developer.apple.com') >= 0 or output.find('bugzilla.redhat.com') >= 0):
        return 1
    except Exception:
      pass
    return 0
  isGNU = staticmethod(isGNU)

  def checkCompiler(self, language):
    '''Check that the given compiler is functional, and if not raise an exception'''
    self.pushLanguage(language)
    if not self.checkCompile():
      raise RuntimeError('Cannot compile '+language+' with '+self.compiler+'.')
    if not self.checkLink():
      raise RuntimeError('Cannot link '+language+' with '+self.linker+'.')
    if not self.checkRun():
      raise RuntimeError('Cannot run executables created with '+language+'.')
    self.popLanguage()
    return

  def generateCCompilerGuesses(self):
    '''Determine the C compiler using CC, then --with-cc, then MPI, then GNU, then vendors
       - Any given category can be excluded'''
    import os

    if self.framework.argDB.has_key('CC'):
      yield self.framework.argDB['CC']
      raise RuntimeError('C compiler you provided with -CC='+self.framework.argDB['CC']+' does not work')
    elif self.framework.argDB.has_key('with-cc'):
      yield self.framework.argDB['with-cc']
      raise RuntimeError('C compiler you provided with -with-cc='+self.framework.argDB['with-cc']+' does not work')
    elif self.framework.argDB.has_key('with-mpi-dir') and os.path.isdir(self.framework.argDB['with-mpi-dir']) and self.framework.argDB['with-mpi-compilers']:
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicc')
      raise RuntimeError('bin/mpicc you provided with -with-mpi-dir='+self.framework.argDB['with-mpi-dir']+' does not work')
    else:
      if self.framework.argDB['with-mpi'] and self.framework.argDB['with-mpi-compilers']:
        if Configure.isGNU('mpicc') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpicc'
        if not Configure.isGNU('mpicc') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpicc'
      if self.framework.argDB['with-gnu-compilers']:
        yield 'gcc'
      vendor = self.framework.argDB['with-vendor-compilers']
      if not vendor == '0':
        if not vendor and not Configure.isGNU('cc'):
          yield 'cc'
        if vendor == 'kai' or not vendor:
          yield 'kcc'
        if vendor == 'ibm' or not vendor:
          yield 'xlc'
        if vendor == 'intel' or not vendor:
          yield 'icc'
        if vendor == 'portland' or not vendor:
          yield 'pgcc'
    return

  def checkCCompiler(self):
    '''Locate a functional C compiler'''
    for compiler in self.generateCCompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CC'):
          self.framework.argDB['CC'] = self.CC
          self.checkCompiler('C')
          break
      except RuntimeError, e:
        import os

        if os.path.basename(self.framework.argDB['CC']) == 'mpicc':
          self.framework.log.write(' MPI installation '+self.compiler+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.\n')
        self.popLanguage()
        self.framework.argDB['CC'] = None
    if 'CC' in self.framework.argDB and not self.framework.argDB['CC'] is None:
      self.addSubstitution('CC', self.framework.argDB['CC'])
      self.isGCC = Configure.isGNU(self.framework.argDB['CC'])
    else:
      raise RuntimeError('Could not locate a functional C compiler')
    return

  def generateCPreprocessorGuesses(self):
    '''Determines the C preprocessor from CPP, then --with-cpp, then the C compiler'''
    if self.framework.argDB.has_key('CPP'):
      yield self.framework.argDB['CPP']
    elif self.framework.argDB.has_key('with-cpp'):
      yield self.framework.argDB['with-cpp']
    else:
      yield self.framework.argDB['CC']+' -E'
    return

  def checkCPreprocessor(self):
    '''Locate a functional C preprocessor'''
    for compiler in self.generateCPreprocessorGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CPP'):
          self.framework.argDB['CPP'] = self.CPP
          self.pushLanguage('C')
          if not self.checkPreprocess('#include <stdlib.h>\n'):
            raise RuntimeError('Cannot preprocess C with '+self.CPP+'.')
          self.popLanguage()
          break
      except RuntimeError, e:
        import os

        self.popLanguage()
        self.framework.argDB['CPP'] = None
    if 'CPP' in self.framework.argDB and not self.framework.argDB['CPP'] is None:
      self.addSubstitution('CPP', self.framework.argDB['CPP'])
      self.addSubstitution('CPPFLAGS', self.framework.argDB['CPPFLAGS'])
    return

  def checkCFlags(self):
    '''Try to turn on debugging if no flags are given'''
    if not self.framework.argDB['CFLAGS']:
      self.pushLanguage('C')
      flag = '-g'
      if self.checkCompilerFlag(flag):
        self.framework.argDB['CFLAGS'] = self.framework.argDB['CFLAGS']+' '+flag
      self.popLanguage()
    self.addSubstitution('CFLAGS', self.framework.argDB['CFLAGS'])
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

  def generateCxxCompilerGuesses(self):
    '''Determine the Cxx compiler using CXX, then --with-cxx, then MPI, then GNU, then vendors
       - Any given category can be excluded'''
    import os

    if self.framework.argDB.has_key('CXX'):
      yield self.framework.argDB['CXX']
      raise RuntimeError('C++ compiler you provided with -CXX='+self.framework.argDB['CXX']+' does not work')
    elif self.framework.argDB.has_key('with-cxx'):
      if self.framework.argDB['with-cxx'] == '0':
        return
      else:
        yield self.framework.argDB['with-cxx']
        raise RuntimeError('C++ compiler you provided with -with-cxx='+self.framework.argDB['with-cxx']+' does not work')
    elif self.framework.argDB.has_key('with-mpi-dir') and os.path.isdir(self.framework.argDB['with-mpi-dir']) and self.framework.argDB['with-mpi-compilers']:
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicxx')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpiCC')
      if os.path.isdir(os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicxx')) or os.path.isdir((os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicxx'))):
        raise RuntimeError('bin/mpiCC[cxx] you provided with -with-mpi-dir='+self.framework.argDB['with-mpi-dir']+' does not work\nRun with -with-cxx=0 if you wish to use this MPI and disable C++')
    else:
      if self.framework.argDB['with-mpi'] and self.framework.argDB['with-mpi-compilers']:
        if Configure.isGNU('mpicxx') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpicxx'
        if not Configure.isGNU('mpicxx') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpicxx'
        if Configure.isGNU('mpiCC') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpiCC'
        if not Configure.isGNU('mpiCC') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpiCC'
      if self.framework.argDB['with-gnu-compilers']:
        yield 'g++'
      vendor = self.framework.argDB['with-vendor-compilers']
      if not vendor == '0':
        if not vendor:
          if not Configure.isGNU('c++'):
            yield 'c++'
          if not Configure.isGNU('CC'):
            yield 'CC'
          yield 'cxx'
          yield 'cc++'
        if vendor == 'ibm' or not vendor:
          yield 'xlC'
        if vendor == 'intel' or not vendor:
          yield 'icc -Kc++'
        if vendor == 'microsoft' or not vendor:
          yield 'cl'
        if vendor == 'portland' or not vendor:
          yield 'pgCC'
    return

  def checkCxxCompiler(self):
    '''Locate a functional Cxx compiler'''
    for compiler in self.generateCxxCompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CXX'):
          self.framework.argDB['CXX'] = self.CXX
          self.checkCompiler('Cxx')
          break
      except RuntimeError, e:
        import os

        if os.path.basename(self.framework.argDB['CXX']) in ['mpicxx', 'mpiCC']:
          self.framework.log.write('  MPI installation '+self.compiler+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.\n')
        self.popLanguage()
        self.framework.argDB['CXX'] = None
    if 'CXX' in self.framework.argDB and not self.framework.argDB['CXX'] is None:
      self.addSubstitution('CXX', self.framework.argDB['CXX'])
      self.isGCXX = Configure.isGNU(self.framework.argDB['CXX'])
    else:
      self.addSubstitution('CXX', '')
      self.isGCXX = 0
      self.framework.argDB['CXX'] = None
    return

  def generateCxxPreprocessorGuesses(self):
    '''Determines the Cxx preprocessor from CXXCPP, then --with-cxxcpp, then the Cxx compiler'''
    if self.framework.argDB.has_key('CXXCPP'):
      yield self.framework.argDB['CXXCPP']
    elif self.framework.argDB.has_key('with-cxxcpp'):
      yield self.framework.argDB['with-cxxcpp']
    else:
      yield self.framework.argDB['CXX']+' -E'
    return

  def checkCxxPreprocessor(self):
    '''Locate a functional Cxx preprocessor'''
    if not self.framework.argDB['CXX']:
      self.addSubstitution('CXXCPP', '')
      return
    for compiler in self.generateCxxPreprocessorGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CXXCPP'):
          self.framework.argDB['CXXCPP'] = self.CXXCPP
          self.pushLanguage('Cxx')
          if not self.checkPreprocess('#include <cstdlib>\n'):
            raise RuntimeError('Cannot preprocess Cxx with '+self.CXXCPP+'.')
          self.popLanguage()
          break
      except RuntimeError, e:
        import os

        if os.path.basename(self.framework.argDB['CXXCPP']) in ['mpicxx', 'mpiCC']:
          self.framework.log.write('MPI installation '+self.compiler+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI\n')
        self.popLanguage()
        self.framework.argDB['CXXCPP'] = None
    if 'CXXCPP' in self.framework.argDB and not self.framework.argDB['CXXCPP'] is None:
      self.addSubstitution('CXXCPP', self.framework.argDB['CXXCPP'])
    return

  def checkCxxFlags(self):
    '''Try to turn on debugging if no flags are given'''
    if not self.framework.argDB['CXX']:
      self.addSubstitution('CXXFLAGS', '')
      return
    if not self.framework.argDB['CXXFLAGS']:
      self.pushLanguage('C++')
      flag = '-g'
      if self.checkCompilerFlag(flag):
        self.framework.argDB['CXXFLAGS'] = self.framework.argDB['CXXFLAGS']+' '+flag
      self.popLanguage()
    self.addSubstitution('CXXFLAGS', self.framework.argDB['CXXFLAGS'])
    return

  def checkCxxNamespace(self):
    '''Checks that C++ compiler supports namespaces, and if it does defines HAVE_CXX_NAMESPACE'''
    if not self.framework.argDB['CXX']:
      return
    if not self.framework.argDB.has_key('CXX'): return
    self.pushLanguage('C++')
    if self.checkCompile('namespace petsc {int dummy;}'):
      self.addDefine('HAVE_CXX_NAMESPACE', 1)
    self.popLanguage()
    return

  def generateFortranCompilerGuesses(self):
    '''Determine the Fortran compiler using FC, then --with-fc, then MPI, then GNU, then vendors
       - Any given category can be excluded'''
    import os

    if self.framework.argDB.has_key('FC'):
      if self.framework.argDB['FC'] == '0': return
      yield self.framework.argDB['FC']
      raise RuntimeError('Fortran compiler you provided with -FC='+self.framework.argDB['FC']+' does not work')
    elif self.framework.argDB.has_key('with-fc'):
      if self.framework.argDB['with-fc'] == '0': return
      yield self.framework.argDB['with-fc']
      raise RuntimeError('Fortran compiler you provided with --with-fc='+self.framework.argDB['with-fc']+' does not work')
    elif self.framework.argDB.has_key('with-mpi-dir') and os.path.isdir(self.framework.argDB['with-mpi-dir']) and self.framework.argDB['with-mpi-compilers']:
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif90')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif77')
      if os.path.isdir(os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif90')) or os.path.isdir((os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif77'))):
        raise RuntimeError('bin/mpif90[f77] you provided with -with-mpi-dir='+self.framework.argDB['with-mpi-dir']+' does not work\nRun with -with-fc=0 if you wish to use this MPI and disable Fortran')
    else:
      if self.framework.argDB['with-mpi'] and self.framework.argDB['with-mpi-compilers']:
        if Configure.isGNU('mpif90') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpif90'
        if not Configure.isGNU('mpif90') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpif90'
        if Configure.isGNU('mpif77') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpif7'
        if not Configure.isGNU('mpif77') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpif77'
      if self.framework.argDB['with-gnu-compilers']:
        yield 'g77'
      vendor = self.framework.argDB['with-vendor-compilers']
      if not vendor == '0':
        if not vendor:
          yield 'f90'
        if vendor == 'ibm' or not vendor:
          yield 'xlf90'
          yield 'xlf'
        if vendor == 'intel' or not vendor:
          yield 'icf'
        if vendor == 'portland' or not vendor:
          yield 'pgf90'
          yield 'pgf77'
        if not vendor:
          yield 'f77'
    return

  def checkFortranCompiler(self):
    '''Locate a functional Fortran compiler'''
    for compiler in self.generateFortranCompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'FC'):
          self.framework.argDB['FC'] = self.FC
          self.checkCompiler('F77')
          break
      except RuntimeError, e:
        import os

        if os.path.basename(self.framework.argDB['FC']) in ['mpif90', 'mpif77']:
         self.framework.log.write(' MPI installation '+self.compiler+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.\n')
        self.popLanguage()
        self.framework.argDB['FC'] = None
    if 'FC' in self.framework.argDB and not self.framework.argDB['FC'] is None:
      self.addSubstitution('FC', self.framework.argDB['FC'])
    else:
      self.addSubstitution('FC', '')
    return

  def checkFortranFlags(self):
    '''Try to turn on debugging if no flags are given'''
    if not self.framework.argDB['FFLAGS']:
      self.pushLanguage('F77')
      flag = '-g'
      if self.checkCompilerFlag(flag):
        self.framework.argDB['FFLAGS'] = self.framework.argDB['FFLAGS']+' '+flag
      # see if compiler (ifc) bitches about real*8, if so try using -w90 -w to eliminate bitch
      (output, returnCode) = self.outputCompile('', '      real*8 variable', 1)
      if output.find('Type size specifiers are an extension to standard Fortran 95') >= 0:
        flag = self.framework.argDB['FFLAGS']
        self.framework.argDB['FFLAGS'] += ' -w90 -w'
        (output, returnCode) = self.outputCompile('', '      real*8 variable', 1)
        if returnCode or output.find('Type size specifiers are an extension to standard Fortran 95') >= 0:
          self.framework.argDB['FFLAGS'] = flag          
      self.popLanguage()

    self.addSubstitution('FFLAGS', self.framework.argDB['FFLAGS'])
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
    for lib in flibs:
      if self.slpath and lib.startswith('-L'):
        self.flibs += ' '+self.slpath+lib[2:]
      self.flibs += ' '+lib
    # Append run path
    if ldRunPath: self.flibs = ldRunPath+self.flibs
    
    # check that these monster libraries can be used from C
    oldLibs = self.framework.argDB['LIBS']
    self.framework.argDB['LIBS'] += ' '+self.flibs
    try:
      self.checkCompiler('C')
    except:
      # try removing this one
      self.flibs = re.sub('-lcrt2.o','',self.flibs)
      self.framework.argDB['LIBS'] = oldLibs+self.flibs
      try:
        self.checkCompiler('C')
      except:
        raise RuntimeError('Fortran libraries cannot be used with C compiler')

    # check these monster libraries work from C++
    if self.framework.argDB['CXX']:
      self.framework.argDB['LIBS'] += oldLibs+self.flibs
      try:
        self.checkCompiler('C++')
      except:
        # try removing this one causes grief with gnu g++ and Intel Fortran
        self.flibs = re.sub('-lintrins','',self.flibs)
        self.framework.argDB['LIBS'] = oldLibs+self.flibs
        try:
          self.checkCompiler('C++')
        except:
          raise RuntimeError('Fortran libraries cannot be used with C++ compiler.\n Run with --with-fc=0 or --with-cxx=0')


    self.framework.argDB['LIBS'] = oldLibs
    self.addSubstitution('FLIBS', self.flibs)
    return

  def checkLinkerFlags(self):
    '''Just substitutes the flags right now'''
    self.addSubstitution('LDFLAGS', self.framework.argDB['LDFLAGS'])
    return

  def checkSharedLinkerFlag(self):
    '''Determine what flags are necessary for dynamic library creation'''
    flag = '-shared'
    if not self.checkLinkerFlag(flag):
      flag = '-dylib'
      if not self.checkLinkerFlag(flag):
        flag = ''
    self.addSubstitution('SHARED_LIBRARY_FLAG', flag)
    return

  def checkSharedLinkerPaths(self):
    '''Determine whether the linker accepts the -rpath'''
    flag = '-Wl,-rpath,'
    if not self.checkLinkerFlag(flag):
      flag = ''
    self.addSubstitution('RPATH', flag)
    self.slpath = flag
    return

  def configure(self):
    self.executeTest(self.checkCCompiler)
    self.executeTest(self.checkCPreprocessor)
    self.executeTest(self.checkCFlags)
    self.executeTest(self.checkCRestrict)
    self.executeTest(self.checkCFormatting)

    self.executeTest(self.checkCxxCompiler)
    self.executeTest(self.checkCxxFlags)
    self.executeTest(self.checkCxxNamespace)

    self.executeTest(self.checkSharedLinkerPaths)
    self.executeTest(self.checkFortranCompiler)
    if 'FC' in self.framework.argDB:
      self.executeTest(self.checkFortranFlags)
      self.executeTest(self.checkFortranNameMangling)
    self.executeTest(self.checkFortranLibraries)
    self.executeTest(self.checkFortran90Interface)

    self.executeTest(self.checkLinkerFlags)
    self.executeTest(self.checkSharedLinkerFlag)
    return
