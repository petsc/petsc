from __future__ import generators
import config.base

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
#     desc.append('  C Compiler Flags:   '+self.compilerFlags)
      if not self.getLinker() == self.getCompiler(): desc.append('  C Linker:           '+self.getLinker())
#     desc.append('  C Linker Flags:     '+self.linkerFlags)
      self.popLanguage()
    if 'CXX' in self.framework.argDB:
      self.pushLanguage('Cxx')
      desc.append('  C++ Compiler:       '+self.getCompiler())
#     desc.append('  C++ Compiler Flags: '+self.compilerFlags)
      if not self.getLinker() == self.getCompiler(): desc.append('  C++ Linker:         '+self.getLinker())
#     desc.append('  C++ Linker Flags:   '+self.linkerFlags)
      self.popLanguage()
    if 'FC' in self.framework.argDB:
      self.pushLanguage('F77')
      desc.append('  Fortran Compiler:   '+self.getCompiler())
#     desc.append('  Fortran Compiler Flags: '+self.compilerFlags)
      if not self.getLinker() == self.getCompiler(): desc.append('  Fortran Linker:     '+self.getLinker())
#     desc.append('  Fortran Linker Flags:   '+self.linkerFlags)
      self.popLanguage()
    return '\n'.join(desc)+'\n'

  def configureHelp(self, help):
    import nargs

    help.addArgument('Compilers', '-with-cpp=<prog>', nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-with-cc=<prog>',  nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-with-cxx=<prog>', nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-with-fc=<prog>',  nargs.Arg(None, None, 'Specify the Fortran compiler'))
    help.addArgument('Compilers', '-with-ld=<prog>',  nargs.Arg(None, None, 'Specify the linker'))

    help.addArgument('Compilers', '-with-gnu-compilers',             nargs.ArgBool(None, 1, 'Try to use GNU compilers'))
    help.addArgument('Compilers', '-with-vendor-compilers=<vendor>', nargs.Arg(None, '', 'Try to use vendor compilers (no argument means all vendors)'))
    help.addArgument('Compilers', '-with-64-bit',                    nargs.ArgBool(None, 0, 'Use 64 bit compilers and libraries'))

    help.addArgument('Compilers', '-CPP=<prog>',            nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-CPPFLAGS=<string>',     nargs.Arg(None, '',   'Specify the C preprocessor options'))
    help.addArgument('Compilers', '-CXXPP=<prog>',          nargs.Arg(None, None, 'Specify the C++ preprocessor'))
    help.addArgument('Compilers', '-CC=<prog>',             nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-CFLAGS=<string>',       nargs.Arg(None, '',   'Specify the C compiler options'))
    help.addArgument('Compilers', '-CXX=<prog>',            nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-CXXFLAGS=<string>',     nargs.Arg(None, '',   'Specify the C++ compiler options'))
    help.addArgument('Compilers', '-CXX_CXXFLAGS=<string>', nargs.Arg(None, '',   'Specify the C++ compiler-only options'))
    help.addArgument('Compilers', '-FC=<prog>',             nargs.Arg(None, None, 'Specify the Fortran compiler'))
    help.addArgument('Compilers', '-FFLAGS=<string>',       nargs.Arg(None, '',   'Specify the Fortran compiler options'))

    help.addArgument('Compilers', '-LD=<prog>',         nargs.Arg(None, None, 'Specify the default linker'))
    help.addArgument('Compilers', '-CC_LD=<prog>',      nargs.Arg(None, None, 'Specify the linker for C only'))
    help.addArgument('Compilers', '-CXX_LD=<prog>',     nargs.Arg(None, None, 'Specify the linker for C++ only'))
    help.addArgument('Compilers', '-FC_LD=<prog>',      nargs.Arg(None, None, 'Specify the linker for Fortran only'))
    help.addArgument('Compilers', '-LDFLAGS=<string>',  nargs.Arg(None, '',   'Specify the linker options'))
    return

  def isGNU(compiler):
    '''Returns true if the compiler is a GNU compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help')
      if output.find('www.gnu.org') >= 0 or output.find('developer.apple.com') >= 0 or output.find('bugzilla.redhat.com') >= 0:
        return 1
    except RuntimeError:
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
    elif self.framework.argDB.has_key('with-mpi-dir') and os.path.isdir(self.framework.argDB['with-mpi-dir']) and self.framework.argDB['with-mpi-compilers'] and not self.framework.argDB['with-mpich']:
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicc')
      raise RuntimeError('bin/mpicc you provided with -with-mpi-dir='+self.framework.argDB['with-mpi-dir']+' does not work')
    else:
      if 'with-mpi' in self.framework.argDB and self.framework.argDB['with-mpi'] and self.framework.argDB['with-mpi-compilers']  and not self.framework.argDB['with-mpich']:
        if Configure.isGNU('mpicc') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpicc'
        if not Configure.isGNU('mpicc') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpicc'
        if not Configure.isGNU('mpxlc') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpxlc'
      vendor = self.framework.argDB['with-vendor-compilers']
      if (not vendor or vendor == '0') and self.framework.argDB['with-gnu-compilers']:
        yield 'gcc'
      if not vendor == '0':
        if not vendor and not Configure.isGNU('cc'):
          yield 'cc'
        if vendor == 'borland' or not vendor:
          yield 'win32fe bcc32'
        if vendor == 'kai' or not vendor:
          yield 'kcc'
        if vendor == 'ibm' or not vendor:
          yield 'xlc'
        if vendor == 'intel' or not vendor:
          yield 'icc'
          yield 'win32fe icl'
        if vendor == 'microsoft' or not vendor:
          yield 'win32fe cl'
        if vendor == 'portland' or not vendor:
          yield 'pgcc'
      if self.framework.argDB['with-gnu-compilers']:
        yield 'gcc'
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
        del self.framework.argDB['CC']
    if 'CC' in self.framework.argDB:
      self.addArgumentSubstitution('CC', 'CC')
      self.addArgumentSubstitution('CFLAGS', 'CFLAGS')
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
        del self.framework.argDB['CPP']
    if 'CPP' in self.framework.argDB:
      self.addArgumentSubstitution('CPP', 'CPP')
      self.addArgumentSubstitution('CPPFLAGS', 'CPPFLAGS')
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
    elif self.framework.argDB.has_key('with-mpi-dir') and os.path.isdir(self.framework.argDB['with-mpi-dir']) and self.framework.argDB['with-mpi-compilers']  and not self.framework.argDB['with-mpich']:
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicxx')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpiCC')
      if os.path.isdir(os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicxx')) or os.path.isdir((os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicxx'))):
        raise RuntimeError('bin/mpiCC[cxx] you provided with -with-mpi-dir='+self.framework.argDB['with-mpi-dir']+' does not work\nRun with -with-cxx=0 if you wish to use this MPI and disable C++')
    else:
      if 'with-mpi' in self.framework.argDB and self.framework.argDB['with-mpi'] and self.framework.argDB['with-mpi-compilers'] and not self.framework.argDB['with-mpich']:
        if Configure.isGNU('mpicxx') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpicxx'
        if not Configure.isGNU('mpicxx') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpicxx'
        if Configure.isGNU('mpiCC') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpiCC'
        if not Configure.isGNU('mpiCC') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpiCC'
        if Configure.isGNU('mpic++') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpic++'
        if not Configure.isGNU('mpic++') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpic++'
        if not Configure.isGNU('mpCC') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpCC'
      vendor = self.framework.argDB['with-vendor-compilers']
      if (not vendor or vendor == '0') and self.framework.argDB['with-gnu-compilers']:
        yield 'g++'
      if not vendor == '0':
        if not vendor:
          if not Configure.isGNU('c++'):
            yield 'c++'
          if not Configure.isGNU('CC'):
            yield 'CC'
          yield 'cxx'
          yield 'cc++'
        if vendor == 'borland' or not vendor:
          yield 'win32fe bcc32'
        if vendor == 'ibm' or not vendor:
          yield 'xlC'
        if vendor == 'intel' or not vendor:
          yield 'icc'
          yield 'win32fe icl'
        if vendor == 'microsoft' or not vendor:
          yield 'cl'
        if vendor == 'portland' or not vendor:
          yield 'pgCC'
      if self.framework.argDB['with-gnu-compilers']:
        yield 'g++'
    return

  def checkCxxCompiler(self):
    '''Locate a functional Cxx compiler'''
    for compiler in self.generateCxxCompilerGuesses():
      # Determine an acceptable extensions for the C++ compiler
      for ext in ['.cc', '.cpp', '.C']:
        self.framework.cxxExt = ext
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
          self.framework.cxxExt = None
          del self.framework.argDB['CXX']
      if 'CXX' in self.framework.argDB:
        break
    if 'CXX' in self.framework.argDB:
      self.addArgumentSubstitution('CXX', 'CXX')
      self.addArgumentSubstitution('CXX_CXXFLAGS', 'CXX_CXXFLAGS')
      self.addArgumentSubstitution('CXXFLAGS', 'CXXFLAGS')
      self.isGCXX = Configure.isGNU(self.framework.argDB['CXX'])
    else:
      self.addSubstitution('CXX', '')
      self.isGCXX = 0
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
    if not 'CXX' in self.framework.argDB:
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
      self.addArgumentSubstitution('CXXCPP', 'CXXCPP')
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
    elif self.framework.argDB.has_key('with-mpi-dir') and os.path.isdir(self.framework.argDB['with-mpi-dir']) and self.framework.argDB['with-mpi-compilers'] and not self.framework.argDB['with-mpich']:
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif90')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif77')
      if os.path.isdir(os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif90')) or os.path.isdir((os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif77'))):
        raise RuntimeError('bin/mpif90[f77] you provided with -with-mpi-dir='+self.framework.argDB['with-mpi-dir']+' does not work\nRun with -with-fc=0 if you wish to use this MPI and disable Fortran')
    else:
      if 'with-mpi' in self.framework.argDB and self.framework.argDB['with-mpi'] and self.framework.argDB['with-mpi-compilers'] and not self.framework.argDB['with-mpich']:
        if Configure.isGNU('mpif90') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpif90'
        if not Configure.isGNU('mpif90') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpif90'
        if Configure.isGNU('mpif77') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpif7'
        if not Configure.isGNU('mpif77') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpif77'
        if not Configure.isGNU('mpxlf') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpxlf'
      vendor = self.framework.argDB['with-vendor-compilers']
      if (not vendor or vendor == '0') and self.framework.argDB['with-gnu-compilers']:
        yield 'g77'
      if not vendor == '0':
        if not vendor:
          yield 'f90'
        if vendor == 'ibm' or not vendor:
          yield 'xlf90'
          yield 'xlf'
        if vendor == 'intel' or not vendor:
          yield 'win32fe ifort'
          yield 'win32fe ifl'
          yield 'ifort'
          yield 'icf'
        if vendor == 'portland' or not vendor:
          yield 'pgf90'
          yield 'pgf77'
        if not vendor:
          yield 'f77'
      if self.framework.argDB['with-gnu-compilers']:
        yield 'g77'
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
      self.addArgumentSubstitution('FC', 'FC')
      self.addArgumentSubstitution('FFLAGS', 'FFLAGS')
    else:
      self.addSubstitution('FC', '')
    return

  def checkLinkerFlags(self):
    '''Just substitutes the flags right now'''
    self.addArgumentSubstitution('LDFLAGS', 'LDFLAGS')
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
    if not self.checkLinkerFlag(flag+'`pwd`'):
      flag = ''
    self.addSubstitution('RPATH', flag)
    self.slpath = flag
    return

  def configure(self):
    self.executeTest(self.checkCCompiler)
    self.executeTest(self.checkCPreprocessor)
    self.executeTest(self.checkCxxCompiler)
    self.executeTest(self.checkFortranCompiler)
    self.executeTest(self.checkLinkerFlags)
    # In the future, we will check that shared and dyanmic linking works here
    #   Right now we are letting self.slpath leak out to config.compilers
    self.executeTest(self.checkSharedLinkerFlag)
    self.executeTest(self.checkSharedLinkerPaths)
    return
