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
      desc.append('  C Compiler:         '+self.getCompiler()+' '+self.getCompilerFlags())
      if not self.getLinker() == self.getCompiler(): desc.append('  C Linker:           '+self.getLinker()+' '+self.getLinkerFlags())
      self.popLanguage()
    if 'CXX' in self.framework.argDB:
      self.pushLanguage('Cxx')
      desc.append('  C++ Compiler:       '+self.getCompiler()+' '+self.getCompilerFlags())
      if not self.getLinker() == self.getCompiler(): desc.append('  C++ Linker:         '+self.getLinker()+' '+self.getLinkerFlags())
      self.popLanguage()
    if 'FC' in self.framework.argDB:
      self.pushLanguage('FC')
      desc.append('  Fortran Compiler:   '+self.getCompiler()+' '+self.getCompilerFlags())
      if not self.getLinker() == self.getCompiler(): desc.append('  Fortran Linker:     '+self.getLinker()+' '+self.getLinkerFlags())
      self.popLanguage()
    return '\n'.join(desc)+'\n'

  def setupHelp(self, help):
    import nargs

    help.addArgument('Compilers', '-with-cpp=<prog>', nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-with-cc=<prog>',  nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-with-cxx=<prog>', nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-with-fc=<prog>',  nargs.Arg(None, None, 'Specify the Fortran compiler'))

    help.addArgument('Compilers', '-with-gnu-compilers=<bool>',      nargs.ArgBool(None, 1, 'Try to use GNU compilers'))
    help.addArgument('Compilers', '-with-vendor-compilers=<vendor>', nargs.Arg(None, '', 'Try to use vendor compilers (no argument all vendors, 0 no vendors)'))
    help.addArgument('Compilers', '-with-64-bit-pointers=<bool>',    nargs.ArgBool(None, 0, 'Use 64 bit compilers and libraries'))

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

    help.addArgument('Compilers', '-LD=<prog>',              nargs.Arg(None, None, 'Specify the default linker'))
    help.addArgument('Compilers', '-CC_LD=<prog>',           nargs.Arg(None, None, 'Specify the linker for C only'))
    help.addArgument('Compilers', '-CXX_LD=<prog>',          nargs.Arg(None, None, 'Specify the linker for C++ only'))
    help.addArgument('Compilers', '-FC_LD=<prog>',           nargs.Arg(None, None, 'Specify the linker for Fortran only'))
    help.addArgument('Compilers', '-LDFLAGS=<string>',       nargs.Arg(None, '',   'Specify the linker options'))
    help.addArgument('Compilers', '-with-ar',                nargs.Arg(None, None,   'Specify the archiver'))
    help.addArgument('Compilers', 'AR',                      nargs.Arg(None, None,   'Specify the archiver flags'))
    help.addArgument('Compilers', 'AR_FLAGS',                nargs.Arg(None, None,   'Specify the archiver flags'))
    help.addArgument('Compilers', '-with-ranlib',            nargs.Arg(None, None,   'Specify ranlib'))
    help.addArgument('Compilers', '-with-shared',            nargs.ArgBool(None, 1, 'Enable shared libraries'))
    help.addArgument('Compilers', '-with-shared-ld=<prog>',  nargs.Arg(None, None, 'Specify the shared linker'))
    return

  def isGNU(compiler):
    '''Returns true if the compiler is a GNU compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help')
      output = output + error
      if output.find('www.gnu.org') >= 0 or output.find('developer.apple.com') >= 0 or output.find('bugzilla.redhat.com') >= 0 or output.find('gcc.gnu.org') >= 0 or (output.find('gcc version')>=0 and not output.find('Intel(R)')>= 0):
        return 1
    except RuntimeError:
      pass
    return 0
  isGNU = staticmethod(isGNU)

  def checkCompiler(self, language):
    '''Check that the given compiler is functional, and if not raise an exception'''
    self.pushLanguage(language)
    if not self.checkCompile():
      raise RuntimeError('Cannot compile '+language+' with '+self.getCompiler()+'.')
    if not self.checkLink():
      raise RuntimeError('Cannot compile/link '+language+' with '+self.getCompiler()+'.')
    if self.framework.argDB['can-execute']:
      if not self.checkRun():
        raise RuntimeError('Cannot run executables created with '+language+'.')
    self.popLanguage()
    return

  def generateCCompilerGuesses(self):
    '''Determine the C compiler using CC, then --with-cc, then MPI, then GNU, then vendors
       - Any given category can be excluded'''
    import os

    if self.framework.argDB['with-vendor-compilers'] == 'no': self.framework.argDB['with-vendor-compilers'] = '0'
    if self.framework.argDB['with-vendor-compilers'] == 'yes': self.framework.argDB['with-vendor-compilers'] = ''      
    if self.framework.argDB['with-vendor-compilers'] == 'false': self.framework.argDB['with-vendor-compilers'] = '0'
    if self.framework.argDB['with-vendor-compilers'] == 'true': self.framework.argDB['with-vendor-compilers'] = ''      

    if 'PETSC_DIR' in self.framework.argDB:
      self.framework.argDB['search-dirs'].append(os.path.join(self.framework.argDB['PETSC_DIR'],'bin','win32fe'))
        
    if self.framework.argDB.has_key('with-cc'):
      if self.framework.argDB['with-cc'] in ['icl','cl','bcc32']: self.framework.argDB['with-cc'] = 'win32fe '+self.framework.argDB['with-cc']
      yield self.framework.argDB['with-cc']
      raise RuntimeError('C compiler you provided with -with-cc='+self.framework.argDB['with-cc']+' does not work')
    elif self.framework.argDB.has_key('CC'):
      if 'CC' in os.environ and os.environ['CC'] == self.framework.argDB['CC']:
        self.startLine()
        print '\n*****WARNING: Using C compiler '+self.framework.argDB['CC']+' from environmental variable CC****\nAre you sure this is what you want? If not, unset that environmental variable and run configure again'
      if self.framework.argDB['CC'] in ['icl','cl','bcc32']: self.framework.argDB['CC'] = 'win32fe '+self.framework.argDB['CC']
      yield self.framework.argDB['CC']
      raise RuntimeError('C compiler you provided with -CC='+self.framework.argDB['CC']+' does not work')
    elif self.framework.argDB.has_key('with-mpi-dir') and os.path.isdir(os.path.join(self.framework.argDB['with-mpi-dir'],'bin')) and self.framework.argDB['with-mpi-compilers'] and not self.framework.argDB['download-mpich'] == 1 and self.framework.argDB['with-mpi']:
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicc')
      raise RuntimeError('bin/mpicc you provided with -with-mpi-dir='+self.framework.argDB['with-mpi-dir']+' does not work')
    else:
      if 'with-mpi' in self.framework.argDB and self.framework.argDB['with-mpi'] and self.framework.argDB['with-mpi-compilers']  and not self.framework.argDB['download-mpich'] == 1:
        if Configure.isGNU('mpicc') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpicc'
        if not Configure.isGNU('mpicc') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpicc'
        if not self.framework.argDB['with-vendor-compilers'] == '0':
          yield 'mpcc_r'
          yield 'mpcc'
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
          yield 'ecc'          
          yield 'win32fe icl'
        if vendor == 'microsoft' or not vendor:
          yield 'win32fe cl'
        if vendor == 'portland' or not vendor:
          yield 'pgcc'
        if vendor == 'solaris' or not vendor:
          if not Configure.isGNU('cc'):
            yield 'cc'
      if self.framework.argDB['with-gnu-compilers']:
        yield 'gcc'
    return

  def checkCCompiler(self):
    '''Locate a functional C compiler'''
    if 'with-cc' in self.framework.argDB and self.framework.argDB['with-cc'] == '0':
      raise RuntimeError('A functional C compiler is necessary for configure')
    for compiler in self.generateCCompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CC'):
          self.framework.argDB['CC'] = self.CC
          self.checkCompiler('C')
          if self.framework.argDB['with-64-bit-pointers']:
            if Configure.isGNU(self.CC):
              raise RuntimeError('Cannot handle 64 bit with gnu compilers yet')
            else:
              if self.framework.argDB['PETSC_ARCH_BASE'].startswith('solaris'):
                self.pushLanguage('C')
                self.addCompilerFlag('-xarch=v9')
                self.popLanguage()
          break
      except RuntimeError, e:
        import os

        if os.path.basename(self.framework.argDB['CC']) == 'mpicc':
          self.framework.logPrint(' MPI installation '+self.getCompiler()+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
        del self.framework.argDB['CC']
    if not 'CC' in self.framework.argDB:
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
      yield self.framework.argDB['CC']+' --use cpp32'
    return

  def checkCPreprocessor(self):
    '''Locate a functional C preprocessor'''
    for compiler in self.generateCPreprocessorGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CPP'):
          self.framework.argDB['CPP'] = self.CPP
          self.CPPFLAGS = self.framework.argDB['CPPFLAGS']
          self.pushLanguage('C')
          if not self.checkPreprocess('#include <stdlib.h>\n'):
            raise RuntimeError('Cannot preprocess C with '+self.CPP+'.')
          self.popLanguage()
          break
      except RuntimeError, e:
        import os

        self.popLanguage()
        del self.framework.argDB['CPP']
    return

  def generateCxxCompilerGuesses(self):
    '''Determine the Cxx compiler using CXX, then --with-cxx, then MPI, then GNU, then vendors
       - Any given category can be excluded'''
    import os

    if self.framework.argDB.has_key('with-c++'):
      raise RuntimeError('Keyword --with-c++ is WRONG, use --with-cxx')
    if self.framework.argDB.has_key('with-CC'):
      raise RuntimeError('Keyword --with-CC is WRONG, use --with-cxx')
    
    if self.framework.argDB.has_key('with-cxx'):
      if self.framework.argDB['with-cxx'] in ['icl','cl','bcc32']: self.framework.argDB['with-cxx'] = 'win32fe '+self.framework.argDB['with-cxx']
      yield self.framework.argDB['with-cxx']
      raise RuntimeError('C++ compiler you provided with -with-cxx='+self.framework.argDB['with-cxx']+' does not work')
    elif self.framework.argDB.has_key('CXX'):
      if 'CXX' in os.environ and os.environ['CXX'] == self.framework.argDB['CXX']:
        self.startLine()
        print '\n*****WARNING: Using C++ compiler '+self.framework.argDB['CXX']+' from environmental variable CXX****\nAre you sure this is what you want? If not, unset that environmental variable and run configure again'
      if self.framework.argDB['CXX'] in ['icl','cl','bcc32']: self.framework.argDB['CXX'] = 'win32fe '+self.framework.argDB['CXX']
      yield self.framework.argDB['CXX']
      raise RuntimeError('C++ compiler you provided with -CXX='+self.framework.argDB['CXX']+' does not work')
    elif self.framework.argDB.has_key('with-mpi-dir') and os.path.isdir(os.path.join(self.framework.argDB['with-mpi-dir'],'bin')) and self.framework.argDB['with-mpi-compilers']  and not self.framework.argDB['download-mpich'] == 1 and self.framework.argDB['with-mpi']:
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicxx')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpiCC')
      if os.path.isfile(os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicxx')) or os.path.isfile((os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpiCC'))):
        raise RuntimeError('bin/mpiCC[cxx] you provided with -with-mpi-dir='+self.framework.argDB['with-mpi-dir']+' does not work\nRun with -with-cxx=0 if you wish to use this MPI and disable C++')
    else:
      if 'with-mpi' in self.framework.argDB and self.framework.argDB['with-mpi'] and self.framework.argDB['with-mpi-compilers'] and not self.framework.argDB['download-mpich'] == 1:
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
        if not self.framework.argDB['with-vendor-compilers'] == '0':
          yield 'mpCC_r'
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
          yield 'ecc'          
          yield 'win32fe icl'
        if vendor == 'microsoft' or not vendor:
          yield 'win32fe cl'
        if vendor == 'portland' or not vendor:
          yield 'pgCC'
        if vendor == 'solaris':
          yield 'CC'
      if self.framework.argDB['with-gnu-compilers']:
        yield 'g++'
    return

  def checkCxxCompiler(self):
    '''Locate a functional Cxx compiler'''
    if 'with-cxx' in self.framework.argDB and self.framework.argDB['with-cxx'] == '0':
      if 'CXX' in self.framework.argDB:
        del self.framework.argDB['CXX']
      return
    for compiler in self.generateCxxCompilerGuesses():
      # Determine an acceptable extensions for the C++ compiler
      for ext in ['.cc', '.cpp', '.C']:
        self.framework.getCompilerObject('C++').sourceExtension = ext
        try:
          if self.getExecutable(compiler, resultName = 'CXX'):
            self.framework.argDB['CXX'] = self.CXX
            self.checkCompiler('Cxx')
            if self.framework.argDB['with-64-bit-pointers']:
              if Configure.isGNU(self.CC):
                raise RuntimeError('Cannot handle 64 bit with gnu compilers yet')
              else:
                if self.framework.argDB['PETSC_ARCH_BASE'].startswith('solaris'):
                  self.pushLanguage('C++')
                  self.addCompilerFlag('-xarch=v9')
                  self.popLanguage()
            break
        except RuntimeError, e:
          import os

          if os.path.basename(self.framework.argDB['CXX']) in ['mpicxx', 'mpiCC']:
            self.framework.logPrint('  MPI installation '+self.getCompiler()+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
          self.popLanguage()
          del self.framework.argDB['CXX']
      if 'CXX' in self.framework.argDB:
        break
    return

  def generateCxxPreprocessorGuesses(self):
    '''Determines the Cxx preprocessor from CXXCPP, then --with-cxxcpp, then the Cxx compiler'''
    if self.framework.argDB.has_key('CXXCPP'):
      yield self.framework.argDB['CXXCPP']
    elif self.framework.argDB.has_key('with-cxxcpp'):
      yield self.framework.argDB['with-cxxcpp']
    else:
      yield self.framework.argDB['CXX']+' -E'
      yield self.framework.argDB['CXX']+' --use cpp32'
    return

  def checkCxxPreprocessor(self):
    '''Locate a functional Cxx preprocessor'''
    if not 'CXX' in self.framework.argDB: return
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
          self.framework.logPrint('MPI installation '+self.getCompiler()+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI')
        self.popLanguage()
        del self.framework.argDB['CXXCPP']
    return

  def generateFortranCompilerGuesses(self):
    '''Determine the Fortran compiler using FC, then --with-fc, then MPI, then GNU, then vendors
       - Any given category can be excluded'''
    import os

    if self.framework.argDB.has_key('with-fc'):
      if self.framework.argDB['with-fc'] in ['ifl','df']:
        self.framework.argDB['with-fc'] = 'win32fe '+self.framework.argDB['with-fc']
      if self.framework.argDB['with-fc'] in ['ifort','f90'] and self.framework.argDB['PETSC_ARCH_BASE'].startswith('cygwin'): self.framework.argDB['with-fc'] = 'win32fe '+self.framework.argDB['with-fc']
      yield self.framework.argDB['with-fc']
      raise RuntimeError('Fortran compiler you provided with --with-fc='+self.framework.argDB['with-fc']+' does not work')
    elif self.framework.argDB.has_key('FC'):
      if 'FC' in os.environ and os.environ['FC'] == self.framework.argDB['FC']:
        self.startLine()
        print '\n*****WARNING: Using Fortran compiler '+self.framework.argDB['FC']+' from environmental variable FC****\nAre you sure this is what you want? If not, unset that environmental variable and run configure again'
      if self.framework.argDB['FC'] in ['ifl','df']:
        self.framework.argDB['FC'] = 'win32fe '+self.framework.argDB['FC']
      if self.framework.argDB['FC'] in ['ifort','f90'] and self.framework.argDB['PETSC_ARCH_BASE'].startswith('cygwin'):
        self.framework.argDB['FC'] = 'win32fe '+self.framework.argDB['FC']
      yield self.framework.argDB['FC']
      raise RuntimeError('Fortran compiler you provided with -FC='+self.framework.argDB['FC']+' does not work')
    elif self.framework.argDB.has_key('with-mpi-dir') and os.path.isdir(os.path.join(self.framework.argDB['with-mpi-dir'],'bin')) and self.framework.argDB['with-mpi-compilers'] and not self.framework.argDB['download-mpich'] == 1 and self.framework.argDB['with-mpi']:
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif90')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif77')
      if os.path.isfile(os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif90')) or os.path.isfile((os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif77'))):
        raise RuntimeError('bin/mpif90[f77] you provided with --with-mpi-dir='+self.framework.argDB['with-mpi-dir']+' does not work\nRun with --with-fc=0 if you wish to use this MPI and disable Fortran')
    else:
      if 'with-mpi' in self.framework.argDB and self.framework.argDB['with-mpi'] and self.framework.argDB['with-mpi-compilers'] and not self.framework.argDB['download-mpich'] == 1:
        if Configure.isGNU('mpif90') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpif90'
        if not Configure.isGNU('mpif90') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpif90'
        if Configure.isGNU('mpif77') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpif77'
        if not Configure.isGNU('mpif77') and (not self.framework.argDB['with-vendor-compilers'] == '0'):
          yield 'mpif77'
        if not self.framework.argDB['with-vendor-compilers'] == '0':
          yield 'mpxlf_r'
          yield 'mpxlf'          
      vendor = self.framework.argDB['with-vendor-compilers']
      if (not vendor or vendor == '0') and self.framework.argDB['with-gnu-compilers']:
        yield 'g77'
      if not vendor == '0':
        if vendor == 'ibm' or not vendor:
          yield 'xlf'
          yield 'xlf90'
        if not vendor:
          yield 'f90'
        if vendor == 'lahaye' or not vendor:
          yield 'lf95'
        if vendor == 'intel' or not vendor:
          yield 'win32fe ifort'
          yield 'win32fe ifl'
          yield 'ifort'
          yield 'ifc'
          yield 'efc'          
        if vendor == 'portland' or not vendor:
          yield 'pgf90'
          yield 'pgf77'
        if vendor == 'solaris' or not vendor:
          yield 'f95'
          yield 'f90'
          if not Configure.isGNU('f77'):
            yield 'f77'
      if self.framework.argDB['with-gnu-compilers']:
        yield 'g77'
    return

  def checkFortranCompiler(self):
    '''Locate a functional Fortran compiler'''
    if 'with-fc' in self.framework.argDB and self.framework.argDB['with-fc'] == '0':
      if 'FC' in self.framework.argDB:
        del self.framework.argDB['FC']
      return
    for compiler in self.generateFortranCompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'FC'):
          self.framework.argDB['FC'] = self.FC
          self.checkCompiler('FC')
          if self.framework.argDB['with-64-bit-pointers']:
            if Configure.isGNU(self.CC):
              raise RuntimeError('Cannot handle 64 bit with gnu compilers yet')
            else:
              if self.framework.argDB['PETSC_ARCH_BASE'].startswith('solaris'):
                self.pushLanguage('FC')
                self.addCompilerFlag('-xarch=v9')
                self.popLanguage()
          break
      except RuntimeError, e:
        import os

        if os.path.basename(self.framework.argDB['FC']) in ['mpif90', 'mpif77']:
         self.framework.logPrint(' MPI installation '+self.getCompiler()+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
        self.popLanguage()
        del self.framework.argDB['FC']
    return

  def checkPIC(self):
    '''Determine the PIC option for each compiler
       - There needs to be a test that checks that the functionality is actually working'''
    # Instead of this, I need to add a link check
    #
    #if self.framework.argDB['PETSC_ARCH'].startswith('hpux') and not self.setCompilers.isGNU(self.framework.argDB['CC']):
    #  return
    #
    # Borland compiler accepts -PIC as -P, which means compile as C++ code,
    # Using this will force compilation, not linking when used as a link flag!!!
    # Since this is not what we intend with -PIC, just kick out if using borland.
    if self.CC == 'win32fe bcc32':
      return
    if not self.framework.argDB['with-shared']:
      self.framework.logPrint("Skipping checking PIC options since shared libraries are turned off")
      return
    languages = ['C']
    if 'CXX' in self.framework.argDB:
      languages.append('C++')
    if 'FC' in self.framework.argDB:
      languages.append('FC')
    for language in languages:
      self.pushLanguage(language)
      for testFlag in ['-PIC', '-fPIC', '-KPIC']:
        try:
          self.framework.logPrint('Trying '+language+' compiler flag '+testFlag)
          if not self.checkLinkerFlag(testFlag):
            self.framework.logPrint('Rejected '+language+' compiler flag '+testFlag+' because linker cannot handle it')
            continue
          self.addCompilerFlag(testFlag, compilerOnly = 1)
          break
        except RuntimeError:
          self.framework.logPrint('Rejected '+language+' compiler flag '+testFlag)
      self.popLanguage()
    return

  def generateArchiverFlags(self,archiver):
    flag = ''
    if 'AR_FLAGS' in self.framework.argDB: flag = self.framework.argDB['AR_FLAGS']
    elif os.path.basename(archiver) == 'ar': flag = 'cr'
    elif archiver == 'win32fe lib': flag = '-a'
    elif archiver == 'win32fe tlib': flag = '-a -P512'
    return flag
  
  def generateArchiverGuesses(self):
    if 'with-ar' in self.framework.argDB:
      if self.framework.argDB['with-ar'] in ['lib','tlib']:
        self.framework.argDB['with-ar'] = 'win32fe '+self.framework.argDB['with-ar']
    if 'AR' in self.framework.argDB:
      if self.framework.argDB['AR'] in ['lib','tlib']:
        self.framework.argDB['AR'] = 'win32fe '+self.framework.argDB['AR']
    #if anyone has a better idea about doing these checks, i'm all for it
    if 'with-ar' in self.framework.argDB and 'with-ranlib' in self.framework.argDB:
      yield(self.framework.argDB['with-ar'],self.generateArchiverFlags(self.framework.argDB['with-ar']),self.framework.argDB['with-ranlib'])
      raise RuntimeError('The archiver set --with-ar="'+self.framework.argDB['with-ar']+'" is incompatible with the ranlib set --with-ranlib="'+self.framework.argDB['with-ranlib']+'".')
    if 'with-ar' in self.framework.argDB and 'RANLIB' in self.framework.argDB:
      yield(self.framework.argDB['with-ar'],self.generateArchiverFlags(self.framework.argDB['with-ar']),self.framework.argDB['RANLIB'])
      raise RuntimeError('The archiver set --with-ar="'+self.framework.argDB['with-ar']+'" is incompatible with the ranlib set (perhaps in your environment) -RANLIB="'+self.framework.argDB['RANLIB']+'".')
    if 'AR' in self.framework.argDB and 'with-ranlib' in self.framework.argDB:
      yield(self.framework.argDB['AR'],self.generateArchiverFlags(self.framework.argDB['AR']),self.framework.argDB['with-ranlib'])
      raise RuntimeError('The archiver set --AR="'+self.framework.argDB['AR']+'" is incompatible with the ranlib set --with-ranlib="'+self.framework.argDB['with-ranlib']+'".')
    if 'AR' in self.framework.argDB and 'RANLIB' in self.framework.argDB:
      yield(self.framework.argDB['AR'],self.generateArchiverFlags(self.framework.argDB['AR']),self.framework.argDB['RANLIB'])
      raise RuntimeError('The archiver set --AR="'+self.framework.argDB['AR']+'" is incompatible with the ranlib set (perhaps in your environment) -RANLIB="'+self.framework.argDB['RANLIB']+'".')
    if 'with-ar' in self.framework.argDB:
      yield (self.framework.argDB['with-ar'],self.generateArchiverFlags(self.framework.argDB['with-ar']),'ranlib')
      yield (self.framework.argDB['with-ar'],self.generateArchiverFlags(self.framework.argDB['with-ar']),'true')
      raise RuntimeError('You set a value for --with-ar='+self.framework.argDB['with-ar']+'", but '+self.framework.argDB['with-ar']+' cannot be used\n')
    if 'AR' in self.framework.argDB:
      yield (self.framework.argDB['AR'],self.generateArchiverFlags(self.framework.argDB['AR']),'ranlib')
      yield (self.framework.argDB['AR'],self.generateArchiverFlags(self.framework.argDB['AR']),'true')
      raise RuntimeError('You set a value for -AR="'+self.framework.argDB['AR']+'" (perhaps in your environment), but '+self.framework.argDB['AR']+' cannot be used\n')
    if 'with-ranlib' in self.framework.argDB:
      yield ('ar',self.generateArchiverFlags('ar'),self.framework.argDB['with-ranlib'])
      yield ('win32fe tlib',self.generateArchiverFlags('win32fe tlib'),self.framework.argDB['with-ranlib'])
      yield ('win32fe lib',self.generateArchiverFlags('win32fe lib'),self.framework.argDB['with-ranlib'])
      raise RuntimeError('You set --with-ranlib="'+self.framework.argDB['with-ranlib']+'", but '+self.framework.argDB['with-ranlib']+' cannot be used\n')
    if 'RANLIB' in self.framework.argDB:
      yield ('ar',self.generateArchiverFlags('ar'),self.framework.argDB['RANLIB'])
      yield ('win32fe tlib',self.generateArchiverFlags('win32fe tlib'),self.framework.argDB['RANLIB'])
      yield ('win32fe lib',self.generateArchiverFlags('win32fe lib'),self.framework.argDB['RANLIB'])
      raise RuntimeError('You set -RANLIB="'+self.framework.argDB['RANLIB']+'" (perhaps in your environment), but '+self.framework.argDB['with-ranlib']+' cannot be used\n')
    yield ('ar',self.generateArchiverFlags('ar'),'ranlib')
    yield ('ar',self.generateArchiverFlags('ar'),'true')
    yield ('win32fe tlib',self.generateArchiverFlags('win32fe tlib'),'true')
    yield ('win32fe lib',self.generateArchiverFlags('win32fe lib'),'true')
    return
  
  def checkArchiver(self):
    '''Check that the archiver exists and can make a library usable by the compiler'''
    def checkArchive(command, status, output, error):
      if error or status:
        self.framework.logPrint('Possible ERROR while running archiver: '+output)
        if status: self.framework.logPrint('ret = '+str(status))
        if error: self.framework.logPrint('error message = {'+error+'}')
        os.remove('conf1.o')
        raise RuntimeError('Archiver is not functional')
      return
    def checkRanlib(command, status, output, error):
      if error or status:
        self.framework.logPrint('Possible ERROR while running ranlib: '+output)
        if status: self.framework.logPrint('ret = '+str(status))
        if error: self.framework.logPrint('error message = {'+error+'}')
        os.remove('conf1.a')
        raise RuntimeError('Ranlib is not functional with your archiver.  Try --with-ranlib=true if ranlib is unnecessary.')
      return
    arext = 'a'
    oldLibs = self.framework.argDB['LIBS']
    self.pushLanguage('C')
    for (archiver, flags, ranlib) in self.generateArchiverGuesses():
      if not self.checkCompile('', 'int foo(int a) {\n  return a+1;\n}\n\n', cleanup = 0, codeBegin = '', codeEnd = ''):
        raise RuntimeError('Compiler is not functional')
      if os.path.isfile('conf1.o'):
        os.remove('conf1.o')
      os.rename(self.compilerObj, 'conf1.o')
      if self.getExecutable(archiver, getFullPath = 1, resultName = 'AR'):
        if self.getExecutable(ranlib, getFullPath = 1, resultName = 'RANLIB'):
          try:
            (output, error, status) = config.base.Configure.executeShellCommand(self.AR+' '+flags+' libconf1.a conf1.o', checkCommand = checkArchive, log = self.framework.log)
            (output, error, status) = config.base.Configure.executeShellCommand(self.RANLIB+' libconf1.a', checkCommand = checkRanlib,log = self.framework.log)
          except RuntimeError, e:
            self.logPrint(str(e))
            continue
          self.framework.argDB['LIBS'] = '-lconf1'
          success =  self.checkLink('extern int foo(int);', '  int b = foo(1);  if (b);\n')
          os.rename('libconf1.a','libconf1.lib')
          if not success:
            success = self.checkLink('extern int foo(int);', '  int b = foo(1);  if (b);\n')
            os.remove('libconf1.lib')
            if success:
              arext = 'lib'
              break
          else:
            os.remove('libconf1.lib')
            break
    else:
      if os.path.isfile('conf1.o'):
        os.remove('conf1.o')
      self.framework.argDB['LIBS'] = oldLibs
      self.popLanguage()
      raise RuntimeError('Could not find a suitable archiver.  Use --with-ar to specify an archiver.')
    self.getExecutable(archiver, getFullPath = 1, resultName = 'AR')
    self.getExecutable(ranlib, getFullPath = 1, resultName = 'RANLIB')
    self.framework.argDB['RANLIB'] = self.RANLIB
    self.framework.argDB['AR_FLAGS'] = flags
    self.framework.addMakeMacro('AR_FLAGS',flags)
    self.AR_FLAGS      = flags
    self.AR_LIB_SUFFIX = arext
    self.addMakeMacro('AR_LIB_SUFFIX',arext)
    os.remove('conf1.o')
    self.framework.argDB['LIBS'] = oldLibs
    self.popLanguage()
    return

  def generateSharedLinkerGuesses(self):
    if not self.framework.argDB['with-shared']:
      self.framework.argDB['LD_SHARED'] = ''
      language = self.framework.normalizeLanguage(self.language[-1])
      linker = self.framework.setSharedLinkerObject(language, self.framework.getLanguageModule(language).StaticLinker(self.framework.argDB))
      yield (self.AR, [self.AR_FLAGS], self.AR_LIB_SUFFIX)
      raise RuntimeError('Archiver failed static link check')
    if 'with-shared-ld' in self.framework.argDB:
      yield (self.framework.argDB['with-shared-ld'], [], 'so')
    # C compiler default
    yield (self.framework.argDB['CC'], ['-shared'], 'so')
    # Mac OSX
    yield ('libtool', ['-noprebind', '-dynamic'], 'dylib')
    # Default to static linker
    self.framework.argDB['LD_SHARED'] = ''
    language = self.framework.normalizeLanguage(self.language[-1])
    linker = self.framework.setSharedLinkerObject(language, self.framework.getLanguageModule(language).StaticLinker(self.framework.argDB))
    yield (self.AR, [self.AR_FLAGS], self.AR_LIB_SUFFIX)
    raise RuntimeError('Archiver failed static link check')

  def checkSharedLinker(self):
    '''Check that the linker can produce shared libraries'''
    self.sharedLibraries = 0
    for linker, flags, ext in self.generateSharedLinkerGuesses():
      self.logPrint('Checking shared linker '+linker+' using flags '+str(flags))
      if self.getExecutable(linker, resultName = 'LD_SHARED'):
        self.framework.argDB['LD_SHARED'] = self.LD_SHARED
        flagsArg = self.getLinkerFlagsArg()
        oldFlags = self.framework.argDB[flagsArg]
        goodFlags = filter(self.checkLinkerFlag, flags)
        testMethod = 'foo'
        self.framework.argDB[flagsArg] += ' '+' '.join(goodFlags)
        if self.checkLink(includes = 'int '+testMethod+'(void) {return 0;}\n', codeBegin = '', codeEnd = '', cleanup = 0, shared = 1):
          os.rename(self.linkerObj, 'libfoo.'+ext)
          oldLibs = self.framework.argDB['LIBS']
          self.framework.argDB['LIBS'] += ' -L. -lfoo'
          self.framework.argDB[flagsArg] = oldFlags
          # Might need to segregate shared linker flags
          if self.checkLink(includes = 'int foo(void);', body = 'int ret = foo();\nif(ret);'):
            os.remove('libfoo.'+ext)
            self.framework.argDB['LIBS'] = oldLibs
            self.sharedLibraries = 1
            self.sharedLinker = linker
            self.sharedLibraryFlags = goodFlags
            self.sharedLibraryExt = ext
            self.logPrint('Using shared linker '+self.sharedLinker+' with flags '+str(self.sharedLibraryFlags)+' and library extension '+self.sharedLibraryExt)
            break
          self.framework.argDB['LIBS'] = oldLibs
          os.remove('libfoo.'+ext)
        if os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
        self.framework.argDB[flagsArg] = oldFlags
        del self.LD_SHARED
        del self.framework.argDB['LD_SHARED']
    return

  def checkSharedLinkerPaths(self):
    '''Determine the shared linker path options
       - IRIX: -rpath
       - Linux, OSF: -Wl,-rpath,
       - Solaris: -R
       - FreeBSD: -Wl,-R,'''
    languages = ['C']
    if 'CXX' in self.framework.argDB:
      languages.append('C++')
    if 'FC' in self.framework.argDB:
      languages.append('FC')
    for language in languages:
      flag = None
      self.pushLanguage(language)
      for testFlag in ['-Wl,-rpath,', '-rpath ', '-R', '-Wl,-R,']:
        self.framework.logPrint('Trying '+language+' linker flag '+testFlag)
        if self.checkLinkerFlag(testFlag+os.path.abspath(os.getcwd())):
          flag = testFlag
          break
        else:
          self.framework.logPrint('Rejected '+language+' linker flag '+testFlag)
      self.popLanguage()
      setattr(self, language.replace('+', 'x')+'SharedLinkerFlag', flag)
    return

  def output(self):
    '''Output module data as defines and substitutions'''
    if 'CC' in self.framework.argDB:
      self.addArgumentSubstitution('CC', 'CC')
      self.addArgumentSubstitution('CFLAGS', 'CFLAGS')
      if not self.CSharedLinkerFlag is None:
        self.addMakeMacro('CC_LINKER_SLFLAG', self.CSharedLinkerFlag)
    if 'CPP' in self.framework.argDB:
      self.addArgumentSubstitution('CPP', 'CPP')
      self.addArgumentSubstitution('CPPFLAGS', 'CPPFLAGS')
    if 'CXX' in self.framework.argDB:
      self.addArgumentSubstitution('CXX', 'CXX')
      self.addArgumentSubstitution('CXX_CXXFLAGS', 'CXX_CXXFLAGS')
      self.addArgumentSubstitution('CXXFLAGS', 'CXXFLAGS')
      if not self.CxxSharedLinkerFlag is None:
        self.addSubstitution('CXX_LINKER_SLFLAG', self.CxxSharedLinkerFlag)
    else:
      self.addSubstitution('CXX', '')
    if 'CXXCPP' in self.framework.argDB:
      self.addArgumentSubstitution('CXXCPP', 'CXXCPP')
    if 'FC' in self.framework.argDB:
      self.addArgumentSubstitution('FC', 'FC')
      self.addArgumentSubstitution('FFLAGS', 'FFLAGS')
      if not self.FCSharedLinkerFlag is None:
        self.addMakeMacro('FC_LINKER_SLFLAG', self.FCSharedLinkerFlag)
    else:
      self.addSubstitution('FC', '')
    self.addArgumentSubstitution('LDFLAGS', 'LDFLAGS')
    if hasattr(self,'sharedLibraryFlags'):
      self.addSubstitution('SHARED_LIBRARY_FLAG', ' '.join(self.sharedLibraryFlags))
    else:
      self.addSubstitution('SHARED_LIBRARY_FLAG','')
    return

  def configure(self):
    self.framework.argDB['LIBS'] = ''
    self.executeTest(self.checkCCompiler)
    self.executeTest(self.checkCPreprocessor)
    self.executeTest(self.checkCxxCompiler)
    self.executeTest(self.checkFortranCompiler)
    self.executeTest(self.checkPIC)
    self.executeTest(self.checkArchiver)
    self.executeTest(self.checkSharedLinker)
    self.executeTest(self.checkSharedLinkerPaths)
    self.executeTest(self.output)
    return
