from __future__ import generators
import config.base
import config.package
from sourceDatabase import SourceDB
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.defaultPrecision    = 'double'
    self.f2c                 = 0  # indicates either the f2cblaslapack are used or there is no Fortran compiler (and system BLAS/LAPACK is used)
    self.has64bitindices     = 0
    self.mkl                 = 0  # indicates BLAS/LAPACK library used is Intel MKL
    self.mkl_spblas_h        = 0  # indicates mkl_spblas.h is found
    self.separateBlas        = 1
    self.required            = 1
    self.alternativedownload = 'f2cblaslapack'
    self.missingRoutines     = []

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.f2cblaslapack = framework.require('config.packages.f2cblaslapack', self)
    self.fblaslapack   = framework.require('config.packages.fblaslapack', self)
    self.blis          = framework.require('config.packages.blis', self)
    self.openblas      = framework.require('config.packages.openblas', self)
    self.flibs         = framework.require('config.packages.flibs',self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.openmp        = framework.require('config.packages.openmp',self)
    self.mpi           = framework.require('config.packages.MPI',self)
    self.deps          = [self.flibs,self.mathlib]
    self.odeps         = [self.mpi]
    return

  def __str__(self):
    output  = config.package.Package.__str__(self)
    if self.has64bitindices:
      output += '  uses 8 byte integers\n'
    else:
      output += '  uses 4 byte integers\n'
    return output

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    help.addArgument('BLAS/LAPACK', '-with-blas-lib=<libraries: e.g. [/Users/..../libblas.a,...]>',    nargs.ArgLibrary(None, None, 'Indicate the library(s) containing BLAS'))
    help.addArgument('BLAS/LAPACK', '-with-lapack-lib=<libraries: e.g. [/Users/..../liblapack.a,...]>',nargs.ArgLibrary(None, None, 'Indicate the library(s) containing LAPACK'))
    help.addArgument('BLAS/LAPACK', '-with-blaslapack-suffix=<string>',nargs.ArgLibrary(None, None, 'Indicate a suffix for BLAS/LAPACK subroutine names.'))
    help.addArgument('BLAS/LAPACK', '-with-64-bit-blas-indices', nargs.ArgBool(None, 0, 'Try to use 64 bit integers for BLAS/LAPACK; will error if not available'))
    help.addArgument('BLAS/LAPACK', '-known-64-bit-blas-indices=<bool>', nargs.ArgBool(None, None, 'Indicate if using 64 bit integer BLAS'))
    return

  def getPrefix(self):
    if self.compilers.fortranMangling == 'caps':
      if self.defaultPrecision == 'single': return 'S'
      if self.defaultPrecision == 'double': return 'D'
      if self.defaultPrecision == '__float128': return 'Q'
      if self.defaultPrecision == '__fp16': return 'H'
      return 'Unknown precision'
    else:
      if self.defaultPrecision == 'single': return 's'
      if self.defaultPrecision == 'double': return 'd'
      if self.defaultPrecision == '__float128': return 'q'
      if self.defaultPrecision == '__fp16': return 'h'
      return 'Unknown precision'

  def getType(self):
    if self.defaultPrecision == 'single': return 'float'
    return self.defaultPrecision

  def getOtherLibs(self, foundBlas = None, blasLibrary = None, separateBlas = None):
    if foundBlas is None:
      foundBlas = self.foundBlas
    if blasLibrary is None:
      blasLibrary = self.blasLibrary
    if separateBlas is None:
      separateBlas = self.separateBlas
    otherLibs = []
    if foundBlas:
      if separateBlas:
        otherLibs += blasLibrary
    otherLibs += self.dlib
    return otherLibs

  def checkBlas(self, blasLibrary, otherLibs, fortranMangle, routineIn = 'dot'):
    '''This checks the given library for the routine, dot by default'''
    oldLibs = self.compilers.LIBS
    routine = self.mangleBlas(routineIn)
    self.libraries.saveLog()
    found = self.libraries.check(blasLibrary, routine, otherLibs = otherLibs, fortranMangle = fortranMangle)
    self.logWrite(self.libraries.restoreLog())
    self.compilers.LIBS = oldLibs
    return found

  def checkLapack(self, lapackLibrary, otherLibs, fortranMangle, routinesIn = ['getrs','geev']):
    oldLibs = self.compilers.LIBS
    if not isinstance(routinesIn, list): routinesIn = [routinesIn]
    routines = list(routinesIn)
    found = 1
    routines = map(self.mangleBlas, routines)

    for routine in routines:
      self.libraries.saveLog()
      found = found and self.libraries.check(lapackLibrary, routine, otherLibs = otherLibs, fortranMangle = fortranMangle)
      self.logWrite(self.libraries.restoreLog())
      if not found: break
    self.compilers.LIBS = oldLibs
    return found

  def checkLib(self, lapackLibrary, blasLibrary = None):
    '''Checking for BLAS and LAPACK symbols'''

    if blasLibrary is None:
      self.separateBlas = 0
      blasLibrary       = lapackLibrary
    else:
      self.separateBlas = 1
    if not isinstance(lapackLibrary, list): lapackLibrary = [lapackLibrary]
    if not isinstance(blasLibrary,   list): blasLibrary   = [blasLibrary]
    foundBlas   = 0
    foundLapack = 0
    self.f2c    = 0
    # allow a user-specified suffix to be appended to BLAS/LAPACK symbols
    self.suffix = self.argDB.get('with-blaslapack-suffix', '')
    mangleFunc = self.compilers.fortranMangling
    self.logPrint('Checking for Fortran name mangling '+mangleFunc+' on BLAS/LAPACK')
    foundBlas = self.checkBlas(blasLibrary, self.getOtherLibs(foundBlas, blasLibrary), mangleFunc,'dot')
    if foundBlas:
      foundLapack = self.checkLapack(lapackLibrary, self.getOtherLibs(foundBlas, blasLibrary), mangleFunc)
      if foundLapack:
        self.mangling = self.compilers.fortranMangling
        self.logPrint('Found Fortran mangling on BLAS/LAPACK which is '+self.compilers.fortranMangling)
        return (foundBlas, foundLapack)
    if not self.compilers.fortranMangling == 'unchanged':
      self.logPrint('Checking for no name mangling on BLAS/LAPACK')
      self.mangling = 'unchanged'
      foundBlas = self.checkBlas(blasLibrary, self.getOtherLibs(foundBlas, blasLibrary), 0, 'dot')
      if foundBlas:
        foundLapack = self.checkLapack(lapackLibrary, self.getOtherLibs(foundBlas, blasLibrary), 0, ['getrs','geev'])
        if foundLapack:
          self.logPrint('Found no name mangling on BLAS/LAPACK')
          return (foundBlas, foundLapack)
    if not self.compilers.fortranMangling == 'underscore':
      save_f2c = self.f2c
      self.f2c = 1 # so that mangleBlas will do its job
      self.logPrint('Checking for underscore name mangling on BLAS/LAPACK')
      self.mangling = 'underscore'
      foundBlas = self.checkBlas(blasLibrary, self.getOtherLibs(foundBlas, blasLibrary), 0, 'dot')
      if foundBlas:
        foundLapack = self.checkLapack(lapackLibrary, self.getOtherLibs(foundBlas, blasLibrary), 0, ['getrs','geev'])
        if foundLapack:
          self.logPrint('Found underscore name mangling on BLAS/LAPACK')
          return (foundBlas, foundLapack)
      self.f2c = save_f2c
    self.logPrint('Unknown name mangling in BLAS/LAPACK')
    self.mangling = 'unknown'
    return (foundBlas, foundLapack)

  def generateGuesses(self):
    # check that user has used the options properly
    if self.f2cblaslapack.found:
      self.f2c = 1
      # TODO: use self.f2cblaslapack.libDir directly
      libDir = os.path.join(self.f2cblaslapack.directory,'lib')
      f2cBlas = [os.path.join(libDir,'libf2cblas.a')]
      if self.blis.found:
        # The real BLAS is provided by libblis, but we still need libf2cblas for aux functions needed by libf2clapack
        f2cBlas += self.blis.lib
      f2cLapack = [os.path.join(libDir,'libf2clapack.a')]
      yield ('f2cblaslapack', f2cBlas, f2cLapack, '32','no')
      yield ('f2cblaslapack', f2cBlas+['-lquadmath'], f2cLapack, '32','no')
      raise RuntimeError('--download-f2cblaslapack libraries cannot be used')
    if self.fblaslapack.found:
      self.f2c = 0
      # TODO: use self.fblaslapack.libDir directly
      libDir = os.path.join(self.fblaslapack.directory,'lib')
      yield ('fblaslapack', os.path.join(libDir,'libfblas.a'), os.path.join(libDir,'libflapack.a'), '32','no')
      raise RuntimeError('--download-fblaslapack libraries cannot be used')
    if self.blis.found:
      self.f2c = 0
      # TODO: Where shall we find liblapack.a?
      yield ('BLIS', self.blis.lib, 'liblapack.a', self.blis.known64, self.blis.usesopenmp)
    if self.openblas.found:
      self.f2c = 0
      self.include = self.openblas.include
      if self.openblas.libDir:
        yield ('OpenBLAS with full path', None, os.path.join(self.openblas.libDir,'libopenblas.a'),self.openblas.known64,self.openblas.usesopenmp)
      else:
        yield ('OpenBLAS', None, self.openblas.lib,self.openblas.known64,self.openblas.usesopenmp)
      raise RuntimeError('--download-openblas libraries cannot be used')
    if 'with-blas-lib' in self.argDB and not 'with-lapack-lib' in self.argDB:
      raise RuntimeError('If you use the --with-blas-lib=<lib> you must also use --with-lapack-lib=<lib> option')
    if not 'with-blas-lib' in self.argDB and 'with-lapack-lib' in self.argDB:
      raise RuntimeError('If you use the --with-lapack-lib=<lib> you must also use --with-blas-lib=<lib> option')
    if 'with-blas-lib' in self.argDB and 'with-blaslapack-dir' in self.argDB:
      raise RuntimeError('You cannot set both the library containing BLAS with --with-blas-lib=<lib>\nand the directory to search with --with-blaslapack-dir=<dir>')
    if 'with-blaslapack-lib' in self.argDB and 'with-blaslapack-dir' in self.argDB:
      raise RuntimeError('You cannot set both the library containing BLAS/LAPACK with --with-blaslapack-lib=<lib>\nand the directory to search with --with-blaslapack-dir=<dir>')

    # Try specified BLASLAPACK library
    if 'with-blaslapack-lib' in self.argDB:
      yield ('User specified BLAS/LAPACK library', None, self.argDB['with-blaslapack-lib'], 'unknown','unknown')
      if self.defaultPrecision == '__float128':
        raise RuntimeError('__float128 precision requires f2c BLAS/LAPACK libraries; they are not available in '+str(self.argDB['with-blaslapack-lib'])+'; suggest --download-f2cblaslapack\n')
      else:
        raise RuntimeError('You set a value for --with-blaslapack-lib=<lib>, but '+str(self.argDB['with-blaslapack-lib'])+' cannot be used\n')
    # Try specified BLAS and LAPACK libraries
    if 'with-blas-lib' in self.argDB and 'with-lapack-lib' in self.argDB:
      yield ('User specified BLAS and LAPACK libraries', self.argDB['with-blas-lib'], self.argDB['with-lapack-lib'], 'unknown', 'unknown')
      if self.defaultPrecision == '__float128':
        raise RuntimeError('__float128 precision requires f2c BLAS/LAPACK libraries; they are not available in '+str(self.argDB['with-blas-lib'])+' and '+str(self.argDB['with-lapack-lib'])+'; suggest --download-f2cblaslapack\n')
      else:
        raise RuntimeError('You set a value for --with-blas-lib=<lib> and --with-lapack-lib=<lib>, but '+str(self.argDB['with-blas-lib'])+' and '+str(self.argDB['with-lapack-lib'])+' cannot be used\n')

    blislib = ['libblis.a']
    if self.openmp.found:
      blislib.insert(0,'libblis-mt.a')

    if not 'with-blaslapack-dir' in self.argDB:
      mkl = os.getenv('MKLROOT')
      if mkl:
        # Since user did not select MKL specifically first try compiler defaults and only if they fail use the MKL
        yield ('Default compiler libraries', '', '','unknown','unknown')
        for lib in blislib:
          for lapack in ['libflame.a','liblapack.a']:
            for libdir in ['',os.path.join('/usr','local','lib')]:
              if libdir:
                lib = os.path.join(libdir,lib)
                lapack = os.path.join(libdir,lapack)
            yield ('BLIS/AMD-AOCL default compiler locations '+libdir,lib,lapack,'unknown','unknown')
        yield ('OpenBLAS default compiler locations', None, 'libopenblas.a','unknown','unknown')
        yield ('OpenBLAS default compiler locations /usr/local/lib', None, os.path.join('/usr','local','lib','libopenblas.a'),'unknown','unknown')
        yield ('Default compiler locations', 'libblas.a', 'liblapack.a','unknown','unknown')
        yield ('Default compiler locations /usr/local/lib', os.path.join('/usr','local','lib','libblas.a'), os.path.join('/usr','local','lib','liblapack.a'),'unknown','unknown')
        yield ('Default compiler locations with gfortran', None, ['liblapack.a', 'libblas.a','libgfortran.a'],'unknown','unknown')
        self.logWrite('Did not detect default BLAS and LAPACK locations so using the value of MKLROOT to search as --with-blas-lapack-dir='+mkl)
        self.argDB['with-blaslapack-dir'] = mkl
        self.checkingMKLROOTautomatically = 1

    if self.argDB['with-64-bit-blas-indices']:
      flexiblas = 'libflexiblas64.a'
      ILP64 = '_ilp64'
      known = '64'
    else:
      flexiblas = 'libflexiblas.a'
      ILP64 = '_lp64'
      known = '32'

    if self.openmp.found:
      ITHREADS=['intel_thread','gnu_thread']
      ompthread = 'yes'
    else:
      ITHREADS=['sequential']
      ompthread = 'no'

    # Try specified installation root
    if 'with-blaslapack-dir' in self.argDB:
      dir = self.argDB['with-blaslapack-dir']
      # error if package-dir is in externalpackages
      if os.path.realpath(dir).find(os.path.realpath(self.externalPackagesDir)) >=0:
        fakeExternalPackagesDir = dir.replace(os.path.realpath(dir).replace(os.path.realpath(self.externalPackagesDir),''),'')
        raise RuntimeError('Bad option: '+'--with-blaslapack-dir='+self.argDB['with-blaslapack-dir']+'\n'+
                           fakeExternalPackagesDir+' is reserved for --download-package scratch space. \n'+
                           'Do not install software in this location nor use software in this directory.')
      if self.defaultPrecision == '__float128':
        yield ('User specified installation root (F2CBLASLAPACK)', os.path.join(dir,'libf2cblas.a'), os.path.join(dir, 'libf2clapack.a'), '32','no')
        raise RuntimeError('__float128 precision requires f2c libraries; they are not available in '+dir+'; suggest --download-f2cblaslapack\n')

      if not (len(dir) > 2 and dir[1] == ':') :
        dir = os.path.abspath(dir)
      self.log.write('Looking for BLAS/LAPACK in user specified directory: '+dir+'\n')
      self.log.write('Files and directories in that directory:\n'+str(os.listdir(dir))+'\n')

      # Look for Multi-Threaded MKL for MKL_C/Pardiso
      useCPardiso=0
      usePardiso=0
      if self.argDB['with-mkl_cpardiso'] or 'with-mkl_cpardiso-dir' in self.argDB or 'with-mkl_cpardiso-lib' in self.argDB:
        useCPardiso=1
        if self.mpi.found and hasattr(self.mpi, 'ompi_major_version'):
          mkl_blacs_64=[['mkl_blacs_openmpi'+ILP64+'']]
          mkl_blacs_32=[['mkl_blacs_openmpi']]
        else:
          mkl_blacs_64=[['mkl_blacs_intelmpi'+ILP64+''],['mkl_blacs_mpich'+ILP64+''],['mkl_blacs_sgimpt'+ILP64+''],['mkl_blacs_openmpi'+ILP64+'']]
          mkl_blacs_32=[['mkl_blacs_intelmpi'],['mkl_blacs_mpich'],['mkl_blacs_sgimpt'],['mkl_blacs_openmpi']]
      elif self.argDB['with-mkl_pardiso'] or 'with-mkl_pardiso-dir' in self.argDB or 'with-mkl_pardiso-lib' in self.argDB:
        usePardiso=1
        mkl_blacs_64=[[]]
        mkl_blacs_32=[[]]
      if useCPardiso or usePardiso:
        self.logPrintBox('BLASLAPACK: Looking for Multithreaded MKL for C/Pardiso')
        for libdir in [os.path.join('lib','64'),os.path.join('lib','ia64'),os.path.join('lib','em64t'),os.path.join('lib','intel64'),'lib','64','ia64','em64t','intel64',
                       os.path.join('lib','32'),os.path.join('lib','ia32'),'32','ia32','']:
          if not os.path.exists(os.path.join(dir,libdir)):
            self.logPrint('MKL Path not found.. skipping: '+os.path.join(dir,libdir))
          else:
            self.log.write('Files and directories in that directory:\n'+str(os.listdir(os.path.join(dir,libdir)))+'\n')
            #  iomp5 is provided by the Intel compilers on MacOS. Run source /opt/intel/bin/compilervars.sh intel64 to have it added to LIBRARY_PATH
            #  then locate libimp5.dylib in the LIBRARY_PATH and copy it to os.path.join(dir,libdir)
            for i in mkl_blacs_64:
              yield ('User specified MKL-C/Pardiso Intel-Linux64', None, [os.path.join(dir,libdir,'libmkl_intel'+ILP64+'.a'),'mkl_core','mkl_intel_thread']+i+['iomp5','dl','pthread'],known,'yes')
              yield ('User specified MKL-C/Pardiso GNU-Linux64', None, [os.path.join(dir,libdir,'libmkl_intel'+ILP64+'.a'),'mkl_core','mkl_gnu_thread']+i+['gomp','dl','pthread'],known,'yes')
              yield ('User specified MKL-Pardiso Intel-Windows64', None, [os.path.join(dir,libdir,'mkl_core.lib'),'mkl_intel'+ILP64+'.lib','mkl_intel_thread.lib']+i+['libiomp5md.lib'],known,'yes')
            for i in mkl_blacs_32:
              yield ('User specified MKL-C/Pardiso Intel-Linux32', None, [os.path.join(dir,libdir,'libmkl_intel.a'),'mkl_core','mkl_intel_thread']+i+['iomp5','dl','pthread'],'32','yes')
              yield ('User specified MKL-C/Pardiso GNU-Linux32', None, [os.path.join(dir,libdir,'libmkl_intel.a'),'mkl_core','mkl_gnu_thread']+i+['gomp','dl','pthread'],'32','yes')
              yield ('User specified MKL-Pardiso Intel-Windows32', None, [os.path.join(dir,libdir,'mkl_core.lib'),'mkl_intel_c.lib','mkl_intel_thread.lib']+i+['libiomp5md.lib'],'32','yes')
        return

      self.log.write('Files and directories in that directory:\n'+str(os.listdir(dir))+'\n')
      # Check MATLAB [ILP64] MKL
      yield ('User specified MATLAB [ILP64] MKL Linux lib dir', None, [os.path.join(dir,'bin','glnxa64','mkl.so'), os.path.join(dir,'sys','os','glnxa64','libiomp5.so'), 'pthread'],'64','yes')
      oldFlags = self.setCompilers.LDFLAGS
      self.setCompilers.LDFLAGS += '-Wl,-rpath,'+os.path.join(dir,'bin','maci64')
      yield ('User specified MATLAB [ILP64] MKL MacOS lib dir', None, [os.path.join(dir,'bin','maci64','mkl.dylib'), os.path.join(dir,'sys','os','maci64','libiomp5.dylib'), 'pthread'],'64','yes')
      self.setCompilers.LDFLAGS = oldFlags
      for ITHREAD in ITHREADS:
        yield ('User specified MKL11/12 and later', None, [os.path.join(dir,'libmkl_intel'+ILP64+'.a'),'mkl_core','mkl_'+ITHREAD,'pthread'],known,ompthread)
      # Some new MKL 11/12 variations
      for libdir in [os.path.join('lib','intel64'),os.path.join('lib','32'),os.path.join('lib','ia32'),'32','ia32','']:
        if not os.path.exists(os.path.join(dir,libdir)):
          self.logPrint('MKL Path not found.. skipping: '+os.path.join(dir,libdir))
        else:
          self.log.write('Files and directories in that directory:\n'+str(os.listdir(os.path.join(dir,libdir)))+'\n')
          for ITHREAD in ITHREADS:
            yield ('User specified MKL11/12 Linux32', None, [os.path.join(dir,libdir,'libmkl_intel'+ILP64+'.a'),'mkl_core','mkl_'+ITHREAD,'pthread'],known,ompthread)
            yield ('User specified MKL11/12 Linux32 for static linking (Cray)', None, ['-Wl,--start-group',os.path.join(dir,libdir,'libmkl_intel'+ILP64+'.a'),'mkl_core','mkl_'+ITHREAD,'-Wl,--end-group','pthread'],known,ompthread)
      for libdir in [os.path.join('lib','intel64'),os.path.join('lib','64'),os.path.join('lib','ia64'),os.path.join('lib','em64t'),os.path.join('lib','intel64'),'lib','64','ia64','em64t','intel64','']:
        if not os.path.exists(os.path.join(dir,libdir)):
          self.logPrint('MKL Path not found.. skipping: '+os.path.join(dir,libdir))
        else:
          self.log.write('Files and directories in that directory:\n'+str(os.listdir(os.path.join(dir,libdir)))+'\n')
          for ITHREAD in ITHREADS:
            yield ('User specified MKL11+ Linux64', None, [os.path.join(dir,libdir,'libmkl_intel'+ILP64+'.a'),'mkl_core','mkl_'+ITHREAD,'mkl_def','pthread'],known,ompthread)
            yield ('User specified MKL11+ Mac-64', None, [os.path.join(dir,libdir,'libmkl_intel'+ILP64+'.a'),'mkl_core','mkl_'+ITHREAD,'pthread'],known,ompthread)
      # Older Linux MKL checks
      yield ('User specified MKL Linux lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), 'mkl', 'guide', 'pthread'],'32','no')
      for libdir in ['32','64','em64t']:
        if not os.path.exists(os.path.join(dir,libdir)):
          self.logPrint('MKL Path not found.. skipping: '+os.path.join(dir,libdir))
        else:
          self.log.write('Files and directories in that directory:\n'+str(os.listdir(os.path.join(dir,libdir)))+'\n')
          yield ('User specified MKL Linux installation root', None, [os.path.join(dir,'lib',libdir,'libmkl_lapack.a'),'mkl', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-x86 lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), 'libmkl_def.a', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-x86 lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), 'libmkl_def.a', 'guide', 'vml','pthread'],'32','no')
      yield ('User specified MKL Linux-ia64 lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), 'libmkl_ipf.a', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-em64t lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), 'libmkl_em64t.a', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-x86 installation root', None, [os.path.join(dir,'lib','32','libmkl_lapack.a'),'libmkl_def.a', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-x86 installation root', None, [os.path.join(dir,'lib','32','libmkl_lapack.a'),'libmkl_def.a', 'guide', 'vml','pthread'],'32','no')
      yield ('User specified MKL Linux-ia64 installation root', None, [os.path.join(dir,'lib','64','libmkl_lapack.a'),'libmkl_ipf.a', 'guide', 'pthread'],'32','no')
      yield ('User specified MKL Linux-em64t installation root', None, [os.path.join(dir,'lib','em64t','libmkl_lapack.a'),'libmkl_em64t.a', 'guide', 'pthread'],'32','no')
      # Mac MKL check
      yield ('User specified MKL Mac-x86 lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), 'libmkl_ia32.a', 'guide'],'32','no')
      yield ('User specified MKL Max-x86 installation root', None, [os.path.join(dir,'Libraries','32','libmkl_lapack.a'),'libmkl_ia32.a', 'guide'],'32','no')
      yield ('User specified MKL Max-x86 installation root', None, [os.path.join(dir,'lib','32','libmkl_lapack.a'),'libmkl_ia32.a', 'guide'],'32','no')
      yield ('User specified MKL Mac-em64t lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), 'libmkl_intel'+ILP64+'.a', 'guide'],known,'no')
      yield ('User specified MKL Max-em64t installation root', None, [os.path.join(dir,'Libraries','32','libmkl_lapack.a'),'libmkl_intel'+ILP64+'.a', 'guide'],'32','no')
      yield ('User specified MKL Max-em64t installation root', None, [os.path.join(dir,'lib','32','libmkl_lapack.a'),'libmkl_intel'+ILP64+'.a', 'guide'],'32','no')
      # Check MKL on windows
      yield ('User specified MKL Windows lib dir', None, [os.path.join(dir, 'mkl_c_dll.lib')],'32','no')
      yield ('User specified stdcall MKL Windows lib dir', None, [os.path.join(dir, 'mkl_s_dll.lib')],'32','no')
      yield ('User specified ia64/em64t MKL Windows lib dir', None, [os.path.join(dir, 'mkl_dll.lib')],'32','no')
      for ITHREAD in ITHREADS:
        yield ('User specified MKL10-32 Windows lib dir', None, [os.path.join(dir, 'mkl_intel_c_dll.lib'),'mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],'32',ompthread)
        yield ('User specified MKL10-32 Windows stdcall lib dir', None, [os.path.join(dir, 'mkl_intel_s_dll.lib'),'mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],'32',ompthread)
        yield ('User specified MKL10-64 Windows lib dir', None, [os.path.join(dir, 'mkl_intel'+ILP64+'_dll.lib'),'mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],known,ompthread)
      mkldir = os.path.join(dir, 'ia32', 'lib')
      yield ('User specified MKL Windows installation root', None, [os.path.join(mkldir, 'mkl_c_dll.lib')],'32','no')
      yield ('User specified stdcall MKL Windows installation root', None, [os.path.join(mkldir, 'mkl_s_dll.lib')],'32','no')
      for ITHREAD in ITHREADS:
        yield ('User specified MKL10-32 Windows installation root', None, [os.path.join(mkldir, 'mkl_intel_c_dll.lib'),'mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],'32',ompthread)
        yield ('User specified MKL10-32 Windows stdcall installation root', None, [os.path.join(mkldir, 'mkl_intel_s_dll.lib'),'mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],'32',ompthread)
      mkldir = os.path.join(dir, 'em64t', 'lib')
      for ITHREAD in ITHREADS:
        yield ('User specified MKL10-64 Windows installation root', None, [os.path.join(mkldir, 'mkl_intel'+ILP64+'_dll.lib'),'mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],known,ompthread)
      yield ('User specified em64t MKL Windows installation root', None, [os.path.join(mkldir, 'mkl_dll.lib')],'32','no')
      mkldir = os.path.join(dir, 'ia64', 'lib')
      yield ('User specified ia64 MKL Windows installation root', None, [os.path.join(mkldir, 'mkl_dll.lib')],'32','no')
      for ITHREAD in ITHREADS:
        yield ('User specified MKL10-64 Windows installation root', None, [os.path.join(mkldir, 'mkl_intel'+ILP64+'_dll.lib'),'mkl_'+ITHREAD+'_dll.lib','mkl_core_dll.lib','libiomp5md.lib'],known,ompthread)
      # Check AMD ACML libraries
      yield ('User specified AMD ACML lib dir', None, os.path.join(dir,'lib','libacml.a'),'32','unknown')
      yield ('User specified AMD ACML lib dir', None, [os.path.join(dir,'lib','libacml.a'), os.path.join(dir,'lib','libacml_mv.a')],'32','unknown')
      yield ('User specified AMD ACML lib dir', None, os.path.join(dir,'lib','libacml_mp.a'),'32','unknown')
      yield ('User specified AMD ACML lib dir', None, [os.path.join(dir,'lib','libacml_mp.a'), os.path.join(dir,'lib','libacml_mv.a')],'32','unknown')
      # Check BLIS/AMD-AOCL libraries
      for lib in blislib:
        for lapack in ['libflame.a','liblapack.a']:
          for libdir in [dir,os.path.join(dir,'lib')]:
            yield ('User specified installation root BLIS/AMD-AOCL', os.path.join(libdir,lib), os.path.join(libdir,lapack), 'unknown', 'unknown')
      # NEC
      yield ('User specified NEC lib dir', os.path.join(dir, 'lib', 'libblas_sequential.a'), [os.path.join(dir, 'lib', 'liblapack.a'), os.path.join(dir, 'lib', 'libasl_sequential.a')], 'unknown', 'unknown')
      yield ('User specified NEC lib dir', os.path.join(dir, 'lib', 'libblas_sequential.a'), os.path.join(dir, 'lib', 'liblapack.a'), 'unknown', 'unknown')
      # Search for FlexiBLAS
      for libdir in ['lib64', 'lib', '']:
        if os.path.exists(os.path.join(dir,libdir)):
            yield ('User specified FlexiBLAS',None,os.path.join(dir,libdir,flexiblas),known,'unknown')
      # Search for OpenBLAS
      for libdir in ['lib','']:
        if os.path.exists(os.path.join(dir,libdir)):
          yield ('User specified OpenBLAS',None,os.path.join(dir,libdir,'libopenblas.a'),'unknown','unknown')
      # Search for atlas
      yield ('User specified ATLAS Linux installation root', [os.path.join(dir, 'libcblas.a'),os.path.join(dir, 'libf77blas.a'), os.path.join(dir, 'libatlas.a')],  [os.path.join(dir, 'liblapack.a')],'32','no')
      yield ('User specified ATLAS Linux installation root', [os.path.join(dir, 'libf77blas.a'), os.path.join(dir, 'libatlas.a')],  [os.path.join(dir, 'liblapack.a')],'32','no')

      yield ('User specified installation root (HPUX)', os.path.join(dir, 'libveclib.a'),  os.path.join(dir, 'liblapack.a'),'32','unknown')
      for libdir in ['lib64','lib','']:
        if os.path.exists(os.path.join(dir,libdir)):
          yield ('User specified installation root (F2CBLASLAPACK)', os.path.join(dir,libdir,'libf2cblas.a'), os.path.join(dir,libdir,'libf2clapack.a'),'32','no')
      yield ('User specified installation root(FBLASLAPACK)', os.path.join(dir, 'libfblas.a'),   os.path.join(dir, 'libflapack.a'),'32','no')
      for lib in ['','lib64']:
        yield ('User specified installation root IBM ESSL', None, os.path.join(dir, lib, 'libessl.a'),'32','unknown')
      # Search for liblapack.a and libblas.a after the implementations with more specific name to avoid
      # finding these in /usr/lib despite using -L<blaslapack-dir> while attempting to get a different library.
      for libdir in ['lib64','lib','']:
        if os.path.exists(os.path.join(dir,libdir)):
          yield ('User specified installation root BLAS/LAPACK',os.path.join(dir,libdir,'libblas.a'),os.path.join(dir,libdir,'liblapack.a'),'unknown','unknown')
      if hasattr(self,'checkingMKROOTautomatically'):
        raise RuntimeError('Unable to locate working BLAS/LAPACK libraries, even tried libraries in MKLROOT '+self.argDB['with-blaslapack-dir']+'\n')
      else:
        raise RuntimeError('You set a value for --with-blaslapack-dir=<dir>, but '+self.argDB['with-blaslapack-dir']+' cannot be used\n')
    if self.defaultPrecision == '__float128':
      raise RuntimeError('__float128 precision requires f2c libraries; suggest --download-f2cblaslapack\n')

    # Try compiler defaults
    yield ('Default compiler libraries', '', '','unknown','unknown')
    yield ('Default NEC', 'libblas_sequential.a', ['liblapack.a','libasl_sequential.a'],'unknown','unknown')
    yield ('Default NEC', 'libblas_sequential.a', 'liblapack.a','unknown','unknown')
    yield ('Default FlexiBLAS', None, flexiblas, known, 'unknown')
    for lib in blislib:
      for lapack in ['libflame.a','liblapack.a']:
        yield ('Default BLIS/AMD-AOCL', lib, lapack,'unknown','unknown')
    yield ('Default compiler locations', 'libblas.a', 'liblapack.a','unknown','unknown')
    yield ('Default OpenBLAS', None, 'libopenblas.a','unknown','unknown')
    # Intel on Mac
    for ITHREAD in ITHREADS:
      yield ('User specified MKL Mac-64', None, [os.path.join('/opt','intel','mkl','lib','libmkl_intel'+ILP64+'.a'),'mkl_'+ITHREAD,'mkl_core','pthread'],known,ompthread)
    # Try Microsoft Windows location
    for MKL_Version in [os.path.join('MKL','9.0'),os.path.join('MKL','8.1.1'),os.path.join('MKL','8.1'),os.path.join('MKL','8.0.1'),os.path.join('MKL','8.0'),'MKL72','MKL70','MKL61','MKL']:
      mklpath = os.path.join('/cygdrive', 'c', 'Program Files', 'Intel', MKL_Version)
      if not os.path.exists(mklpath):
        self.logPrint('MKL Path not found.. skipping: '+mklpath)
      else:
        mkldir = os.path.join(mklpath, 'ia32', 'lib')
        yield ('Microsoft Windows, Intel MKL library', None, os.path.join(mkldir,'mkl_c_dll.lib'),'32','no')
        yield ('Microsoft Windows, Intel MKL stdcall library', None, os.path.join(mkldir,'mkl_s_dll.lib'),'32','no')
        mkldir = os.path.join(mklpath, 'em64t', 'lib')
        yield ('Microsoft Windows, em64t Intel MKL library', None, os.path.join(mkldir,'mkl_dll.lib'),'32','no')
        mkldir = os.path.join(mklpath, 'ia64', 'lib')
        yield ('Microsoft Windows, ia64 Intel MKL library', None, os.path.join(mkldir,'mkl_dll.lib'),'32','no')
    # IRIX locations
    yield ('IRIX Mathematics library', None, 'libcomplib.sgimath.a','32','unknown')
    yield ('Another IRIX Mathematics library', None, 'libscs.a','32','unknown')
    yield ('Compaq/Alpha Mathematics library', None, 'libcxml.a','32','unknown')
    # IBM ESSL locations
    yield ('IBM ESSL Mathematics library', None, 'libessl.a','32','unknown')
    yield ('IBM ESSL Mathematics library for Blue Gene', None, 'libesslbg.a','32','unknown')
    yield ('HPUX', 'libveclib.a', 'liblapack.a','unknown','unknown')
    # /usr/local/lib
    dir = os.path.join('/usr','local','lib')
    yield ('Default compiler locations /usr/local/lib', os.path.join(dir,'libblas.a'), os.path.join(dir,'liblapack.a'),'unknown','unknown')
    yield ('Default compiler locations /usr/local/lib', None, os.path.join(dir,'libopenblas.a'),'unknown','unknown')
    yield ('Default compiler locations with gfortran', None, ['liblapack.a', 'libblas.a','libgfortran.a'],'unknown','unknown')
    yield ('Default Atlas location',['libcblas.a','libf77blas.a','libatlas.a'],  ['liblapack.a'],'unknown','unknown')
    yield ('Default Atlas location',['libf77blas.a','libatlas.a'],  ['liblapack.a'],'unknown','unknown')
    yield ('Default compiler locations with G77', None, ['liblapack.a', 'libblas.a','libg2c.a'],'unknown','unknown')
    # Try MacOSX location
    dir = os.path.join('/Library', 'Frameworks', 'Intel_MKL.framework','Libraries','32')
    yield ('MacOSX with Intel MKL', None, [os.path.join(dir,'libmkl_lapack.a'),'libmkl_ia32.a','libguide.a'],'32','no')
    yield ('MacOSX BLAS/LAPACK library', None, os.path.join('/System', 'Library', 'Frameworks', 'vecLib.framework', 'vecLib'),'32','unknown')
    # Sun locations; this don't currently work
    yield ('Sun sunperf BLAS/LAPACK library', None, ['libsunperf.a','libsunmath.a'],'32','no')
    yield ('Sun sunperf BLAS/LAPACK library', None, ['libsunperf.a','libF77.a','libM77.a','libsunmath.a'],'32','no')
    yield ('Sun sunperf BLAS/LAPACK library', None, ['libsunperf.a','libfui.a','libfsu.a','libsunmath.a'],'32','no')
    # Try Microsoft Windows location
    for MKL_Version in [os.path.join('MKL','9.0'),os.path.join('MKL','8.1.1'),os.path.join('MKL','8.1'),os.path.join('MKL','8.0.1'),os.path.join('MKL','8.0'),'MKL72','MKL70','MKL61','MKL']:
      mklpath = os.path.join('/cygdrive', 'c', 'Program Files', 'Intel', MKL_Version)
      if not os.path.exists(mklpath):
        self.logPrint('MKL Path not found.. skipping: '+mklpath)
      else:
        mkldir = os.path.join(mklpath, 'ia32', 'lib')
        if os.path.exists(mkldir):
          self.log.write('Files and directories in that directory:\n'+str(os.listdir(mkldir))+'\n')
          yield ('Microsoft Windows, Intel MKL library', None, os.path.join(mkldir,'mkl_c_dll.lib'),'32','no')
          yield ('Microsoft Windows, Intel MKL stdcall library', None, os.path.join(mkldir,'mkl_s_dll.lib'),'32','no')
        mkldir = os.path.join(mklpath, 'em64t', 'lib')
        if os.path.exists(mkldir):
          self.log.write('Files and directories in that directory:\n'+str(os.listdir(mkldir))+'\n')
          yield ('Microsoft Windows, em64t Intel MKL library', None, os.path.join(mkldir,'mkl_dll.lib'),'32','no')
        mkldir = os.path.join(mklpath, 'ia64', 'lib')
        if os.path.exists(mkldir):
          self.log.write('Files and directories in that directory:\n'+str(os.listdir(mkldir))+'\n')
          yield ('Microsoft Windows, ia64 Intel MKL library', None, os.path.join(mkldir,'mkl_dll.lib'),'32','no')
    return

  def configureLibrary(self):
    if hasattr(self.compilers, 'FC'):
      self.alternativedownload = 'fblaslapack'
    for (name, self.blasLibrary, self.lapackLibrary, self.known64, self.usesopenmp) in self.generateGuesses():
      self.foundBlas   = 0
      self.foundLapack = 0
      self.log.write('================================================================================\n')
      self.log.write('Checking for BLAS and LAPACK in '+name+'\n')
      (self.foundBlas, self.foundLapack) = self.executeTest(self.checkLib, [self.lapackLibrary, self.blasLibrary])
      if self.foundBlas and self.foundLapack:
        if not isinstance(self.blasLibrary,   list): self.blasLibrary   = [self.blasLibrary]
        if not isinstance(self.lapackLibrary, list): self.lapackLibrary = [self.lapackLibrary]
        self.lib = []
        if self.lapackLibrary[0]: self.lib.extend(self.lapackLibrary)
        if self.blasLibrary[0]:   self.lib.extend(self.blasLibrary)
        self.dlib = self.lib+self.dlib
        self.framework.packages.append(self)
        break
      self.include = []
    if not self.foundBlas:
      # check for split blas/blas-dev packages
      import glob
      blib = glob.glob('/usr/lib/libblas.*')
      if blib != [] and not (os.path.isfile('/usr/lib/libblas.so') or os.path.isfile('/usr/lib/libblas.a')):
        raise RuntimeError('Incomplete system BLAS install detected. Perhaps you need to install blas-dev or blas-devel package - that contains /usr/lib/libblas.so using apt or yum or equivalent package manager?')
      if hasattr(self.compilers, 'FC') and (self.defaultPrecision != '__float128') and (self.defaultPrecision != '__fp16') : pkg = 'fblaslapack'
      else: pkg = 'f2cblaslapack'
      raise RuntimeError('Could not find a functional BLAS. Run with --with-blas-lib=<lib> to indicate the library containing BLAS.\n Or --download-'+pkg+'=1 to have one automatically downloaded and installed\n')
    if not self.foundLapack:
      # check for split blas/blas-dev packages
      import glob
      llib = glob.glob('/usr/lib/liblapack.*')
      if llib != [] and not (os.path.isfile('/usr/lib/liblapack.so') or os.path.isfile('/usr/lib/liblapack.a')):
        raise RuntimeError('Incomplete system LAPACK install detected. Perhaps you need to install lapack-dev or lapack-devel package - that contains /usr/lib/liblapack.so using apt or yum or equivalent package manager?')
      if hasattr(self.compilers, 'FC') and (self.defaultPrecision != '__float128') and (self.defaultPrecision != '__fp16') : pkg = 'fblaslapack'
      else: pkg = 'f2cblaslapack'
      raise RuntimeError('Could not find a functional LAPACK. Run with --with-lapack-lib=<lib> to indicate the library containing LAPACK.\n Or --download-'+pkg+'=1 to have one automatically downloaded and installed\n')

    #  allow user to dictate which blas/lapack mangling to use (some blas/lapack libraries, like on Apple, provide several)
    if 'known-blaslapack-mangling' in self.argDB:
      self.mangling = self.argDB['known-blaslapack-mangling']

    if self.mangling == 'underscore':
      self.addDefine('BLASLAPACK_UNDERSCORE', 1)
    elif self.mangling == 'caps':
      self.addDefine('BLASLAPACK_CAPS', 1)

    if self.suffix != '':
      self.addDefine('BLASLAPACK_SUFFIX', self.suffix)

    self.found = 1
    if not self.f2cblaslapack.found and not self.fblaslapack.found:
      self.executeTest(self.checkMKL)
      if not self.mkl:
        self.executeTest(self.checkESSL)
        self.executeTest(self.checkPESSL)
        self.executeTest(self.checkMissing)
    self.executeTest(self.checklsame)
    if self.argDB['with-shared-libraries']:
      symbol = 'dgeev'+self.suffix
      if self.f2c:
        if self.mangling == 'underscore': symbol = symbol+'_'
      elif hasattr(self.compilers, 'FC'):
        symbol = self.compilers.mangleFortranFunction(symbol)
      if not self.setCompilers.checkIntoShared(symbol,self.lapackLibrary+self.getOtherLibs()):
        raise RuntimeError('The BLAS/LAPACK libraries '+self.libraries.toStringNoDupes(self.lapackLibrary+self.getOtherLibs())+'\ncannot be used with a shared library\nEither run ./configure with --with-shared-libraries=0 or use a different BLAS/LAPACK library');
    self.executeTest(self.checkRuntimeIssues)
    if self.mkl and self.has64bitindices:
      self.addDefine('HAVE_MKL_INTEL_ILP64',1)
    if self.argDB['with-64-bit-blas-indices'] and not self.has64bitindices:
      raise RuntimeError('You requested 64 bit integer BLAS/LAPACK using --with-64-bit-blas-indices but they are not available given your other BLAS/LAPACK options')

  def checkMKL(self):
    '''Check for Intel MKL library'''
    self.libraries.saveLog()
    if self.libraries.check(self.dlib, 'mkl_set_num_threads') and not self.libraries.check(self.dlib, 'flexiblas_avail'):
      self.mkl = 1
      self.addDefine('HAVE_MKL_LIBS',1)
      '''Set include directory for mkl.h and friends'''
      '''(the include directory is in CPATH if mklvars.sh has been sourced.'''
      ''' if the script hasn't been sourced, we still try to pick up the include dir)'''
      if 'with-blaslapack-include' in self.argDB:
        incl = self.argDB['with-blaslapack-include']
        if not isinstance(incl, list): incl = [incl]
        self.include = incl
      if self.checkCompile('#include "mkl_spblas.h"',''):
        self.mkl_spblas_h = 1
        self.logPrint('MKL mkl_spblas.h found in default include path.')
      else:
        self.logPrint('MKL include path not automatically picked up by compiler. Trying to find mkl_spblas.h...')
        if 'with-blaslapack-dir' in self.argDB:
          pathlist = [os.path.join(self.argDB['with-blaslapack-dir'],'include'),
                      os.path.join(self.argDB['with-blaslapack-dir'],'..','include'),
                      os.path.join(self.argDB['with-blaslapack-dir'],'..','..','include')]
        elif 'with-blaslapack-include' in self.argDB:
          pathlist = self.include
        else:
          pathlist = []
        for path in pathlist:
          if os.path.isdir(path) and self.checkInclude([path], ['mkl_spblas.h']):
            self.include = [path]
            self.mkl_spblas_h = 1
            self.logPrint('MKL mkl_spblas.h found at:'+path)
            break

        if not self.mkl_spblas_h:
          self.logPrint('Unable to find MKL include directory!')
        else:
          self.logPrint('MKL include path set to ' + str(self.include))
      self.versionname    = 'INTEL_MKL_VERSION'
      self.versioninclude = 'mkl_version.h'
      self.versiontitle   = 'Intel MKL Version'
      self.checkVersion()
      if self.foundversion:
        self.addDefine('HAVE_MKL_INCLUDES',1)
    self.logWrite(self.libraries.restoreLog())
    return

  def checkESSL(self):
    '''Check for the IBM ESSL library'''
    self.libraries.saveLog()
    if self.libraries.check(self.dlib, 'iessl'):
      self.essl = 1
      self.addDefine('HAVE_ESSL',1)

      if 'with-blaslapack-include' in self.argDB:
        incl = self.argDB['with-blaslapack-include']
        if not isinstance(incl, list): incl = [incl]
      elif 'with-blaslapack-dir' in self.argDB:
        incl = [os.path.join(self.argDB['with-blaslapack-dir'],'include')]
      else:
        return
      linc = self.include + incl
      if self.checkInclude(linc, ['essl.h']):
        self.include = linc
    self.logWrite(self.libraries.restoreLog())
    return

  def checkPESSL(self):
    '''Check for the IBM PESSL library - and error out - if used instead of ESSL'''
    self.libraries.saveLog()
    if self.libraries.check(self.dlib, 'ipessl'):
      self.logWrite(self.libraries.restoreLog())
      raise RuntimeError('Cannot use PESSL instead of ESSL!')
    self.logWrite(self.libraries.restoreLog())
    return

  def mangleBlas(self, baseName):
    prefix = self.getPrefix()
    if self.f2c and self.mangling == 'underscore':
      return prefix+baseName+self.suffix+'_'
    else:
      return prefix+baseName+self.suffix

  def mangleBlasNoPrefix(self, baseName):
    if self.f2c:
      if self.mangling == 'underscore':
        return baseName+self.suffix+'_'
      else:
        return baseName+self.suffix
    else:
      return self.compilers.mangleFortranFunction(baseName+self.suffix)

  def checkMissing(self):
    '''Check for missing LAPACK routines'''
    if self.foundLapack:
      mangleFunc = hasattr(self.compilers, 'FC') and not self.f2c
    routines = ['gelss','gerfs','gges','hgeqz','hseqr','orgqr','ormqr','stebz',
                'stegr','stein','steqr','sytri','tgsen','trsen','trtrs','geqp3']
    self.libraries.saveLog()
    oldLibs = self.compilers.LIBS
    found, missing = self.libraries.checkClassify(self.lapackLibrary, map(self.mangleBlas,routines), otherLibs = self.getOtherLibs(), fortranMangle = mangleFunc)
    for baseName in routines:
      if self.mangleBlas(baseName) in missing:
        self.missingRoutines.append(baseName)
        self.addDefine('MISSING_LAPACK_'+baseName.upper(), 1)
    self.compilers.LIBS = oldLibs
    self.logWrite(self.libraries.restoreLog())

  def checklsame(self):
    ''' Do the BLAS/LAPACK libraries have a valid lsame() function with correct binding.'''
    routine = 'lsame';
    if self.f2c:
      if self.mangling == 'underscore':
        routine = routine + self.suffix + '_'
    else:
      routine = self.compilers.mangleFortranFunction(routine)
    self.libraries.saveLog()
    if not self.libraries.check(self.dlib,routine,fortranMangle = 0):
      self.addDefine('MISSING_LAPACK_'+routine, 1)
    self.logWrite(self.libraries.restoreLog())

  def checkForRoutine(self,routine):
    ''' used by other packages to see if a BLAS routine is available
        This is not really correct because other packages do not (usually) know about f2cblasLapack'''
    self.libraries.saveLog()
    if self.f2c:
      if self.mangling == 'underscore':
        ret = self.libraries.check(self.dlib,routine+self.suffix+'_')
      else:
        ret = self.libraries.check(self.dlib,routine+self.suffix)
    else:
      ret = self.libraries.check(self.dlib,routine,fortranMangle = hasattr(self.compilers, 'FC'))
    self.logWrite(self.libraries.restoreLog())
    return ret

  def runTimeTest(self,name,includes,body,lib = None,nobatch=0):
    '''Either runs a test or adds it to the batch of runtime tests'''
    if name in self.framework.clArgDB: return self.argDB[name]
    if self.argDB['with-batch']:
      if nobatch:
        raise RuntimeError('In batch mode you must provide the value for --'+name)
      else:
        self.framework.addBatchInclude(includes)
        self.framework.addBatchBody(body)
        if lib: self.framework.addBatchLib(lib)
        if self.include: self.framework.batchIncludeDirs.extend([self.headers.getIncludeArgument(inc) for inc in self.include])
        return None
    else:
      result = None
      self.pushLanguage('C')
      filename = 'runtimetestoutput'
      body = '''FILE *output = fopen("'''+filename+'''","w");\n'''+body
      if lib:
        if not isinstance(lib, list): lib = [lib]
        oldLibs  = self.compilers.LIBS
        self.compilers.LIBS = self.libraries.toString(lib)+' '+self.compilers.LIBS
      if self.checkRun(includes, body) and os.path.exists(filename):
        f    = open(filename)
        out  = f.read()
        f.close()
        os.remove(filename)
        result = out.split("=")[1].split("'")[0]
      self.popLanguage()
      if lib:
        self.compilers.LIBS = oldLibs
      return result

  def checkRuntimeIssues(self):
    '''Determines if BLAS/LAPACK routines use 32 or 64 bit integers'''
    if self.known64 == '64':
      self.addDefine('HAVE_64BIT_BLAS_INDICES', 1)
      self.has64bitindices = 1
      self.log.write('64 bit blas indices based on the BLAS/LAPACK library being used\n')
    elif self.known64 == '32':
      self.log.write('32 bit blas indices based on the BLAS/LAPACK library being used\n')
    elif 'known-64-bit-blas-indices' in self.argDB:
      if self.argDB['known-64-bit-blas-indices']:
        self.addDefine('HAVE_64BIT_BLAS_INDICES', 1)
        self.has64bitindices = 1
      else:
        self.has64bitindices = 0
    elif self.argDB['with-batch']:
      self.logPrintWarning('Cannot determine if BLAS/LAPACK uses 32 bit or 64 bit integers \
in batch-mode! Assuming 32 bit integers. Run with --known-64-bit-blas-indices \
if you know they are 64 bit. Run with --known-64-bit-blas-indices=0 to remove \
this warning message')
      self.has64bitindices = 0
      self.log.write('In batch mode with unknown size of BLAS/LAPACK defaulting to 32 bit\n')
    else:
      includes = '''#include <sys/types.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include <stddef.h>\n\n'''
      t = self.getType()
      body     = '''extern '''+t+''' '''+self.getPrefix()+self.mangleBlasNoPrefix('dot')+'''(const int*,const '''+t+'''*,const int *,const '''+t+'''*,const int*);
                  '''+t+''' x1mkl[4] = {3.0,5.0,7.0,9.0};
                  int one1mkl = 1,nmkl = 2;
                  '''+t+''' dotresultmkl = 0;
                  dotresultmkl = '''+self.getPrefix()+self.mangleBlasNoPrefix('dot')+'''(&nmkl,x1mkl,&one1mkl,x1mkl,&one1mkl);
                  fprintf(output, "-known-64-bit-blas-indices=%d",dotresultmkl != 34);'''
      result = self.runTimeTest('known-64-bit-blas-indices',includes,body,self.dlib,nobatch=1)
      if result is not None:
        self.log.write('Checking for 64 bit blas indices: result ' +str(result)+'\n')
        result = int(result)
        if result:
          if self.defaultPrecision == 'single':
            self.log.write('Checking for 64 bit blas indices: special check for Apple single precision\n')
            # On Apple single precision sdot() returns a double so we need to test that case
            body     = '''extern double '''+self.getPrefix()+self.mangleBlasNoPrefix('dot')+'''(const int*,const '''+t+'''*,const int *,const '''+t+'''*,const int*);
                  '''+t+''' x1mkl[4] = {3.0,5.0,7.0,9.0};
                  int one1mkl = 1,nmkl = 2;
                  double dotresultmkl = 0;
                  dotresultmkl = '''+self.getPrefix()+self.mangleBlasNoPrefix('dot')+'''(&nmkl,x1mkl,&one1mkl,x1mkl,&one1mkl);
                  fprintf(output, "--known-64-bit-blas-indices=%d",dotresultmkl != 34);'''
            result = self.runTimeTest('known-64-bit-blas-indices',includes,body,self.dlib,nobatch=1)
            result = int(result)
        if result:
          self.addDefine('HAVE_64BIT_BLAS_INDICES', 1)
          self.has64bitindices = 1
          self.log.write('Checking for 64 bit blas indices: result not equal to 1 so assuming 64 bit blas indices\n')
      else:
        self.addDefine('HAVE_64BIT_BLAS_INDICES', 1)
        self.has64bitindices = 1
        self.log.write('Checking for 64 bit blas indices: program did not return therefore assuming 64 bit blas indices\n')
    self.log.write('Checking if sdot() returns a float or a double\n')
    if 'known-sdot-returns-double' in self.argDB:
      if self.argDB['known-sdot-returns-double']:
        self.addDefine('BLASLAPACK_SDOT_RETURNS_DOUBLE', 1)
    elif self.argDB['with-batch']:
      self.logPrintWarning('Cannot determine if BLAS sdot() returns a float or a double \
in batch-mode! Assuming float. Run with --known-sdot-returns-double=1 \
if you know it returns a double (very unlikely). Run with \
--known-sdor-returns-double=0 to remove this warning message')
    else:
      includes = '''#include <sys/types.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include <stddef.h>\n'''
      body     = '''extern float '''+self.mangleBlasNoPrefix('sdot')+'''(const int*,const float*,const int *,const float*,const int*);
                  float x1[1] = {3.0};
                  int one1 = 1;
                  long long int ione1 = 1;
                  float sdotresult = 0;
                  int blasint64 = '''+str(self.has64bitindices)+''';\n
                  if (!blasint64) {
                       sdotresult = '''+self.mangleBlasNoPrefix('sdot')+'''(&one1,x1,&one1,x1,&one1);
                     } else {
                       sdotresult = '''+self.mangleBlasNoPrefix('sdot')+'''((const int*)&ione1,x1,(const int*)&ione1,x1,(const int*)&ione1);
                     }
                  fprintf(output, "--known-sdot-returns-double=%d",sdotresult != 9);\n'''
      result = self.runTimeTest('known-sdot-returns-double',includes,body,self.dlib,nobatch=1)
      if result:
        self.log.write('Checking for sdot() return double: result ' +str(result)+'\n')
        result = int(result)
        if result:
          self.addDefine('BLASLAPACK_SDOT_RETURNS_DOUBLE', 1)
          self.log.write('Checking sdot(): Program did return with not 1 for output so assume returns double\n')
      else:
        self.log.write('Checking sdot(): Program did not return with output so assume returns single\n')
    self.log.write('Checking if snrm() returns a float or a double\n')
    if 'known-snrm2-returns-double' in self.argDB:
      if self.argDB['known-snrm2-returns-double']:
        self.addDefine('BLASLAPACK_SNRM2_RETURNS_DOUBLE', 1)
    elif self.argDB['with-batch']:
      self.logPrintWarning('Cannot determine if BLAS snrm2() returns a float or a double \
in batch-mode! Assuming float. Run with --known-snrm2-returns-double=1 \
if you know it returns a double (very unlikely). Run with \
--known-snrm2-returns-double=0 to remove this warning message')
    else:
      includes = '''#include <sys/types.h>\n#include <stdlib.h>\n#include <stdio.h>\n#include <stddef.h>\n'''
      body     = '''extern float '''+self.mangleBlasNoPrefix('snrm2')+'''(const int*,const float*,const int*);
                  float x2[1] = {3.0};
                  int one2 = 1;
                  long long int ione2 = 1;
                  float normresult = 0;
                  int blasint64 = '''+str(self.has64bitindices)+''';\n
                  if (!blasint64) {
                       normresult = '''+self.mangleBlasNoPrefix('snrm2')+'''(&one2,x2,&one2);
                     } else {
                       normresult = '''+self.mangleBlasNoPrefix('snrm2')+'''((const int*)&ione2,x2,(const int*)&ione2);
                     }
                  fprintf(output, "--known-snrm2-returns-double=%d",normresult != 3);\n'''
      result = self.runTimeTest('known-snrm2-returns-double',includes,body,self.dlib,nobatch=1)
      if result:
        self.log.write('Checking for snrm2() return double: result ' +str(result)+'\n')
        result = int(result)
        if result:
          self.log.write('Checking snrm2(): Program did return with 1 for output so assume returns double\n')
          self.addDefine('BLASLAPACK_SNRM2_RETURNS_DOUBLE', 1)
      else:
        self.log.write('Checking snrm2(): Program did not return with output so assume returns single\n')
