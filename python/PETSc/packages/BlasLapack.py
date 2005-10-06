from __future__ import generators
import user
import config.base
import md5
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.headerPrefix    = ''
    self.substPrefix     = ''
    self.argDB           = framework.argDB
    self.found           = 0
    self.f2c             = 0
    self.fblaslapack     = 0
    self.missingRoutines = []
    self.separateBlas    = 1
    return

  def __str__(self):
    return 'BLAS/LAPACK: '+self.libraries.toString(self.lib)+'\n'

  def setupHelp(self, help):
    import nargs
    help.addArgument('BLAS/LAPACK', '-with-blas-lapack-dir=<dir>',                nargs.ArgDir(None, None, 'Indicate the directory containing BLAS and LAPACK libraries'))
    help.addArgument('BLAS/LAPACK', '-with-blas-lapack-lib=<lib>',                nargs.Arg(None, None, 'Indicate the library containing BLAS and LAPACK'))
    help.addArgument('BLAS/LAPACK', '-with-blas-lib=<lib>',                       nargs.Arg(None, None, 'Indicate the library(s) containing BLAS'))
    help.addArgument('BLAS/LAPACK', '-with-lapack-lib=<lib>',                     nargs.Arg(None, None, 'Indicate the library(s) containing LAPACK'))
    help.addArgument('BLAS/LAPACK', '-download-c-blas-lapack=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Automatically install a C version of BLAS/LAPACK'))
    help.addArgument('BLAS/LAPACK', '-download-f-blas-lapack=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Automatically install a Fortran version of BLAS/LAPACK'))
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    framework.require('PETSc.packages.sowing', self)
    self.libraries     = self.framework.require('config.libraries',self)
    return

  def getChecksum(self,source, chunkSize = 1024*1024):
    '''Return the md5 checksum for a given file, which may also be specified by its filename
       - The chunkSize argument specifies the size of blocks read from the file'''
    if isinstance(source, file):
      f = source
    else:
      f = file(source)
    m = md5.new()
    size = chunkSize
    buf  = f.read(size)
    while buf:
      m.update(buf)
      buf = f.read(size)
    f.close()
    return m.hexdigest()

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
        otherLibs = blasLibrary
    if self.useCompatibilityLibs:
      otherLibs += self.compilers.flibs
    return otherLibs

  def checkBlas(self, blasLibrary, otherLibs, fortranMangle, routine = 'ddot'):
    '''This checks the given library for the routine, ddot by default'''
    oldLibs = self.compilers.LIBS
    prototype = ''
    call      = ''
    if fortranMangle=='stdcall':
      if routine=='ddot':
        prototype = 'double __stdcall DDOT(int*,double*,int*,double*,int*);'
        call      = 'DDOT(0,0,0,0,0,0);'
    found   = self.libraries.check(blasLibrary, routine, otherLibs = otherLibs, fortranMangle = fortranMangle, prototype = prototype, call = call)
    self.compilers.LIBS = oldLibs
    return found

  def checkLapack(self, lapackLibrary, otherLibs, fortranMangle, routines = ['dgetrs', 'dgeev']):
    oldLibs = self.compilers.LIBS
    found   = 0
    prototypes = ['','']
    calls      = ['','']
    if fortranMangle=='stdcall':
      if routines == ['dgetrs','dgeev']:
        prototypes = ['void __stdcall DGETRS(char*,int,int*,int*,double*,int*,int*,double*,int*,int*);',
                      'void __stdcall DGEEV(char*,int,char*,int,int*,double*,int*,PetscReal*,PetscReal*,double*,int*,double*,int*,double*,int*,int*);']
        calls      = ['DGETRS(0,0,0,0,0,0,0,0,0,0);',
                      'DGEEV(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);']
    for routine, prototype, call in zip(routines, prototypes, calls):
      found = found or self.libraries.check(lapackLibrary, routine, otherLibs = otherLibs, fortranMangle = fortranMangle, prototype = prototype, call = call)
      if found: break
    self.compilers.LIBS = oldLibs
    return found

  def checkLib(self, lapackLibrary, blasLibrary = None):
    '''Checking for BLAS and LAPACK symbols'''

    #check for BLASLAPACK_STDCALL calling convention!!!!
    
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
    mangleFunc = self.compilers.fortranMangling
    foundBlas = self.checkBlas(blasLibrary, self.getOtherLibs(foundBlas, blasLibrary), mangleFunc)
    if foundBlas:
      foundLapack = self.checkLapack(lapackLibrary, self.getOtherLibs(foundBlas, blasLibrary), mangleFunc)
    else:
      foundBlas = self.checkBlas(blasLibrary, self.getOtherLibs(foundBlas, blasLibrary), 0, 'ddot_')
      if foundBlas:
        foundLapack = self.checkLapack(lapackLibrary, self.getOtherLibs(foundBlas, blasLibrary), 0, ['dgetrs_', 'dgeev_'])
        if foundLapack:
          self.f2c = 1
    return (foundBlas, foundLapack)

  def generateGuesses(self):
    # check that user has used the options properly
    if 'with-blas-lib' in self.framework.argDB and not 'with-lapack-lib' in self.framework.argDB:
      raise RuntimeError('If you use the --with-blas-lib=<lib> you must also use --with-lapack-lib=<lib> option')
    if not 'with-blas-lib' in self.framework.argDB and 'with-lapack-lib' in self.framework.argDB:
      raise RuntimeError('If you use the --with-lapack-lib=<lib> you must also use --with-blas-lib=<lib> option')
    if 'with-blas-lib' in self.framework.argDB and 'with-blas-lapack-dir' in self.framework.argDB:
      raise RuntimeError('You cannot set both the library containing BLAS with --with-blas-lib=<lib>\nand the directory to search with --with-blas-lapack-dir=<dir>')
    if 'with-blas-lapack-lib' in self.framework.argDB and 'with-blas-lapack-dir' in self.framework.argDB:
      raise RuntimeError('You cannot set both the library containing BLAS/LAPACK with --with-blas-lapack-lib=<lib>\nand the directory to search with --with-blas-lapack-dir=<dir>')

    if self.framework.argDB['download-c-blas-lapack'] == 1:
      if hasattr(self.compilers, 'FC'):
        raise RuntimeError('Should request f-blas-lapack, not --download-c-blas-lapack=yes since you have a fortran compiler?')
      libdir = self.downLoadBlasLapack('f2c', 'c')
      yield ('Downloaded BLAS/LAPACK library', [os.path.join(libdir,'libf2cblas.a')]+self.libraries.math, os.path.join(libdir,'libf2clapack.a'), 0)
      raise RuntimeError('Could not use downloaded c-blas-lapack?')
    if self.framework.argDB['download-f-blas-lapack'] == 1:
      if not hasattr(self.compilers, 'FC'):
        raise RuntimeError('Cannot request f-blas-lapack without Fortran compiler, maybe you want --download-c-blas-lapack=1?')
      libdir = self.downLoadBlasLapack('f','f')            
      yield ('Downloaded BLAS/LAPACK library', os.path.join(libdir,'libfblas.a'), os.path.join(libdir,'libflapack.a'), 1)
      self.fblaslapack = 1
      raise RuntimeError('Could not use downloaded f-blas-lapack?')
    # Try specified BLASLAPACK library
    if 'with-blas-lapack-lib' in self.framework.argDB:
      yield ('User specified BLAS/LAPACK library', None, self.framework.argDB['with-blas-lapack-lib'], 1)
      raise RuntimeError('You set a value for --with-blas-lapack-lib=<lib>, but '+str(self.framework.argDB['with-blas-lapack-lib'])+' cannot be used\n')
    # Try specified BLAS and LAPACK libraries
    if 'with-blas-lib' in self.framework.argDB and 'with-lapack-lib' in self.framework.argDB:
      yield ('User specified BLAS and LAPACK libraries', self.framework.argDB['with-blas-lib'], self.framework.argDB['with-lapack-lib'], 1)
      raise RuntimeError('You set a value for --with-blas-lib=<lib> and --with-lapack-lib=<lib>, but '+str(self.framework.argDB['with-blas-lib'])+' and '+str(self.framework.argDB['with-lapack-lib'])+' cannot be used\n')
    # Try specified installation root
    if 'with-blas-lapack-dir' in self.framework.argDB:
      dir = self.framework.argDB['with-blas-lapack-dir']
      if not (len(dir) > 2 and dir[1] == ':') :
        dir = os.path.abspath(dir)
      yield ('User specified installation root (HPUX)', os.path.join(dir, 'libveclib.a'),  os.path.join(dir, 'liblapack.a'), 1)
      yield ('User specified installation root (F2C)', [os.path.join(dir, 'libf2cblas.a')]+self.libraries.math, os.path.join(dir, 'libf2clapack.a'), 1)
      yield ('User specified installation root', os.path.join(dir, 'libfblas.a'),   os.path.join(dir, 'libflapack.a'), 1)
      yield ('User specified ATLAS Linux installation root', [os.path.join(dir, 'libcblas.a'),os.path.join(dir, 'libf77blas.a'), os.path.join(dir, 'libatlas.a')],  [os.path.join(dir, 'liblapack.a')], 1)
      yield ('User specified ATLAS Linux installation root', [os.path.join(dir, 'libf77blas.a'), os.path.join(dir, 'libatlas.a')],  [os.path.join(dir, 'liblapack.a')], 1)
      # Check Linux MKL variations
      yield ('User specified MKL Linux-x86 lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), 'libmkl_def.a', 'guide', 'pthread'], 1)
      yield ('User specified MKL Linux-ia64 lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), 'libmkl_ipf.a', 'guide', 'pthread'], 1)
      yield ('User specified MKL Linux-em64t lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), 'libmkl_em64t.a', 'guide', 'pthread'], 1)
      yield ('User specified MKL Linux-x86 installation root', None, [os.path.join(dir,'lib','32','libmkl_lapack.a'),'libmkl_def.a', 'guide', 'pthread'], 1)
      yield ('User specified MKL Linux-ia64 installation root', None, [os.path.join(dir,'lib','64','libmkl_lapack.a'),'libmkl_ipf.a', 'guide', 'pthread'], 1)
      yield ('User specified MKL Linux-em64t installation root', None, [os.path.join(dir,'lib','em64t','libmkl_lapack.a'),'libmkl_em64t.a', 'guide', 'pthread'], 1)
      if self.setCompilers.use64BitPointers:
        mkldir = os.path.join(dir, 'ia64', 'lib')
      else:
        mkldir = os.path.join(dir, 'ia32', 'lib')
      yield ('User specified MKL Windows installation root', None, [os.path.join(mkldir, 'mkl_c_dll.lib')], 1)
      yield ('User specified MKL Windows lib dir', None, [os.path.join(dir, 'mkl_c_dll.lib')], 1)
      yield ('User specified stdcall MKL Windows installation root', None, [os.path.join(mkldir, 'mkl_s_dll.lib')], 1)
      # Search for liblapack.a and libblas.a after the implementations with more specific name to avoid
      # finding these in /usr/lib despite using -L<blas-lapack-dir> while attempting to get a different library.
      yield ('User specified installation root', os.path.join(dir, 'libblas.a'),    os.path.join(dir, 'liblapack.a'), 1)
      raise RuntimeError('You set a value for --with-blas-lapack-dir=<dir>, but '+self.framework.argDB['with-blas-lapack-dir']+' cannot be used\n')
    # IRIX locations
    yield ('IRIX Mathematics library', None, 'libcomplib.sgimath.a', 1)
    yield ('Another IRIX Mathematics library', None, 'libscs.a', 1)
    # IBM ESSL locations
    yield ('IBM ESSL Mathematics library', None, 'libessl.a', 1)
    # Portland group compiler blas and lapack
    if 'PGI' in os.environ:
      dir = os.path.join(os.environ['PGI'],'linux86','5.1','lib')
      yield ('User specified installation root', os.path.join(dir, 'libblas.a'), os.path.join(dir, 'liblapack.a'), 1)
      dir = os.path.join(os.environ['PGI'],'linux86','5.0','lib')
      yield ('User specified installation root', os.path.join(dir, 'libblas.a'), os.path.join(dir, 'liblapack.a'), 1)
      dir = os.path.join(os.environ['PGI'],'linux86','lib')
      yield ('User specified installation root', os.path.join(dir, 'libblas.a'), os.path.join(dir, 'liblapack.a'), 1)
    # Try compiler defaults
    yield ('Default compiler locations', 'libblas.a', 'liblapack.a', 1)
    yield ('HPUX', 'libveclib.a', 'liblapack.a', 1)
    # /usr/local/lib
    dir = os.path.join('/usr','local','lib')
    yield ('Default compiler locations /usr/local/lib', os.path.join(dir,'libblas.a'), os.path.join(dir,'liblapack.a'), 1)
    yield ('Default compiler locations with G77', None, ['liblapack.a', 'libblas.a','libg2c.a'], 1)
    # Try MacOSX location
    yield ('MacOSX BLAS/LAPACK library', None, os.path.join('/System', 'Library', 'Frameworks', 'vecLib.framework', 'vecLib'), 1)
    # Sun locations
    yield ('Sun sunperf BLAS/LAPACK library', None, ['libsunperf.a','libsunmath.a','libm.a'], 1)
    yield ('Sun sunperf BLAS/LAPACK library', None, ['libsunperf.a','libF77.a','libM77.a','libsunmath.a','libm.a'], 1)
    yield ('Sun sunperf BLAS/LAPACK library', None, ['libsunperf.a','libfui.a','libfsu.a','libsunmath.a','libm.a'], 1)
    # Try Microsoft Windows location
    for MKL_Version in [os.path.join('MKL','8.0'),'MKL72','MKL70','MKL61','MKL']:
      MKL_Dir = os.path.join('/cygdrive', 'c', 'Program\\ Files', 'Intel', MKL_Version)
      if self.setCompilers.use64BitPointers:
        MKL_Dir = os.path.join(MKL_Dir, 'ia64', 'lib')
      else:
        MKL_Dir = os.path.join(MKL_Dir, 'ia32', 'lib')
      yield ('Microsoft Windows, Intel MKL library', None, os.path.join(MKL_Dir,'mkl_c_dll.lib'), 1)
      yield ('Microsoft Windows, Intel MKL stdcall library', None, os.path.join(MKL_Dir,'mkl_s_dll.lib'), 1)
    # Try PETSc location
    if self.petscdir.dir and self.arch.arch:
      dir1 = os.path.abspath(os.path.join(self.petscdir.dir, '..', 'blaslapack', 'lib'))
      yield ('PETSc location 1', os.path.join(dir1, 'libblas.a'), os.path.join(dir1, 'liblapack.a'), 1)
      dir2 = os.path.join(dir1, 'libg_c++', self.arch.arch)
      yield ('PETSc location 2', os.path.join(dir2, 'libblas.a'), os.path.join(dir2, 'liblapack.a'), 1)
      dir3 = os.path.join(dir1, 'libO_c++', self.arch.arch)
      yield ('PETSc location 3', os.path.join(dir3, 'libblas.a'), os.path.join(dir3, 'liblapack.a'), 1)
    if self.framework.argDB['download-c-blas-lapack'] == 2:
      if hasattr(self.compilers, 'FC'):
        raise RuntimeError('Should request f-blas-lapack, not --download-c-blas-lapack=yes since you have a fortran compiler?')
      libdir = self.downLoadBlasLapack('f2c', 'c')
      yield ('Downloaded BLAS/LAPACK library', [os.path.join(libdir,'libf2cblas.a')]+self.libraries.math, os.path.join(libdir,'libf2clapack.a'), 0)
    if self.framework.argDB['download-f-blas-lapack'] == 2:
      if not hasattr(self.compilers, 'FC'):
        raise RuntimeError('Cannot request f-blas-lapack without Fortran compiler, maybe you want --download-c-blas-lapack=1?')
      libdir = self.downLoadBlasLapack('f','f')            
      yield ('Downloaded BLAS/LAPACK library', os.path.join(libdir,'libfblas.a'), os.path.join(libdir,'libflapack.a'), 1)
    return

  def getSharedFlag(self,cflags):
    for flag in ['-PIC', '-fPIC', '-KPIC']:
      if cflags.find(flag) >=0: return flag
    return ''
      
  def downLoadBlasLapack(self, f2c, l):
    self.framework.log.write('Downloading '+l+'blaslapack\n')
    packages = self.petscdir.externalPackagesDir
    if not os.path.isdir(packages):
      os.mkdir(packages)
    if f2c == 'f2c':
      self.f2c = 1
    libdir = os.path.join(packages,f2c+'blaslapack',self.arch.arch)
    if not os.path.isdir(os.path.join(packages,f2c+'blaslapack')):
      self.framework.log.write('Actually need to ftp '+l+'blaslapack\n')
      import urllib
      try:
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/'+f2c+'blaslapack.tar.gz',os.path.join(packages,f2c+'blaslapack.tar.gz'))
      except:
        raise RuntimeError('Error downloading '+f2c+'blaslapack.tar.gz requested with -with-'+l+'-blas-lapack option')
      try:
        self.executeShellCommand('cd '+packages+'; gunzip '+f2c+'blaslapack.tar.gz', log = self.framework.log, timeout = 360.0)
      except:
        raise RuntimeError('Error unzipping '+f2c+'blaslapack.tar.gz requested with -with-'+l+'-blas-lapack option')
      try:
        self.executeShellCommand('cd '+packages+'; tar -xf '+f2c+'blaslapack.tar', log = self.framework.log, timeout = 360.0)
      except:
        raise RuntimeError('Error doing tar -xf '+f2c+'blaslapack.tar requested with -with-'+l+'-blas-lapack option')
      os.unlink(os.path.join(packages,f2c+'blaslapack.tar'))
      self.framework.actions.addArgument('BLAS/LAPACK', 'Download', 'Downloaded PETSc '+f2c+'blaslapack into '+os.path.dirname(libdir))
    else:
      self.framework.log.write('Found '+l+'blaslapack, do not need to download\n')
    if not os.path.isdir(libdir):
      os.mkdir(libdir)
    blasDir = os.path.join(packages,f2c+'blaslapack')
    g = open(os.path.join(blasDir,'tmpmakefile'),'w')
    f = open(os.path.join(blasDir,'makefile'),'r')    
    line = f.readline()
    while line:
      if line.startswith('CC  '):
        cc = self.compilers.CC
        line = 'CC = '+cc+'\n'
      if line.startswith('COPTFLAGS '):
        self.setCompilers.pushLanguage('C')
        line = 'COPTFLAGS  = '+self.setCompilers.getCompilerFlags()+'\n'
        self.setCompilers.popLanguage()
      if line.startswith('CNOOPT'):
        self.setCompilers.pushLanguage('C')
        line = 'CNOOPT = '+self.getSharedFlag(self.setCompilers.getCompilerFlags())+'\n'
        self.setCompilers.popLanguage()
      if line.startswith('FC  '):
        fc = self.compilers.FC
        if fc.find('f90') >= 0:
          import commands
          output  = commands.getoutput(fc+' -v')
          if output.find('IBM') >= 0:
            fc = os.path.join(os.path.dirname(fc),'xlf')
            self.framework.log.write('Using IBM f90 compiler for PETSc, switching to xlf for compiling BLAS/LAPACK\n')
        line = 'FC = '+fc+'\n'
      if line.startswith('FOPTFLAGS '):
        self.setCompilers.pushLanguage('FC')
        line = 'FOPTFLAGS  = '+self.setCompilers.getCompilerFlags().replace('-Mfree','')+'\n'
        self.setCompilers.popLanguage()       
      if line.startswith('FNOOPT'):
        self.setCompilers.pushLanguage('FC')
        line = 'FNOOPT = '+self.getSharedFlag(self.setCompilers.getCompilerFlags())+'\n'
        self.setCompilers.popLanguage()
      if line.startswith('AR  '):
        line = 'AR      = '+self.setCompilers.AR+'\n'
      if line.startswith('AR_FLAGS  '):
        line = 'AR_FLAGS      = '+self.setCompilers.AR_FLAGS+'\n'
      if line.startswith('LIB_SUFFIX '):
        line = 'LIB_SUFFIX = '+self.setCompilers.AR_LIB_SUFFIX+'\n'
      if line.startswith('RANLIB  '):
        line = 'RANLIB = '+self.setCompilers.RANLIB+'\n'
      if line.startswith('RM  '):
        line = 'RM = '+self.programs.RM+'\n'
      

      if line.startswith('include'):
        line = '\n'
      g.write(line)
      line = f.readline()
    f.close()
    g.close()
    if os.path.isfile(os.path.join(libdir,'tmpmakefile')) and (self.getChecksum(os.path.join(libdir,'tmpmakefile')) == self.getChecksum(os.path.join(blasDir,'tmpmakefile'))):
      self.framework.log.write('Do not need to compile '+l+'blaslapack, already compiled\n')
      return libdir
    try:
      self.logPrintBox('Compiling '+l.upper()+'BLASLAPACK; this may take several minutes')
      output  = config.base.Configure.executeShellCommand('cd '+blasDir+';make -f tmpmakefile cleanblaslapck cleanlib; make -f tmpmakefile', timeout=800, log = self.framework.log)[0]
    except RuntimeError, e:
      raise RuntimeError('Error running make on '+l+'blaslapack: '+str(e))
    try:
      output  = config.base.Configure.executeShellCommand('cd '+blasDir+';mv -f lib'+f2c+'blas.'+self.setCompilers.AR_LIB_SUFFIX+' lib'+f2c+'lapack.'+self.setCompilers.AR_LIB_SUFFIX+' '+self.arch.arch, timeout=30, log = self.framework.log)[0]
    except RuntimeError, e:
      raise RuntimeError('Error moving '+l+'blaslapack libraries: '+str(e))
    try:
      output  = config.base.Configure.executeShellCommand('cd '+blasDir+';cp -f tmpmakefile '+self.arch.arch, timeout=30, log = self.framework.log)[0]
    except RuntimeError, e:
      pass
    return libdir
  
  def configureLibrary(self):
    self.functionalBlasLapack = []
    self.foundBlas   = 0
    self.foundLapack = 0
    for (name, blasLibrary, lapackLibrary, self.useCompatibilityLibs) in self.generateGuesses():
      self.framework.log.write('================================================================================\n')
      self.framework.log.write('Checking for a functional BLAS and LAPACK in '+name+'\n')
      (foundBlas, foundLapack) = self.executeTest(self.checkLib, [lapackLibrary, blasLibrary])
      if foundBlas:   self.foundBlas   = 1
      if foundLapack: self.foundLapack = 1
      if foundBlas and foundLapack:
        self.functionalBlasLapack.append((name, blasLibrary, lapackLibrary))
        if not self.framework.argDB['with-alternatives']:
          break
    # User chooses one or take first (sort by version)
    if self.foundBlas and self.foundLapack:
      name, self.blasLibrary, self.lapackLibrary = self.functionalBlasLapack[0]
      if not isinstance(self.blasLibrary,   list): self.blasLibrary   = [self.blasLibrary]
      if not isinstance(self.lapackLibrary, list): self.lapackLibrary = [self.lapackLibrary]
      self.lib = []
      if self.lapackLibrary[0]: self.lib.extend(self.lapackLibrary)
      if self.blasLibrary[0]:   self.lib.extend(self.blasLibrary)
      self.dlib = self.lib[:]
      if self.useCompatibilityLibs:
        self.dlib.extend(self.compilers.flibs)
      self.framework.packages.append(self)
      if self.f2c:
        self.addDefine('BLASLAPACK_UNDERSCORE', 1)
    else:
      if not self.foundBlas:
        raise RuntimeError('Could not find a functional BLAS. Run with --with-blas-lib=<lib> to indicate the library containing BLAS.\n Or --download-c-blas-lapack=1 or --download-f-blas-lapack=1 to have one automatically downloaded and installed\n')
      if not self.foundLapack:
        raise RuntimeError('Could not find a functional LAPACK. Run with --with-lapack-lib=<lib> to indicate the library containing LAPACK.\n Or --download-c-blas-lapack=1 or --download-f-blas-lapack=1 to have one automatically downloaded and installed\n')
    self.found = 1
    return

  def checkESSL(self):
    '''Check for the IBM ESSL library'''
    if self.libraries.check(self.lapackLibrary, 'iessl'):
      self.addDefine('HAVE_ESSL',1)
    return

  def checkMissing(self):
    '''Check for missing LAPACK routines'''
    if self.foundLapack:
      mangleFunc = hasattr(self.compilers, 'FC') and not self.f2c
      for baseName in ['gesvd','geev','getrf','potrf','getrs','potrs']:
        if self.f2c:
          routine = 'd'+baseName+'_'
        else:
          routine = 'd'+baseName
        oldLibs = self.compilers.LIBS
        if not self.libraries.check(self.lapackLibrary, routine, otherLibs = self.getOtherLibs(), fortranMangle = mangleFunc):
          self.missingRoutines.append(baseName)
          self.addDefine('MISSING_LAPACK_'+baseName.upper(), 1)
        self.compilers.LIBS = oldLibs
    return

  def checkForRoutine(self,routine):
    ''' used by other packages to see if a BLAS routine is available
        This is not really correct because other packages do not (usually) know about f2cblasLapack'''
    if self.f2c:
      return self.libraries.check(self.dlib,routine+'_')
    else:
      return self.libraries.check(self.dlib,routine,fortranMangle = hasattr(self.compilers, 'FC'))

  def configure(self):
    self.executeTest(self.configureLibrary)
    self.executeTest(self.checkESSL)
    self.executeTest(self.checkMissing)
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging()
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
