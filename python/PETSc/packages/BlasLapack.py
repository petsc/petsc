from __future__ import generators
import user
import config.base

import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.argDB        = framework.argDB
    self.found        = 0
    # Assume that these libraries are Fortran if we have a Fortran compiler
    self.compilers    = self.framework.require('config.compilers',     self)
    self.libraries    = self.framework.require('config.libraries',     self)
    return

  def __str__(self):
    dirs    = []
    libFlag = []
    for lib in self.lapackLibrary+self.blasLibrary:
      if lib is None: continue
      dir = os.path.dirname(lib)
      if not dir in dirs:
        dirs.append(dir)
      else:
        lib = os.path.basename(lib)
      libFlag.append(self.libraries.getLibArgument(lib))
    return 'BLAS/LAPACK: '+' '.join(libFlag)+'\n'

  def configureHelp(self, help):
    import nargs
    help.addArgument('BLAS/LAPACK', '-with-blas-lapack-dir=<lib>',   nargs.ArgDir(None, None, 'Indicate the directory containing BLAS and LAPACK libraries'))
    help.addArgument('BLAS/LAPACK', '-with-blas-lapack=<lib>',       nargs.Arg(None, None, 'Indicate the library containing BLAS and LAPACK'))
    help.addArgument('BLAS/LAPACK', '-with-blas=<lib>',              nargs.Arg(None, None, 'Indicate the library containing BLAS'))
    help.addArgument('BLAS/LAPACK', '-with-lapack=<lib>',            nargs.Arg(None, None, 'Indicate the library containing LAPACK'))
    help.addArgument('BLAS/LAPACK', '-with-c-blas-lapack',           nargs.ArgBool(None, 0, 'Automatically install a C version of BLAS/LAPACK'))
    help.addArgument('BLAS/LAPACK', '-with-f-blas-lapack',           nargs.ArgBool(None, 0, 'Automatically install a Fortran version of BLAS/LAPACK'))
    help.addArgument('BLAS/LAPACK', '-with-c-blas-lapack-if-needed', nargs.ArgBool(None, 0, 'Automatically install a C version of BLAS/LAPACK if no BLAS/LAPACK found'))
    help.addArgument('BLAS/LAPACK', '-with-f-blas-lapack-if-needed', nargs.ArgBool(None, 0, 'Automatically install a Fortran version of BLAS/LAPACK if no BLAS/LAPACK found'))
    return

  def parseLibrary(self, library):
    (dir, lib)  = os.path.split(library)
    lib         = os.path.splitext(lib)[0]
    if lib.startswith('lib'): lib = lib[3:]
    return (dir, lib)

  def checkLib(self, lapackLibrary, blasLibrary = None):
    '''Checking for BLAS and LAPACK symbols'''
    f2c = 0
    if blasLibrary is None:
      separateBlas = 0
      blasLibrary  = lapackLibrary
    else:
      separateBlas = 1
    if not isinstance(lapackLibrary, list): lapackLibrary = [lapackLibrary]
    if not isinstance(blasLibrary,   list): blasLibrary   = [blasLibrary]
    foundBlas   = 0
    foundLapack = 0
    mangleFunc  = 'FC' in self.framework.argDB
    if mangleFunc:
      otherLibs = self.compilers.flibs
    else:
      otherLibs = ''
    # Check for BLAS
    oldLibs   = self.framework.argDB['LIBS']
    foundBlas = self.libraries.check(blasLibrary, 'ddot', otherLibs = otherLibs, fortranMangle = mangleFunc)
    if not foundBlas:
      foundBlas = self.libraries.check(blasLibrary, 'ddot_', otherLibs = otherLibs, fortranMangle = 0)
    self.framework.argDB['LIBS'] = oldLibs
    # Check for LAPACK
    if foundBlas and separateBlas:
      otherLibs = ' '.join(map(self.libraries.getLibArgument, blasLibrary))+' '+otherLibs
    oldLibs     = self.framework.argDB['LIBS']
    foundLapack = self.libraries.check(lapackLibrary, 'dgetrs', otherLibs = otherLibs, fortranMangle = mangleFunc) or self.libraries.check(lapackLibrary, 'dgeev', otherLibs = otherLibs, fortranMangle = mangleFunc)
    if not foundLapack:
      foundLapack = self.libraries.check(lapackLibrary, 'dgetrs_', otherLibs = otherLibs, fortranMangle = 0) or self.libraries.check(lapackLibrary, 'dgeev_', otherLibs = otherLibs, fortranMangle = 0)
      if foundLapack:
        self.addDefine('BLASLAPACK_F2C',1)
        mangleFunc = 0
        f2c        = 1
    if foundLapack:
      #check for missing symbols from lapack
      for i in ['gesvd','geev','getrf','potrf','getrs','potrs']:
        if f2c: ii = 'd'+i+'_'
        else:   ii = 'd'+i
        if not self.libraries.check(lapackLibrary, ii, otherLibs = otherLibs, fortranMangle = mangleFunc):
           self.addDefine('MISSING_LAPACK_'+i.upper(),1)
      
    self.framework.argDB['LIBS'] = oldLibs
    return (foundBlas, foundLapack)

  def generateGuesses(self):
    # Try specified BLASLAPACK library
    if 'with-blas-lapack' in self.framework.argDB:
      yield ('User specified BLAS/LAPACK library', None, self.framework.argDB['with-blas-lapack'])
      raise RuntimeError('You set a value for --with-blas-lapack, but '+str(self.framework.argDB['with-blas-lapack'])+' cannot be used\n')
    # Try specified BLAS and LAPACK libraries
    if 'with-blas' in self.framework.argDB and 'with-lapack' in self.framework.argDB:
      yield ('User specified BLAS and LAPACK libraries', self.framework.argDB['with-blas'], self.framework.argDB['with-lapack'])
      raise RuntimeError('You set a value for --with-blas and --with-lapack, but '+str(self.framework.argDB['with-blas'])+' and '+str(self.framework.argDB['with-lapack'])+' cannot be used\n')
    # Try specified installation root
    if 'with-blas-lapack-dir' in self.framework.argDB:
      dir = self.framework.argDB['with-blas-lapack-dir']
      yield ('User specified installation root', os.path.join(dir, 'libblas.a'),    os.path.join(dir, 'liblapack.a'))
      yield ('User specified installation root', os.path.join(dir, 'libf2cblas.a'), os.path.join(dir, 'libf2clapack.a'))
      yield ('User specified installation root', os.path.join(dir, 'libfblas.a'),   os.path.join(dir, 'libflapack.a'))
      if self.framework.argDB['with-64-bit']:
        dir = os.path.join(dir, 'lib', '64')
      else:
        dir = os.path.join(dir, 'lib', '32')
      yield ('User specified MKL Linux installation root', None, [os.path.join(dir, 'libmkl_lapack.a'), os.path.join(dir, 'libmkl_def.a'), 'guide', 'pthread'])
      dir = self.framework.argDB['with-blas-lapack-dir']
      if self.framework.argDB['with-64-bit']:
        dir = os.path.join(dir, 'ia64', 'lib')
      else:
        dir = os.path.join(dir, 'ia32', 'lib')
      yield ('User specified MKL Windows installation root', None, [os.path.join(dir, 'mkl_c_dll.lib')])
      raise RuntimeError('You set a value for --with-blas-lapack-dir, but '+self.framework.argDB['with-blas-lapack-dir']+' cannot be used\n')
    # IRIX locations
    yield ('IRIX Mathematics library', None, 'libcomplib.sgimath.a')
    # IBM ESSL locations
    yield ('IBM ESSL Mathematics library', None, 'libessl.a')
    # Portland group compiler blas and lapack
    if 'PGI' in os.environ:
      dir = os.path.join(os.environ['PGI'],'linux86','lib')
      yield ('User specified installation root', os.path.join(dir, 'libblas.a'), os.path.join(dir, 'liblapack.a'))
    # Try compiler defaults
    yield ('Default compiler locations', 'libblas.a', 'liblapack.a')
    # /usr/local/lib
    dir = os.path.join('/usr','local','lib')
    yield ('Default compiler locations /usr/local/lib', os.path.join(dir,'libblas.a'), os.path.join(dir,'liblapack.a'))    
    yield ('Default compiler locations with G77', None, ['liblapack.a', 'libblas.a','libg2c.a'])
    # Try MacOSX location
    yield ('MacOSX BLAS/LAPACK library', None, os.path.join('/System', 'Library', 'Frameworks', 'vecLib.framework', 'vecLib'))
    # Sun locations
    yield ('Sun BLAS/LAPACK library', None, ['libsunperf.a','libsunmath.a'])
    yield ('Sun BLAS/LAPACK library', None, ['libsunperf.a','libF77.a','libM77.a','libsunmath.a'])
    # Try Microsoft Windows location
    MKL_Dir = os.path.join('/cygdrive', 'c', 'Program\\ Files', 'Intel', 'MKL')
    if self.framework.argDB['with-64-bit']:
      MKL_Dir = os.path.join(MKL_Dir, 'ia64', 'lib')
    else:
      MKL_Dir = os.path.join(MKL_Dir, 'ia32', 'lib')
    yield ('Microsoft Windows, Intel MKL library', None, os.path.join(MKL_Dir,'mkl_c_dll.lib'))
    # Try PETSc location
    PETSC_DIR  = None
    PETSC_ARCH = None
    if 'PETSC_DIR' in self.framework.argDB and 'PETSC_ARCH' in self.framework.argDB:
      PETSC_DIR  = self.framework.argDB['PETSC_DIR']
      PETSC_ARCH = self.framework.argDB['PETSC_ARCH']
    elif os.getenv('PETSC_DIR') and os.getenv('PETSC_ARCH'):
      PETSC_DIR  = os.getenv('PETSC_DIR')
      PETSC_ARCH = os.getenv('PETSC_ARCH')

    if PETSC_ARCH and PETSC_DIR:
      dir1 = os.path.abspath(os.path.join(PETSC_DIR, '..', 'blaslapack', 'lib'))
      yield ('PETSc location 1', os.path.join(dir1, 'libblas.a'), os.path.join(dir1, 'liblapack.a'))
      dir2 = os.path.join(dir1, 'libg_c++', PETSC_ARCH)
      yield ('PETSc location 2', os.path.join(dir2, 'libblas.a'), os.path.join(dir2, 'liblapack.a'))
      dir3 = os.path.join(dir1, 'libO_c++', PETSC_ARCH)
      yield ('PETSc location 3', os.path.join(dir3, 'libblas.a'), os.path.join(dir3, 'liblapack.a'))
    return

  def downLoadBlasLapack(self,f2c,l):
    self.framework.log.write('Downloading '+l+'blaslapack')

    if f2c == 'f2c': self.addDefine('BLASLAPACK_F2C',1)
    self.foundBlas       = 1
    self.foundLapack     = 1
    libdir               = os.path.join(self.framework.argDB['PETSC_DIR'],f2c+'blaslapack',self.framework.argDB['PETSC_ARCH'])
    self.functionalBlasLapack.append((f2c+'blaslapack', os.path.join(libdir,'lib'+f2c+'blas.a'), os.path.join(libdir,'lib'+f2c+'lapack.a')))
    if not os.path.isdir(os.path.join(self.framework.argDB['PETSC_DIR'],f2c+'blaslapack')):
      import urllib
      try:
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/'+f2c+'blaslapack.tar.gz',f2c+'blaslapack.tar.gz')
      except:
        raise RuntimeError('Error downloading '+f2c+'blaslapack.tar.gz requested with -with-'+l+'-blas-lapack option')
      try:
        config.base.Configure.executeShellCommand('gunzip '+f2c+'blaslapack.tar.gz', log = self.framework.log)
      except:
        raise RuntimeError('Error unzipping '+f2c+'blaslapack.tar.gz requested with -with-'+l+'-blas-lapack option')
      try:
        config.base.Configure.executeShellCommand('tar -xf '+f2c+'blaslapack.tar', log = self.framework.log)
      except:
        raise RuntimeError('Error doing tar -xf '+f2c+'blaslapack.tar requested with -with-'+l+'-blas-lapack option')
      os.unlink(f2c+'blaslapack.tar')
    if not os.path.isdir(libdir):
      os.mkdir(libdir)
    
  def configureLibrary(self):
    self.functionalBlasLapack = []
    self.foundBlas       = 0
    self.foundLapack     = 0
    if self.framework.argDB['with-c-blas-lapack'] or self.framework.argDB['with-f-blas-lapack']:
      if self.framework.argDB['with-c-blas-lapack']:
        f2c = 'f2c'
        l   = 'c'
      else:
        if not hasattr(self.framework.argDB,'FC'):
          raise RuntimeError('Cannot request f-blas-lapack without Fortran compiler, maybe you want --with-c-blas-lapack?')
        f2c = 'f'
        l   = 'f'
      self.downLoadBlasLapack(f2c,l)        
    else:
      for (name, blasLibrary, lapackLibrary) in self.generateGuesses():
        self.framework.log.write('================================================================================\n')
        self.framework.log.write('Checking for a functional BLAS and LAPACK in '+name+'\n')
        (foundBlas, foundLapack) = self.executeTest(self.checkLib, [lapackLibrary, blasLibrary])
        if foundBlas:   self.foundBlas   = 1
        if foundLapack: self.foundLapack = 1
        if foundBlas and foundLapack:
          self.functionalBlasLapack.append((name, blasLibrary, lapackLibrary))
          if not self.framework.argDB['with-alternatives']:
            break

    if not (self.foundBlas and self.foundLapack):
      if self.framework.argDB['with-c-blas-lapack-if-needed'] or self.framework.argDB['with-f-blas-lapack-if-needed']:
        if self.framework.argDB['with-c-blas-lapack-if-needed']:
          f2c = 'f2c'
          l   = 'c'
        else:
          if not hasattr(self.framework.argDB,'FC'):
            raise RuntimeError('Cannot request f-blas-lapack without Fortran compiler, maybe you want --with-c-blas-lapack?')
          f2c = 'f'
          l   = 'f'
        self.downLoadBlasLapack(f2c,l)        
        
    # User chooses one or take first (sort by version)
    if self.foundBlas and self.foundLapack:
      name, self.blasLibrary, self.lapackLibrary = self.functionalBlasLapack[0]
      if not isinstance(self.blasLibrary,   list): self.blasLibrary   = [self.blasLibrary]
      if not isinstance(self.lapackLibrary, list): self.lapackLibrary = [self.lapackLibrary]
      
      #ugly stuff to decide if BLAS/LAPACK are dynamic or static
      self.framework.sharedBlasLapack = 1
      if len(self.blasLibrary) > 0 and self.blasLibrary[0]:
        if ' '.join(self.blasLibrary).find('blas.a') >= 0: self.framework.sharedBlasLapack = 0
      if len(self.lapackLibrary) > 0 and self.lapackLibrary[0]:
        if ' '.join(self.lapackLibrary).find('lapack.a') >= 0: self.framework.sharedBlasLapack = 0

    else:
      if not self.foundBlas:
        raise RuntimeError('Could not find a functional BLAS. Run with --with-blas=<lib> to indicate location of BLAS.\n Or --with-c-blas-lapack or --with-f-blas-lapack to have one automatically downloaded and installed\n')
      if not self.foundLapack:
        raise RuntimeError('Could not find a functional LAPACK. Run with --with-lapack=<lib> to indicate location of LAPACK.\n Or --with-c-blas-lapack or --with-f-blas-lapack to have one automatically downloaded and installed\n')
    return

  def configureESSL(self):
    if self.libraries.check(self.lapackLibrary, 'iessl'):
      self.addDefine('HAVE_ESSL',1)
    return

  def unique(self, l):
    m = []
    for i in l:
      if not i in m: m.append(i)
    return m

  def setOutput(self):
    '''Add defines and substitutions
       - BLAS_DIR is the location of the BLAS library
       - LAPACK_DIR is the location of the LAPACK library
       - LAPACK_LIB is the LAPACK linker flags'''
    if self.foundBlas:
      if None in self.blasLibrary:
        lib = self.lapackLibrary
      else:
        lib = self.blasLibrary
      dir = self.unique(map(os.path.dirname, lib))
      self.addSubstitution('BLAS_DIR', dir)
      libFlag = map(self.libraries.getLibArgument, lib)
      self.addSubstitution('BLAS_LIB', ' '.join(libFlag))
    if self.foundLapack:
      dir = self.unique(map(os.path.dirname, self.lapackLibrary))
      self.addSubstitution('LAPACK_DIR', dir)
      libFlag = map(self.libraries.getLibArgument, self.lapackLibrary)
      self.addSubstitution('LAPACK_LIB', ' '.join(libFlag))
    if self.foundBlas and self.foundLapack:
      dirs    = []
      libFlag = []
      for lib in self.lapackLibrary+self.blasLibrary:
        if lib is None: continue
        dir = os.path.dirname(lib)
        if not dir in dirs:
          dirs.append(dir)
        else:
          lib = os.path.basename(lib)
        libFlag.append(self.libraries.getLibArgument(lib))
      self.addSubstitution('BLASLAPACK_DIR', dirs)
      self.addSubstitution('BLASLAPACK_LIB', ' '.join(libFlag))
    return

  def configure(self):
    self.executeTest(self.configureLibrary)
    self.executeTest(self.configureESSL)
    self.setOutput()
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging()
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
