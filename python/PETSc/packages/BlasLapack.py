from __future__ import generators
import user
import config.base
import md5
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.argDB        = framework.argDB
    self.found        = 0
    # Assume that these libraries are Fortran if we have a Fortran compiler
    self.compilers    = self.framework.require('config.compilers',            self)
    self.setcompilers = self.framework.require('config.setCompilers',            self)    
    self.libraries    = self.framework.require('config.libraries',            self)
    self.framework.require('PETSc.packages.Sowing', self)
    
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

  def setupHelp(self, help):
    import nargs
    help.addArgument('BLAS/LAPACK', '-with-blas-lapack-dir=<dir>',                nargs.ArgDir(None, None, 'Indicate the directory containing BLAS and LAPACK libraries'))
    help.addArgument('BLAS/LAPACK', '-with-blas-lapack-lib=<lib>',                nargs.Arg(None, None, 'Indicate the library containing BLAS and LAPACK'))
    help.addArgument('BLAS/LAPACK', '-with-blas-lib=<lib>',                       nargs.Arg(None, None, 'Indicate the library(s) containing BLAS'))
    help.addArgument('BLAS/LAPACK', '-with-lapack-lib=<lib>',                     nargs.Arg(None, None, 'Indicate the library(s) containing LAPACK'))
    help.addArgument('BLAS/LAPACK', '-download-c-blas-lapack=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Automatically install a C version of BLAS/LAPACK'))
    help.addArgument('BLAS/LAPACK', '-download-f-blas-lapack=<no,yes,ifneeded>',  nargs.ArgFuzzyBool(None, 0, 'Automatically install a Fortran version of BLAS/LAPACK'))
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
    foundBlas = (self.libraries.check(blasLibrary, 'ddot', otherLibs = otherLibs, fortranMangle = mangleFunc))
    if not foundBlas:
      foundBlas = (self.libraries.check(blasLibrary, 'ddot_', otherLibs = otherLibs, fortranMangle = 0))
    self.framework.argDB['LIBS'] = oldLibs
    # Check for LAPACK
    if foundBlas and separateBlas:
      otherLibs = ' '.join(map(self.libraries.getLibArgument, blasLibrary))+' '+otherLibs
    oldLibs     = self.framework.argDB['LIBS']
    foundLapack = ((self.libraries.check(lapackLibrary, 'dgetrs', otherLibs = otherLibs, fortranMangle = mangleFunc) or
                    self.libraries.check(lapackLibrary, 'dgeev', otherLibs = otherLibs, fortranMangle = mangleFunc)))
    if not foundLapack:
      foundLapack = ((self.libraries.check(lapackLibrary, 'dgetrs_', otherLibs = otherLibs, fortranMangle = 0) or
                      self.libraries.check(lapackLibrary, 'dgeev_', otherLibs = otherLibs, fortranMangle = 0)))
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
    # check that user has used the options properly
    if 'with-blas-lib' in self.framework.argDB and not 'with-lapack-lib' in self.framework.argDB:
      raise RuntimeError('If you use the --with-blas-lib=<lib> you must also use --with-lapack-lib=<lib> option')
    if not 'with-blas-lib' in self.framework.argDB and 'with-lapack-lib' in self.framework.argDB:
      raise RuntimeError('If you use the --with-lapack-lib=<lib> you must also use --with-blas-lib=<lib> option')
    if 'with-blas-lib' in self.framework.argDB and 'with-blas-lapack-dir' in self.framework.argDB:
      raise RuntimeError('You cannot set both the library containing BLAS with --with-blas-lib=<lib>\nand the directory to search with --with-blas-lapack-dir=<dir>')
    if 'with-blas-lapack-lib' in self.framework.argDB and 'with-blas-lapack-dir' in self.framework.argDB:
      raise RuntimeError('You cannot set both the library containing BLAS/LAPACK with --with-blas-lapack-lib=<lib>\nand the directory to search with --with-blas-lapack-dir=<dir>')

    if self.framework.argDB['download-f-blas-lapack'] == 1:
      if not 'FC' in self.framework.argDB:
        raise RuntimeError('Cannot request f-blas-lapack without Fortran compiler, maybe you want --download-c-blas-lapack=1?')
      libdir = self.downLoadBlasLapack('f','f')            
      yield ('Downloaded BLAS/LAPACK library', os.path.join(libdir,'libfblas.a'), os.path.join(libdir,'libflapack.a'))
      raise RuntimeError('Could not use downloaded f-blas-lapack?')
    if self.framework.argDB['download-c-blas-lapack'] == 1:
      if 'FC' in self.framework.argDB:
        raise RuntimeError('Should request f-blas-lapack, not --download-c-blas-lapack=yes since you have a fortran compiler?')
      libdir = self.downLoadBlasLapack('f2c','c')            
      yield ('Downloaded BLAS/LAPACK library', os.path.join(libdir,'libf2cblas.a'), os.path.join(libdir,'libf2clapack.a'))
      raise RuntimeError('Could not use downloaded c-blas-lapack?')
    # Try specified BLASLAPACK library
    if 'with-blas-lapack-lib' in self.framework.argDB:
      yield ('User specified BLAS/LAPACK library', None, self.framework.argDB['with-blas-lapack-lib'])
      raise RuntimeError('You set a value for --with-blas-lapack-lib=<lib>, but '+str(self.framework.argDB['with-blas-lapack-lib'])+' cannot be used\n')
    # Try specified BLAS and LAPACK libraries
    if 'with-blas-lib' in self.framework.argDB and 'with-lapack-lib' in self.framework.argDB:
      yield ('User specified BLAS and LAPACK libraries', self.framework.argDB['with-blas-lib'], self.framework.argDB['with-lapack-lib'])
      raise RuntimeError('You set a value for --with-blas-lib=<lib> and --with-lapack-lib=<lib>, but '+str(self.framework.argDB['with-blas-lib'])+' and '+str(self.framework.argDB['with-lapack-lib'])+' cannot be used\n')
    # Try specified installation root
    if 'with-blas-lapack-dir' in self.framework.argDB:
      dir = self.framework.argDB['with-blas-lapack-dir']
      if not (len(dir) > 2 and dir[1] == ':') :
        dir = os.path.abspath(dir)
      yield ('User specified installation root (HPUX)', os.path.join(dir, 'libveclib.a'),  os.path.join(dir, 'liblapack.a'))      
      yield ('User specified installation root (F2C)', os.path.join(dir, 'libf2cblas.a'), os.path.join(dir, 'libf2clapack.a'))
      yield ('User specified installation root', os.path.join(dir, 'libfblas.a'),   os.path.join(dir, 'libflapack.a'))
      yield ('User specified ATLAS Linux installation root', [os.path.join(dir, 'libcblas.a'),os.path.join(dir, 'libf77blas.a'), os.path.join(dir, 'libatlas.a')],  [os.path.join(dir, 'liblapack.a')])
      yield ('User specified ATLAS Linux installation root', [os.path.join(dir, 'libf77blas.a'), os.path.join(dir, 'libatlas.a')],  [os.path.join(dir, 'liblapack.a')])      
      yield ('User specified MKL Linux lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), os.path.join(dir, 'libmkl_def.a'), 'guide', 'pthread'])
      yield ('User specified MKL Linux lib dir', None, [os.path.join(dir, 'libmkl_lapack.a'), os.path.join(dir, 'libmkl_ipf.a'), 'guide', 'pthread'])      
      mkldir = dir
      if self.framework.argDB['with-64-bit-pointers']:
        mkldir = os.path.join(mkldir, 'lib', '64')
      else:
        mkldir = os.path.join(mkldir, 'lib', '32')
      yield ('User specified MKL Linux installation root', None, [os.path.join(mkldir, 'libmkl_lapack.a'), os.path.join(mkldir, 'libmkl_def.a'), 'guide', 'pthread'])
      yield ('User specified MKL Linux installation root', None, [os.path.join(mkldir, 'libmkl_lapack.a'), os.path.join(mkldir, 'libmkl_ipf.a'), 'guide', 'pthread'])
      if self.framework.argDB['with-64-bit-pointers']:
        mkldir = os.path.join(dir, 'ia64', 'lib')
      else:
        mkldir = os.path.join(dir, 'ia32', 'lib')
      yield ('User specified MKL Windows installation root', None, [os.path.join(mkldir, 'mkl_c_dll.lib')])
      yield ('User specified MKL Windows lib dir', None, [os.path.join(dir, 'mkl_c_dll.lib')])
      # Search for liblapack.a and libblas.a after the implementations with more specific name to avoid
      # finding these in /usr/lib despite using -L<blas-lapack-dir> while attempting to get a different library.
      yield ('User specified installation root', os.path.join(dir, 'libblas.a'),    os.path.join(dir, 'liblapack.a'))
      raise RuntimeError('You set a value for --with-blas-lapack-dir=<dir>, but '+self.framework.argDB['with-blas-lapack-dir']+' cannot be used\n')
    # IRIX locations
    yield ('IRIX Mathematics library', None, 'libcomplib.sgimath.a')
    yield ('Another IRIX Mathematics library', None, 'libscs.a')    
    # IBM ESSL locations
    yield ('IBM ESSL Mathematics library', None, 'libessl.a')
    # Portland group compiler blas and lapack
    if 'PGI' in os.environ:
      dir = os.path.join(os.environ['PGI'],'linux86','lib')
      yield ('User specified installation root', os.path.join(dir, 'libblas.a'), os.path.join(dir, 'liblapack.a'))
    # Try compiler defaults
    yield ('Default compiler locations', 'libblas.a', 'liblapack.a')
    yield ('HPUX', 'libveclib.a', 'liblapack.a')
    # /usr/local/lib
    dir = os.path.join('/usr','local','lib')
    yield ('Default compiler locations /usr/local/lib', os.path.join(dir,'libblas.a'), os.path.join(dir,'liblapack.a'))    
    yield ('Default compiler locations with G77', None, ['liblapack.a', 'libblas.a','libg2c.a'])
    # Try MacOSX location
    yield ('MacOSX BLAS/LAPACK library', None, os.path.join('/System', 'Library', 'Frameworks', 'vecLib.framework', 'vecLib'))
    # Sun locations
    yield ('Sun sunperf BLAS/LAPACK library', None, ['libsunperf.a','libsunmath.a','libm.a'])
    yield ('Sun sunperf BLAS/LAPACK library', None, ['libsunperf.a','libF77.a','libM77.a','libsunmath.a','libm.a'])
    yield ('Sun sunperf BLAS/LAPACK library', None, ['libsunperf.a','libfui.a','libfsu.a','libsunmath.a','libm.a'])    
    # Try Microsoft Windows location
    for MKL_Version in ['MKL70','MKL61','MKL']:
      MKL_Dir = os.path.join('/cygdrive', 'c', 'Program\\ Files', 'Intel', MKL_Version)
      if self.framework.argDB['with-64-bit-pointers']:
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
    if self.framework.argDB['download-f-blas-lapack'] == 2:
      if not 'FC' in self.framework.argDB:
        raise RuntimeError('Cannot request f-blas-lapack without Fortran compiler, maybe you want --download-c-blas-lapack=1?')
      libdir = self.downLoadBlasLapack('f','f')            
      yield ('Downloaded BLAS/LAPACK library', os.path.join(libdir,'libfblas.a'), os.path.join(libdir,'libflapack.a'))
    if self.framework.argDB['download-c-blas-lapack'] == 2:
      if 'FC' in self.framework.argDB:
        raise RuntimeError('Should request f-blas-lapack, not --download-c-blas-lapack=ifneeded since you have a fortran compiler?')
      libdir = self.downLoadBlasLapack('f2c','c')            
      yield ('Downloaded BLAS/LAPACK library', os.path.join(libdir,'libf2cblas.a'), os.path.join(libdir,'libf2clapack.a'))
    return

  def downLoadBlasLapack(self,f2c,l):
    self.framework.log.write('Downloading '+l+'blaslapack\n')

    packages = os.path.join(self.framework.argDB['PETSC_DIR'],'packages')
    if not os.path.isdir(packages):
      os.mkdir(packages)

    if f2c == 'f2c': self.addDefine('BLASLAPACK_F2C',1)
    libdir               = os.path.join(packages,f2c+'blaslapack',self.framework.argDB['PETSC_ARCH'])
    if not os.path.isdir(os.path.join(packages,f2c+'blaslapack')):
      self.framework.log.write('Actually need to ftp '+l+'blaslapack\n')
      import urllib
      try:
        urllib.urlretrieve('ftp://ftp.mcs.anl.gov/pub/petsc/'+f2c+'blaslapack.tar.gz',os.path.join('packages',f2c+'blaslapack.tar.gz'))
      except:
        raise RuntimeError('Error downloading '+f2c+'blaslapack.tar.gz requested with -with-'+l+'-blas-lapack option')
      try:
        config.base.Configure.executeShellCommand('cd packages; gunzip '+f2c+'blaslapack.tar.gz', log = self.framework.log)
      except:
        raise RuntimeError('Error unzipping '+f2c+'blaslapack.tar.gz requested with -with-'+l+'-blas-lapack option')
      try:
        config.base.Configure.executeShellCommand('cd packages; tar -xf '+f2c+'blaslapack.tar', log = self.framework.log)
      except:
        raise RuntimeError('Error doing tar -xf '+f2c+'blaslapack.tar requested with -with-'+l+'-blas-lapack option')
      os.unlink(os.path.join('packages',f2c+'blaslapack.tar'))
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
        cc = self.framework.argDB['CC']
        line = 'CC = '+cc+'\n'
      if line.startswith('COPTFLAGS '):
        self.setcompilers.pushLanguage('C')
        #line = 'COPTFLAGS  = '+self.setcompilers.getCompilerFlags()+'\n'
        self.setcompilers.popLanguage()
        line = 'COPTFLAGS = -O '+self.framework.argDB['CFLAGS']+'\n'
      if line.startswith('FC  '):
        fc = self.framework.argDB['FC']
        if fc.find('f90') >= 0:
          import commands
          output  = commands.getoutput(fc+' -v')
          if output.find('IBM') >= 0:
            fc = os.path.join(os.path.dirname(fc),'xlf')
            self.framework.log.write('Using IBM f90 compiler for PETSc, switching to xlf for compiling BLAS/LAPACK\n')
        line = 'FC = '+fc+'\n'
      if line.startswith('FOPTFLAGS '):
        self.setcompilers.pushLanguage('F77')
        #line = 'FOPTFLAGS  = '+self.setcompilers.getCompilerFlags()+'\n'
        self.setcompilers.popLanguage()
        line = 'FOPTFLAGS  = -O '+self.framework.argDB['FFLAGS']+'\n'
      if line.startswith('AR '):
        line = 'AR         = '+self.setcompilers.AR+'\n'
      if line.startswith('AR_FLAGS '):
        line = 'AR_FLAGS   = '+self.setcompilers.AR_FLAGS+'\n'

      if line.startswith('LIB_SUFFIX '):
        line = 'LIB_SUFFIX = '+self.framework.argDB['LIB_SUFFIX']+'\n'
      if line.startswith('RANLIB '):
        line = 'RANLIB     = '+self.framework.argDB['RANLIB']+'\n'
      if line.startswith('RM '):
        line = 'RM         = rm \n'

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
      output  = config.base.Configure.executeShellCommand('cd '+blasDir+';make -f tmpmakefile', timeout=800, log = self.framework.log)[0]
    except RuntimeError, e:
      raise RuntimeError('Error running make on '+l+'blaslapack: '+str(e))
    try:
      output  = config.base.Configure.executeShellCommand('cd '+blasDir+';mv -f lib'+f2c+'blas.'+self.framework.argDB['LIB_SUFFIX']+' lib'+f2c+'lapack.'+self.framework.argDB['LIB_SUFFIX']+' '+self.framework.argDB['PETSC_ARCH'], timeout=30, log = self.framework.log)[0]
    except RuntimeError, e:
      raise RuntimeError('Error moving '+l+'blaslapack libraries: '+str(e))
    try:
      output  = config.base.Configure.executeShellCommand('cd '+blasDir+';cp -f tmpmakefile '+self.framework.argDB['PETSC_ARCH'], timeout=30, log = self.framework.log)[0]
    except RuntimeError, e:
      pass
    return libdir
  
  def configureLibrary(self):
    self.functionalBlasLapack = []
    self.foundBlas       = 0
    self.foundLapack     = 0
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

    # User chooses one or take first (sort by version)
    if self.foundBlas and self.foundLapack:
      name, self.blasLibrary, self.lapackLibrary = self.functionalBlasLapack[0]
      if not isinstance(self.blasLibrary,   list): self.blasLibrary   = [self.blasLibrary]
      if not isinstance(self.lapackLibrary, list): self.lapackLibrary = [self.lapackLibrary]
      
      #ugly stuff to decide if BLAS/LAPACK are dynamic or static
      self.sharedBlasLapack = 1
      if len(self.blasLibrary) > 0 and self.blasLibrary[0]:
        if ' '.join(self.blasLibrary).find('blas.a') >= 0: self.sharedBlasLapack = 0
        if len(self.lapackLibrary) > 0 and self.lapackLibrary[0]:
          if ' '.join(self.lapackLibrary).find('lapack.a') >= 0: self.sharedBlasLapack = 0

    else:
      if not self.foundBlas:
        raise RuntimeError('Could not find a functional BLAS. Run with --with-blas-lib=<lib> to indicate the library containing BLAS.\n Or --download-c-blas-lapack=1 or --download-f-blas-lapack=1 to have one automatically downloaded and installed\n')
      if not self.foundLapack:
        raise RuntimeError('Could not find a functional LAPACK. Run with --with-lapack-lib=<lib> to indicate the library containing LAPACK.\n Or --download-c-blas-lapack=1 or --download-f-blas-lapack=1 to have one automatically downloaded and installed\n')

    # check if Mac OS BLAS/LAPACK and IBM Fortran compiler?
    # if yes, then force IBM Fortran to add _ for subroutine names
    # so cab access BLAS/LAPACK from Fortran
    if name.find('MacOSX') >= 0 and 'FC' in self.framework.argDB:
      self.setcompilers.pushLanguage('F77')
      if self.setcompilers.getCompiler().find('xlf') >= 0 or self.setcompilers.getCompiler().find('xlF') >= 0:
        # should check if compiler is already using underscore and that -qextname works 
        self.compilers.fortranMangling = 'underscore'
        self.framework.argDB['FFLAGS'] = self.framework.argDB['FFLAGS'] + ' -qextname'
        self.framework.log.write('Using the MacOX blas/lapack libraries and xlF so forcing _ after Fortran symbols\n')
        self.compilers.delDefine('HAVE_FORTRAN_NOUNDERSCORE')
        self.compilers.addDefine('HAVE_FORTRAN_UNDERSCORE',1)
        self.delDefine('BLASLAPACK_F2C')
      self.setcompilers.popLanguage()
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
    return

  def configurePIC(self):
    '''Determine the PIC option for each compiler
       - There needs to be a test that checks that the functionality is actually working'''
    if (self.framework.argDB['PETSC_ARCH_BASE'].startswith('osf')  or self.framework.argDB['PETSC_ARCH_BASE'].startswith('hpux') or self.framework.argDB['PETSC_ARCH_BASE'].startswith('aix')) and not config.setCompilers.Configure.isGNU(self.framework.argDB['CC']):
      return
    languages = ['C']
    if 'CXX' in self.framework.argDB:
      languages.append('C++')
    if 'FC' in self.framework.argDB:
      languages.append('F77')
    for language in languages:
      self.pushLanguage(language)
      for testFlag in ['-PIC', '-fPIC', '-KPIC']:
        try:
          self.framework.log.write('Trying '+language+' compiler flag '+testFlag+'\n')
          self.addCompilerFlag(testFlag)
          break
        except RuntimeError:
          self.framework.log.write('Rejected '+language+' compiler flag '+testFlag+'\n')
      self.popLanguage()
    return

  def configure(self):
    self.executeTest(self.configurePIC)
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
