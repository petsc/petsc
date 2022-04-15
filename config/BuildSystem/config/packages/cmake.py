import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version            = '3.23.1'
    self.download           = ['https://github.com/Kitware/CMake/releases/download/v'+self.version+'/cmake-'+self.version+'.tar.gz',
                              'https://gitlab.kitware.com/cmake/cmake/-/archive/v'+self.version+'/cmake-v'+self.version+'.tar.gz']
    self.download_solaris   = ['https://cmake.org/files/v3.11/cmake-3.11.4.tar.gz',
                               'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/cmake-3.11.4.tar.gz']
    self.downloadonWindows  = 1
    self.lookforbydefault   = 1
    self.publicInstall      = 0  # always install in PETSC_DIR/PETSC_ARCH (not --prefix) since this is not used by users
    self.linkedbypetsc      = 0
    self.executablename     = 'cmake'
    self.useddirectly       = 0
    self.maxminCmakeVersion = (2,0,0) # minimum CMake version needed by all active packages
    return

  def setupHelp(self, help):
    import nargs
    config.package.GNUPackage.setupHelp(self, help)
    help.addArgument('CMAKE', '-download-cmake-cc=<prog>',                   nargs.Arg(None, None, 'C compiler for Cmake configure'))
    help.addArgument('CMAKE', '-download-cmake-cxx=<prog>',                  nargs.Arg(None, None, 'C++ compiler for Cmake configure'))
    help.addArgument('CMAKE', '-with-cmake-exec=<executable>',                nargs.Arg(None, None, 'CMake executable to look for'))
    help.addArgument('CMAKE', '-with-ctest-exec=<executable>',                nargs.Arg(None, None, 'Ctest executable to look for'))
    return

  # Due to -DCMAKE_USE_OPENSSL=OFF usage below, currently -download-cmake-configure-arguments will only work with bootstrap cmake-options and not regular options
  def formGNUConfigureArgs(self):
    '''Does not use the standard arguments at all since this does not use the MPI compilers etc
       Cmake will chose its own compilers if they are not provided explicitly here'''
    args = ['--prefix='+self.installDir]
    args.append('--parallel='+str(self.make.make_np))
    if 'download-cmake-cc' in self.argDB and self.argDB['download-cmake-cc']:
      args.append('CC="'+self.argDB['download-cmake-cc']+'"')
    if 'download-cmake-cxx' in self.argDB and self.argDB['download-cmake-cxx']:
      args.append('CXX="'+self.argDB['download-cmake-cxx']+'"')
    if not config.setCompilers.Configure.isSolaris(self.log):
      args.append('-- -DCMAKE_USE_OPENSSL=OFF')
    return args

  def Install(self):
    save_path = os.environ['PATH']
    make_loc = os.path.dirname(self.make.make)
    self.log.write('CMAKE build - adding MAKE location to PATH: '+make_loc+'\n')
    os.environ['PATH'] += ':'+make_loc
    retdir = config.package.GNUPackage.Install(self)
    os.environ['PATH'] = save_path
    return retdir

  def locateCMake(self):
    if 'with-cmake-exec' in self.argDB:
      self.log.write('Looking for specified CMake executable '+self.argDB['with-cmake-exec']+'\n')
      self.getExecutable(self.argDB['with-cmake-exec'], getFullPath=1, resultName='cmake')
    else:
      self.log.write('Looking for default CMake executable\n')
      self.getExecutable('cmake', getFullPath=1, resultName='cmake')
    if 'with-ctest-exec' in self.argDB:
      self.log.write('Looking for specified Ctest executable '+self.argDB['with-ctest-exec']+'\n')
      self.getExecutable(self.argDB['with-ctest-exec'], getFullPath=1, resultName='ctest')
    else:
      self.log.write('Looking for default CTest executable\n')
      self.getExecutable('ctest', getFullPath=1, resultName='ctest')
    return

  def alternateConfigureLibrary(self):
    self.checkDownload()

  def configure(self):
    '''Locate cmake and download it if requested'''
    if self.argDB['download-cmake']:
      self.log.write('Building CMake\n')
      config.package.GNUPackage.configure(self)
      self.log.write('Looking for Cmake in '+os.path.join(self.installDir,'bin')+'\n')
      self.getExecutable('cmake',    path=os.path.join(self.installDir,'bin'), getFullPath = 1)
      self.getExecutable('ctest',    path=os.path.join(self.installDir,'bin'), getFullPath = 1)
    elif (not self.argDB['with-cmake']  == 0 and not self.argDB['with-cmake']  == 'no') or 'with-cmake-exec' in self.argDB:
      self.executeTest(self.locateCMake)
    else:
      self.log.write('Not checking for CMake\n')
    if hasattr(self, 'cmake'):
      import re
      self.found = 1
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(self.cmake+' --version', log = self.log)
        if status:
          self.log.write('cmake --version failed: '+str(error)+'\n')
          return
      except RuntimeError as e:
        self.log.write('cmake --version failed: '+str(e)+'\n')
        return
      output = output.replace('stdout: ','')
      gver = None
      try:
        gver = re.compile('cmake version ([0-9]+).([0-9]+).([0-9]+)').match(output)
      except: pass
      if gver:
        try:
           self.foundversion = ".".join(gver.groups())
        except: pass
        else:
          self.log.write('cmake version found '+self.foundversion+'\n')
          if self.versionToTuple(self.foundversion) < self.maxminCmakeVersion:
            raise RuntimeError('A package requires CMake version '+'.'.join(map(str, self.maxminCmakeVersion))+' (detected version is '+'.'.join(map(str, self.versionToTuple(self.foundversion)))+'): use --download-cmake')
          return
      gver = None
      try:
        gver = re.compile('cmake version ([0-9]+).([0-9]+)-patch ([0-9]+)').match(output)
      except: pass
      if gver:
        try:
           val = list(gver.groups())
           v = [val[0],val[1],'0',val[2]]
           self.foundversion = ".".join(v)
        except: pass
        else:
          self.log.write('cmake version found '+self.foundversion+'\n')
          if self.versionToTuple(self.foundversion) < self.maxminCmakeVersion:
            raise RuntimeError('A package requires CMake version '+'.'.join(map(str, self.maxminCmakeVersion))+' (detected version is '+'.'.join(map(str, self.versionToTuple(self.foundversion)))+'): use --download-cmake')
          return
        self.log.write('cmake version check failed\n')
    else:
      self.log.write('cmake not found\n')
    return
