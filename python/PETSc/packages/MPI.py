import config.base

import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix   = ''
    self.substPrefix    = ''
    self.argDB          = framework.argDB
    self.foundLib       = 0
    self.foundInclude   = 0
    self.compilers      = self.framework.require('config.compilers', self)
    self.types          = self.framework.require('config.types',     self)
    self.libraries      = self.framework.require('config.libraries', self)
    return

  def configureHelp(self, help):
    help.addOption('MPI', '-with-mpi', 'Use MPI')
    help.addOption('MPI', '-with-mpi-dir=<root dir>', 'Specify the root directory of the MPI installation')
    help.addOption('MPI', '-with-mpi-include=<dir>', 'The directory containing mpi.h')
    help.addOption('MPI', '-with-mpi-lib=<lib>', 'The MPI library or list of libraries')
    help.addOption('MPI', '-with-mpirun=<prog>', 'The utility used to launch MPI jobs')
    return

  def checkLib(self, fullLib):
    '''Checks for MPI_Init in fullLib, which can be a list of libraries'''
    if not isinstance(fullLib, list): fullLib = [fullLib]
    oldLibs  = self.framework.argDB['LIBS']
    self.dir = []
    self.lib = []
    for lib in fullLib:
      (dir, base) = os.path.split(lib)
      if dir and not dir in self.dir:
        self.dir.append(dir)
      if base[:3] == 'lib': base = base[3:]
      self.lib.append(os.path.splitext(base)[0])
    otherLibs = self.compilers.flibs
    if self.libraries.check(self.lib, 'MPI_Init', libDir = self.dir, otherLibs = otherLibs):
      self.foundLib = 1
    self.framework.argDB['LIBS'] = oldLibs
    return self.foundLib

  def checkInclude(self):
    oldFlags = self.framework.argDB['CPPFLAGS']
    if self.include: self.framework.argDB['CPPFLAGS'] += ' -I'+self.include
    if self.checkPreprocess('#include <mpi.h>\n'):
      self.foundInclude = 1
    self.framework.argDB['CPPFLAGS'] = oldFlags
    return self.foundInclude

  def checkMPILink(self, includes, body):
    success  = 0
    oldFlags = self.framework.argDB['CPPFLAGS']
    if self.include: self.framework.argDB['CPPFLAGS'] += ' -I'+self.include
    oldLibs  = self.framework.argDB['LIBS']
    for dir in self.dir:
      self.framework.argDB['LIBS'] += ' -L'+dir
    for lib in self.lib:
      self.framework.argDB['LIBS'] += ' -l'+lib
    self.framework.argDB['LIBS'] += ' '+self.compilers.flibs
    if self.checkLink(includes, body):
      success = 1
    self.framework.argDB['CPPFLAGS'] = oldFlags
    self.framework.argDB['LIBS']     = oldLibs
    return success

  def configureLibrary(self):
    '''Checking for the MPI library'''
    # Try specified library or default
    self.include = None
    if self.framework.argDB.has_key('with-mpi-lib'):
      fullLib = self.framework.argDB['with-mpi-lib']
      if not isinstance(fullLib, list):
        fullLib = [fullLib]
      if os.path.dirname(fullLib[0]):
        self.include = os.path.join(os.path.dirname(os.path.dirname(fullLib[0])), 'include')
    else:
      fullLib = 'libmpich.a'
    if self.checkLib(fullLib): return 1
    fullLib = 'libmpi.a'
    if self.checkLib(fullLib): return 1

    # Try library from MPI_DIR/lib
    if self.framework.argDB.has_key('with-mpi-dir'):
      self.baseDir = self.framework.argDB['with-mpi-dir']
    elif self.argDB.has_key('MPI_DIR'):
      self.baseDir = self.argDB['MPI_DIR']
    else:
      self.baseDir = ''
    if self.baseDir:
      mpiLibPath   = os.path.join(self.baseDir, 'lib')
      fullLib      = os.path.join(mpiLibPath, 'libmpich.a')
      self.include = os.path.join(self.baseDir, 'include')
      if self.checkLib(fullLib): return 1
      fullLib = os.path.join(mpiLibPath, 'libmpi.a')
      if self.checkLib(fullLib): return 1

    # Obsolete support for MPICH env variables
    if self.argDB.has_key('MPILIBPATH'):
      mpiLibPath = self.argDB['MPILIBPATH']
      fullLib    = os.path.join(mpiLibPath, 'libmpich.a')
      if self.argDB.has_key('MPIINCLUDEPATH'):
        self.include = self.argDB['MPIINCLUDEPATH']
      else:
        self.include = None
      if self.checkLib(fullLib): return 1
      fullLib = os.path.join(mpiLibPath, 'libmpi.a')
      if self.checkLib(fullLib): return 1
    return 0

  def configureInclude(self):
    '''Checking for the MPI include file'''
    if self.framework.argDB.has_key('with-mpi-include'):
      self.include = self.framework.argDB['with-mpi-include']
    elif self.include is None:
      self.include = ''
    if self.checkInclude(): return 1
    raise RuntimeError('Could not find MPI header mpi.h in '+self.include+'!')

  def configureTypes(self):
    '''Checking for MPI types'''
    oldFlags = self.framework.argDB['CPPFLAGS']
    if self.include: self.framework.argDB['CPPFLAGS'] += ' -I'+self.include
    self.types.checkSizeof('MPI_Comm', 'mpi.h')
    self.types.checkSizeof('MPI_Fint', 'mpi.h')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    return

  def checkWorkingLink(self):
    '''Checking that we can link an MPI executable'''
    if self.checkMPILink('#include <mpi.h>\n', 'MPI_Comm comm = MPI_COMM_WORLD;\nint size;\n\nMPI_Comm_size(comm, &size);\n'):
      return 1
    raise RuntimeError('MPI cannot link, which indicates a mismatch between the header ('+os.path.join(self.include, 'mpi.h')+
                       ') and library ('+str(self.lib)+').')

  def configureConversion(self):
    if self.checkMPILink('#include <mpi.h>\n', 'if (MPI_Comm_f2c(MPI_COMM_WORLD));\n'):
      self.addDefine('HAVE_MPI_COMM_F2C', 1)
    if self.checkMPILink('#include <mpi.h>\n', 'if (MPI_Comm_c2f(MPI_COMM_WORLD));\n'):
      self.addDefine('HAVE_MPI_COMM_C2F', 1)
    return

  def configureMPIRUN(self):
    '''Checking for mpirun'''
    if self.framework.argDB.has_key('with-mpirun'):
      self.mpirun = self.framework.argDB['with-mpirun']
    else:
      self.mpirun = 'mpirun'
    path = os.path.dirname(self.mpirun)
    for dir in self.dir:
      path += ':'+os.path.join(os.path.dirname(dir), 'bin')
    if path[0] == ':': path = path[1:]
    if not path[-1] == ':': path += ':'
    self.getExecutable('mpirun', path = path, getFullPath = 1)
    return

  def configureMPE(self):
    '''Checking for MPE'''
    self.addSubstitution('MPE_INCLUDE', '')
    self.addSubstitution('MPE_LIB',     '')
    return

  def setOutput(self):
    if self.foundLib and self.foundInclude:
      self.addDefine('HAVE_MPI', 1)
      if self.include:
        self.addSubstitution('MPI_INCLUDE', '-I'+self.include)
      else:
        self.addSubstitution('MPI_INCLUDE', '')
      self.addSubstitution('MPI_LIB_DIR', self.dir and self.dir[0])
      libFlag = ''
      for dir in self.dir:
        libFlag = '-L'+dir+' '
      for lib in self.lib:
        libFlag += ' -l'+lib
      self.addSubstitution('MPI_LIB', libFlag)
    return

  def configure(self):
    if not self.framework.argDB.has_key('with-mpi') or not int(self.framework.argDB['with-mpi']): return
    self.executeTest(self.configureLibrary)
    if self.foundLib:
      self.executeTest(self.configureInclude)
    if self.foundLib and self.foundInclude:
      self.executeTest(self.configureTypes)
      self.executeTest(self.checkWorkingLink)
      self.executeTest(self.configureConversion)
      self.executeTest(self.configureMPIRUN)
      self.executeTest(self.configureMPE)
    self.setOutput()
    return
