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
    self.checkedInclude = []
    self.compilers      = self.framework.require('config.compilers', self)
    self.types          = self.framework.require('config.types',     self)
    self.libraries      = self.framework.require('config.libraries', self)
    return

  def configureHelp(self, help):
    help.addOption('MPI', '-with-mpi', 'Use MPI')
    help.addOption('MPI', '-with-dir=<root dir>', 'Specify the root directory of the MPI installation')
    help.addOption('MPI', '-with-mpi-include=<dir>', 'The directory containing mpi.h')
    help.addOption('MPI', '-with-mpi-lib=<lib>', 'The MPI library')
    help.addOption('MPI', '-with-mpirun=<prog>', 'The utility used to launch MPI jobs')
    return

  def checkLib(self, fullLib):
    oldLibs              = self.framework.argDB['LIBS']
    (self.dir, self.lib) = os.path.split(self.fullLib)
    self.lib             = os.path.splitext(self.lib)[0]
    if self.lib.startswith('lib'): self.lib = self.lib[3:]
    otherLibs            = self.compilers.flibs
    if self.libraries.check(self.lib, 'MPI_Init', libDir = self.dir, otherLibs = otherLibs):
      self.foundLib = 1
    self.framework.argDB['LIBS'] = oldLibs
    return self.foundLib

  def checkInclude(self):
    if self.include in self.checkedInclude: return
    oldFlags = self.framework.argDB['CPPFLAGS']
    if self.include: self.framework.argDB['CPPFLAGS'] += ' -I'+self.include
    if self.checkPreprocess('#include <mpi.h>\n'):
      self.foundInclude = 1
    self.framework.argDB['CPPFLAGS'] = oldFlags
    self.checkedInclude.append(self.include)
    return self.foundInclude

  def checkMPILink(self, includes, body):
    success  = 0
    oldFlags = self.framework.argDB['CPPFLAGS']
    if self.include: self.framework.argDB['CPPFLAGS'] += ' -I'+self.include
    oldLibs  = self.framework.argDB['LIBS']
    if self.dir: self.framework.argDB['LIBS'] += ' -L'+self.dir
    self.framework.argDB['LIBS'] += ' -l'+self.lib+' '+self.compilers.flibs
    if self.checkLink(includes, body):
      success = 1
    self.framework.argDB['CPPFLAGS'] = oldFlags
    self.framework.argDB['LIBS']     = oldLibs
    return success

  def configureLibrary(self):
    '''Checking for the MPI library'''
    if self.framework.argDB.has_key('with-mpi-lib'):
      self.fullLib = self.framework.argDB['with-mpi-lib']
    else:
      self.fullLib = 'libmpich.a'
    self.include = ''
    if os.path.dirname(self.fullLib):
      self.include = os.path.join(os.path.dirname(os.path.dirname(self.fullLib)), 'include')
    if self.checkLib(self.fullLib): return 1
    self.fullLib = 'libmpi.a'
    if self.checkLib(self.fullLib): return 1

    if self.framework.argDB.has_key('with-mpi-dir'):
      self.baseDir = self.framework.argDB['with-mpi-dir']
    elif self.argDB.has_key('MPI_DIR'):
      self.baseDir = self.argDB['MPI_DIR']
    else:
      self.baseDir = ''
    if self.baseDir:
      mpiLibPath   = os.path.join(self.baseDir, 'lib')
      self.fullLib = os.path.join(mpiLibPath, 'libmpich.a')
      self.include = os.path.join(self.baseDir, 'include')
      if self.checkLib(self.fullLib): return 1
      self.fullLib = os.path.join(mpiLibPath, 'libmpi.a')
      if self.checkLib(self.fullLib): return 1

    # Obsolete support for MPICH env variables
    if self.argDB.has_key('MPILIBPATH'):
      mpiLibPath   = self.argDB['MPILIBPATH']
      self.fullLib = os.path.join(mpiLibPath, 'libmpich.a')
      if self.argDB.has_key('MPIINCLUDEPATH'):
        self.include = self.argDB['MPIINCLUDEPATH']
      else:
        self.include = ''
      if self.checkLib(self.fullLib): return 1
      self.fullLib = os.path.join(mpiLibPath, 'libmpi.a')
      if self.checkLib(self.fullLib): return 1
    return 0

  def configureInclude(self):
    '''Checking for the MPI include file'''
    if self.checkInclude(): return 1

    if self.framework.argDB.has_key('with-mpi-include'):
      self.include = self.framework.argDB['with-mpi-include']
    else:
      self.include = ''
    if self.checkInclude(): return 1
    raise RuntimeError('Could not find MPI header mpi.h in '+str(self.checkedInclude)+'!')

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
                       ') and library ('+self.lib+').')

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
    path = os.path.join(os.path.dirname(self.dir), 'bin')+':'+os.path.dirname(self.mpirun)
    if not path[-1] == ':': path += ':'
    self.getExecutable('mpirun', path = path, getFullPath = 1, comment = 'The utility for launching MPI jobs')
    return

  def configureMPE(self):
    '''Checking for MPE'''
    self.addSubstitution('MPE_INCLUDE', '', 'The MPE include flags')
    self.addSubstitution('MPE_LIB',     '', 'The MPE library flags')
    return

  def setOutput(self):
    if self.foundLib and self.foundInclude:
      self.addDefine('HAVE_MPI', 1)
      if self.include:
        self.addSubstitution('MPI_INCLUDE', '-I'+self.include, 'The MPI include flags')
      else:
        self.addSubstitution('MPI_INCLUDE', '', 'The MPI include flags')
      self.addSubstitution('MPI_LIB_DIR', self.dir, 'The MPI library directory')
      libFlag = ''
      if self.dir: libFlag = '-L'+self.dir+' '
      libFlag += '-l'+self.lib
      self.addSubstitution('MPI_LIB',     libFlag, 'The MPI library flags')
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
