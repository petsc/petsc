from __future__ import generators
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
    self.headers        = self.framework.require('config.headers',   self)
    self.libraries      = self.framework.require('config.libraries', self)
    self.headers.headers.append('dlfcn.h')
    self.libraries.libraries.append(('dl', 'dlopen'))
    return

  def configureHelp(self, help):
    import nargs
    help.addArgument('MPI', '-with-mpi-dir=<root dir>', nargs.ArgDir(None, None, 'Specify the root directory of the MPI installation'))
    help.addArgument('MPI', '-with-mpi-include=<dir>',  nargs.ArgDir(None, None, 'The directory containing mpi.h'))
    help.addArgument('MPI', '-with-mpi-lib=<lib>',      nargs.Arg(None, None, 'The MPI library or list of libraries'))
    help.addArgument('MPI', '-with-mpirun=<prog>',      nargs.Arg(None, None, 'The utility used to launch MPI jobs'))
    help.addArgument('MPI', '-with-mpi-shared',         nargs.ArgBool(None, 0, 'Require that the MPI library be shared'))
    return

  def executeShellCommand(self, command):
    '''Execute a shell command returning the output, and optionally provide a custom error checker'''
    import commands

    (status, output) = commands.getstatusoutput(command)
    if status: raise RuntimeError('Could not execute \''+command+'\':\n'+output)
    return output

  def checkLib(self, libraries):
    '''Check for MPI_Init and MPI_Comm_create in libraries, which can be a list of libraries or a single library'''
    if not isinstance(libraries, list): libraries = [libraries]
    oldLibs = self.framework.argDB['LIBS']
    found   = self.libraries.check(libraries, 'MPI_Init', otherLibs = self.compilers.flibs)
    if found:
      found = self.libraries.check(libraries, 'MPI_Comm_create',otherLibs=self.compilers.flibs)
    self.framework.argDB['LIBS'] = oldLibs
    return found

  def checkInclude(self, includeDir):
    '''Check that mpi.h is present'''
    oldFlags = self.framework.argDB['CPPFLAGS']
    for inc in includeDir:
      self.framework.argDB['CPPFLAGS'] += ' -I'+inc
    found = self.checkPreprocess('#include <mpi.h>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    return found

  def checkMPILink(self, includes, body, cleanup = 1):
    '''Analogous to checkLink(), but the MPI includes and libraries are automatically provided'''
    success  = 0
    oldFlags = self.framework.argDB['CPPFLAGS']
    oldLibs  = self.framework.argDB['LIBS']
    for inc in self.include:
      self.framework.argDB['CPPFLAGS'] += ' -I'+inc
    for lib in self.lib:
      self.framework.argDB['LIBS'] += ' '+self.libraries.getLibArgument(lib)
    self.framework.argDB['LIBS'] += ' '+self.compilers.flibs
    if self.checkLink(includes, body, cleanup):
      success = 1
    self.framework.argDB['CPPFLAGS'] = oldFlags
    self.framework.argDB['LIBS']     = oldLibs
    return success

  def outputMPIRun(self, includes, body, cleanup = 1):
    '''Analogous to outputRun(), but the MPI includes and libraries are automatically provided'''
    oldFlags = self.framework.argDB['CPPFLAGS']
    oldLibs  = self.framework.argDB['LIBS']
    for inc in self.include:
      self.framework.argDB['CPPFLAGS'] += ' -I'+inc
    for lib in self.lib:
      self.framework.argDB['LIBS'] += ' '+self.libraries.getLibArgument(lib)
    self.framework.argDB['LIBS'] += ' '+self.compilers.flibs
    output, status = self.outputRun(includes, body, cleanup)
    self.framework.argDB['CPPFLAGS'] = oldFlags
    self.framework.argDB['LIBS']     = oldLibs
    return (output, status)

  def checkWorkingLink(self):
    '''Checking that we can link an MPI executable'''
    if self.checkMPILink('#include <mpi.h>\n', 'MPI_Comm comm = MPI_COMM_WORLD;\nint size;\n\nMPI_Comm_size(comm, &size);\n'):
      return 1
    self.framework.log.write('MPI cannot link, which indicates a mismatch between the header ('+os.path.join(self.include[0], 'mpi.h')+') and library ('+str(self.lib)+').\n')
    return 0

  def checkSharedLibrary(self):
    '''Check that the libraries for MPI are shared libraries'''
    isShared                         = 0
    oldFlags                         = self.framework.argDB['LDFLAGS']
    self.framework.argDB['LDFLAGS'] += ' -shared'
    for lib in self.lib:
      self.framework.argDB['LDFLAGS'] += ' -Wl,-rpath,'+os.path.dirname(lib)

    # Make a library which calls MPI_Init
    self.codeBegin = '''
#ifdef __cplusplus
extern "C"
#endif
int init(int argc,  char *argv[]) {
'''
    self.codeEnd   = '\n}\n'
    body           = '''
  int isInitialized;

  MPI_Init(&argc, &argv);
  MPI_Initialized(&isInitialized);
  return isInitialized;
'''
    if not self.checkMPILink('#include <mpi.h>\n', body, cleanup = 0):
      if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
      self.framework.argDB['LDFLAGS'] = oldFlags
      self.framework.log.write('Could not complete shared library check\n')
      return 0
    if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    os.rename(self.linkerObj, 'lib1.so')

    # Make a library which calls MPI_Initialized
    self.codeBegin = '''
#ifdef __cplusplus
extern "C"
#endif
int checkInit(void) {
'''
    self.codeEnd   = '\n}\n'
    body           = '''
  int isInitialized;

  MPI_Initialized(&isInitialized);
  return isInitialized;
'''
    if not self.checkMPILink('#include <mpi.h>\n', body, cleanup = 0):
      if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
      self.framework.argDB['LDFLAGS'] = oldFlags
      self.framework.log.write('Could not complete shared library check\n')
      return 0
    if os.path.isfile(self.compilerObj): os.remove(self.compilerObj)
    os.rename(self.linkerObj, 'lib2.so')

    self.codeBegin = ''
    self.codeEnd   = ''
    self.framework.argDB['LDFLAGS'] = oldFlags

    guard = self.headers.getDefineName('dlfcn.h')
    if self.headers.headerPrefix:
      guard = self.headers.headerPrefix+'_'+guard
    includes = '''
#include <stdio.h>
#include <stdlib.h>
#ifdef %s
  #include <dlfcn.h>
#endif
    ''' % guard
    body = '''
  int   argc    = 1;
  char *argv[1] = {"conftest"};
  void *lib;
  int (*init)(int, char **);
  int (*checkInit)(void);

  lib = dlopen("./lib1.so", RTLD_LAZY);
  if (!lib) {
    fprintf(stderr, "Could not open lib1.so: %s\\n", dlerror());
    exit(1);
  }
  init = (int (*)(int, char **)) dlsym(lib, "init");
  if (!init) {
    fprintf(stderr, "Could not find initialization function\\n");
    exit(1);
  }
  if (!(*init)(argc, argv)) {
    fprintf(stderr, "Could not initialize MPI\\n");
    exit(1);
  }
  lib = dlopen("./lib2.so", RTLD_LAZY);
  if (!lib) {
    fprintf(stderr, "Could not open lib2.so: %s\\n", dlerror());
    exit(1);
  }
  checkInit = (int (*)(void)) dlsym(lib, "checkInit");
  if (!checkInit) {
    fprintf(stderr, "Could not find initialization check function\\n");
    exit(1);
  }
  if (!(*checkInit)()) {
    fprintf(stderr, "Did not link with shared library\\n");
    exit(2);
  }
    '''
    oldLibs = self.framework.argDB['LIBS']
    if self.libraries.haveLib('dl'):
      self.framework.argDB['LIBS'] += ' -ldl'
    if self.checkRun(includes, body):
      isShared = 1
    self.framework.argDB['LIBS'] = oldLibs
    if os.path.isfile('lib1.so'): os.remove('lib1.so')
    if os.path.isfile('lib2.so'): os.remove('lib2.so')
    if not isShared:
      self.framework.log.write('MPI library was not shared\n')
    return isShared

  def configureVersion(self):
    '''Determine the MPI version'''
    output, status = self.outputMPIRun('#include <stdio.h>\n#include <mpi.h>\n', 'int ver, subver;\n if (MPI_Get_version(&ver, &subver));\nprintf("%d.%d\\n", ver, subver)\n')
    if not status:
      return output
    return 'Unknown'

  def includeGuesses(self, path):
    '''Return all include directories present in path or its ancestors'''
    while path:
      dir = os.path.join(path, 'include')
      if os.path.isdir(dir):
        yield [dir]
      path = os.path.dirname(path)
    return

  def libraryGuesses(self, root = None):
    '''Return standard library name guesses for a given installation root'''
    if root:
      yield [os.path.join(root, 'lib', 'shared', 'libmpich.a')]
      yield [os.path.join(root, 'lib', 'shared', 'libmpi.a')]
      yield [os.path.join(root, 'lib', 'shared', 'libmpich.a'), os.path.join(root, 'lib', 'shared', 'libpmpich.a')]
      yield [os.path.join(root, 'lib', 'libmpich.a')]
      yield [os.path.join(root, 'lib', 'libmpi.a')]
      yield [os.path.join(root, 'lib', 'libmpich.a'), os.path.join(root, 'lib', 'libpmpich.a')]
    else:
      yield ['']
      yield ['mpich']
      yield ['mpi']
      yield ['mpich', 'pmpich']
    return

  def generateGuesses(self):
    # Try specified library and include
    if 'with-mpi-lib' in self.framework.argDB:
      libs = self.framework.argDB['with-mpi-lib']
      if not isinstance(libs, list): libs = [libs]
      if 'with-mpi-include' in self.framework.argDB:
        includes = [[self.framework.argDB['with-mpi-include']]]
      else:
        includes = self.includeGuesses(map(lambda inc: os.path.dirname(os.path.dirname(inc)), libs))
      yield ('User specified library and includes', [libs], includes)
    # Try specified installation root
    if 'with-mpi-dir' in self.framework.argDB:
      dir = self.framework.argDB['with-mpi-dir']
      yield ('User specified installation root', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try compiler defaults
    yield ('Default compiler locations', self.libraryGuesses(), [['']])
    # Try SUSE location
    dir = os.path.abspath(os.path.join('/opt', 'mpich'))
    yield ('Default SUSE location', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
    # Try /usr/local
    dir = os.path.abspath(os.path.join('/usr', 'local'))
    yield ('Frequent user install location (/usr/local)', self.libraryGuesses(dir), [[os.path.join(dir, 'include')]])
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
      try:
        libArgs = self.executeShellCommand('cd '+PETSC_DIR+'; make BOPT=g_c++ getmpilinklibs')
        incArgs = self.executeShellCommand('cd '+PETSC_DIR+'; make BOPT=g_c++ getmpiincludedirs')
        runArgs = self.executeShellCommand('cd '+PETSC_DIR+'; make getmpirun')

        libArgs = self.splitLibs(libArgs)
        incArgs = self.splitIncludes(incArgs)
        if runArgs and not 'with-mpirun' in self.framework.argDB:
          self.framework.argDB['with-mpirun'] = runArgs
        if libArgs and incArgs:
          yield ('PETSc location', [libArgs], [incArgs])
      except RuntimeError:
        # This happens with older Petsc versions which are missing those targets
        pass
    return

  def configureLibrary(self):
    '''Find all working MPI libraries and then choose one'''
    self.foundMPI = 0
    functionalMPI = []
    nonsharedMPI  = []
    for (name, libraryGuesses, includeGuesses) in self.generateGuesses():
      self.framework.log.write('================================================================================\n')
      self.framework.log.write('Checking for a functional MPI in '+name+'\n')
      self.lib     = None
      self.include = None
      for libraries in libraryGuesses:
        if self.checkLib(libraries):
          self.lib = libraries
          break
      if not self.lib: continue
      for includeDir in includeGuesses:
        if self.checkInclude(includeDir):
          self.include = includeDir
          break
      if not self.include: continue
      if not self.executeTest(self.checkWorkingLink): continue
      version = self.executeTest(self.configureVersion)
      if self.framework.argDB['with-mpi-shared']:
        if not self.executeTest(self.checkSharedLibrary):
          nonsharedMPI.append((name, self.lib, self.include, version))
          continue
      self.foundMPI = 1
      functionalMPI.append((name, self.lib, self.include, version))
    # User chooses one or take first (sort by version)
    if self.foundMPI:
      name, self.lib, self.include, version = functionalMPI[0]
      self.framework.log.write('Choose MPI '+version+' in '+name+'\n')
    elif len(nonsharedMPI):
      raise RuntimeError('Could not locate any MPI with shared libraries')
    else:
      raise RuntimeError('Could not locate any functional MPI\n Rerun config/configure.py with --with-mpi=0 to install without MPI\n or -with-mpi-dir=directory to use a MPICH installation\n or -with-mpi-include=directory -with-mpi-lib=library (or list of libraries) -with-mpirun=mpiruncommand for other MPIs')
    return

  def configureTypes(self):
    '''Checking for MPI types'''
    oldFlags = self.framework.argDB['CPPFLAGS']
    for inc in self.include: self.framework.argDB['CPPFLAGS'] += ' -I'+inc
    self.types.checkSizeof('MPI_Comm', 'mpi.h')
    self.types.checkSizeof('MPI_Fint', 'mpi.h')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    return

  def configureConversion(self):
    '''Check for the functions which convert communicators between C and Fortran
       - Define HAVE_MPI_COMM_F2C and HAVE_MPI_COMM_C2F if they are present
       - Some older MPI 1 implementations are missing these'''
    if self.checkMPILink('#include <mpi.h>\n', 'if (MPI_Comm_f2c(MPI_COMM_WORLD));\n'):
      self.addDefine('HAVE_MPI_COMM_F2C', 1)
    if self.checkMPILink('#include <mpi.h>\n', 'if (MPI_Comm_c2f(MPI_COMM_WORLD));\n'):
      self.addDefine('HAVE_MPI_COMM_C2F', 1)
    return

  def configureMPIRUN(self):
    '''Checking for mpirun'''
    if 'with-mpirun' in self.framework.argDB:
      self.mpirun = self.framework.argDB['with-mpirun']
    else:
      self.mpirun = 'mpirun'
    path = []
    if os.path.dirname(self.mpirun):
      path.append(os.path.dirname(self.mpirun))
    if 'with-mpi-dir' in self.framework.argDB:
      path.append(os.path.join(self.framework.argDB['with-mpi-dir'], 'bin'))
    for inc in self.include:
      path.append(os.path.join(os.path.dirname(inc), 'bin'))
    for lib in self.lib:
      path.append(os.path.join(os.path.dirname(os.path.dirname(lib)), 'bin'))
    path.append('')
    self.getExecutable('mpirun', path = ':'.join(path), getFullPath = 1)
    return

  def setOutput(self):
    '''Add defines and substitutions
       - HAVE_MPI is defined if a working MPI is found
       - MPI_INCLUDE and MPI_LIB are command line arguments for the compile and link
       - MPI_INCLUDE_DIR is the directory containing mpi.h
       - MPI_LIBRARY is the list of MPI libraries'''
    if self.foundMPI:
      self.addDefine('HAVE_MPI', 1)
      if self.include:
        self.addSubstitution('MPI_INCLUDE',     ' '.join(['-I'+inc for inc in self.include]))
        self.addSubstitution('MPI_INCLUDE_DIR', self.include[0])
      else:
        self.addSubstitution('MPI_INCLUDE',     '')
        self.addSubstitution('MPI_INCLUDE_DIR', '')
      self.addSubstitution('MPI_LIB',     ' '.join(map(self.libraries.getLibArgument, self.lib)))
      self.addSubstitution('MPI_LIBRARY', self.lib)
    return

  def configure(self):
    if 'with-mpi' in self.framework.argDB and not self.framework.argDB['with-mpi']:
      return
    self.executeTest(self.configureLibrary)
    if self.foundMPI:
      self.executeTest(self.configureTypes)
      self.executeTest(self.configureConversion)
      self.executeTest(self.configureMPIRUN)
    self.setOutput()
    return
