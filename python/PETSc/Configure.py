import config.base

import commands
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.defineAutoconfMacros()
    headersC = map(lambda name: name+'.h', ['dos', 'endian', 'fcntl', 'io', 'limits', 'malloc', 'pwd', 'search', 'strings',
                                            'stropts', 'unistd', 'machine/endian', 'sys/param', 'sys/procfs', 'sys/resource',
                                            'sys/stat', 'sys/systeminfo', 'sys/times', 'sys/utsname','string', 'stdlib'])
    functions = ['access', '_access', 'clock', 'drand48', 'getcwd', '_getcwd', 'getdomainname', 'gethostname', 'getpwuid',
                 'gettimeofday', 'getrusage', 'getwd', 'memalign', 'memmove', 'mkstemp', 'popen', 'PXFGETARG', 'rand',
                 'readlink', 'realpath', 'sbreak', 'sigaction', 'signal', 'sigset', 'sleep', '_sleep', 'socket', 'times',
                 'uname']
    libraries = [('dl', 'dlopen')]
    self.compilers   = self.framework.require('config.compilers', self)
    self.types       = self.framework.require('config.types',     self)
    self.headers     = self.framework.require('config.headers',   self)
    self.functions   = self.framework.require('config.functions', self)
    self.libraries   = self.framework.require('config.libraries', self)
    self.blas        = self.framework.require('PETSc.packages.BLAS',   self)
    self.lapack      = self.framework.require('PETSc.packages.LAPACK', self)
    self.mpi         = self.framework.require('PETSc.packages.MPI',    self)
    self.adic        = self.framework.require('PETSc.packages.ADIC',        self)
    self.matlab      = self.framework.require('PETSc.packages.Matlab',      self)
    self.mathematica = self.framework.require('PETSc.packages.Mathematica', self)
    self.triangle    = self.framework.require('PETSc.packages.Triangle',    self)
    self.parmetis    = self.framework.require('PETSc.packages.ParMetis',    self)
    self.plapack     = self.framework.require('PETSc.packages.PLAPACK',     self)
    self.pvode       = self.framework.require('PETSc.packages.PVODE',       self)
    self.blocksolve  = self.framework.require('PETSc.packages.BlockSolve',  self)
    self.headers.headers.extend(headersC)
    self.functions.functions.extend(functions)
    self.libraries.libraries.extend(libraries)
    # Put all defines in the PETSc namespace
    self.compilers.headerPrefix   = self.headerPrefix
    self.types.headerPrefix       = self.headerPrefix
    self.headers.headerPrefix     = self.headerPrefix
    self.functions.headerPrefix   = self.headerPrefix
    self.libraries.headerPrefix   = self.headerPrefix
    self.blas.headerPrefix        = self.headerPrefix
    self.lapack.headerPrefix      = self.headerPrefix
    self.mpi.headerPrefix         = self.headerPrefix
    self.adic.headerPrefix        = self.headerPrefix
    self.matlab.headerPrefix      = self.headerPrefix
    self.mathematica.headerPrefix = self.headerPrefix
    self.triangle.headerPrefix    = self.headerPrefix
    self.parmetis.headerPrefix    = self.headerPrefix
    self.plapack.headerPrefix     = self.headerPrefix
    self.pvode.headerPrefix       = self.headerPrefix
    self.blocksolve.headerPrefix  = self.headerPrefix
    return

  def configureHelp(self, help):
    import nargs

    help.addOption('PETSc', 'PETSC_DIR', 'The root directory of the PETSc installation')
    help.addOption('PETSc', 'PETSC_ARCH', 'The machine architecture')
    help.addOption('PETSc', '-enable-debug', 'Activate debugging code in PETSc', nargs.ArgBool)
    help.addOption('PETSc', '-enable-log', 'Activate logging code in PETSc', nargs.ArgBool)
    help.addOption('PETSc', '-enable-stack', 'Activate manual stack tracing code in PETSc', nargs.ArgBool)
    help.addOption('PETSc', '-enable-dynamic', 'Build dynamic libraries for PETSc', nargs.ArgBool)
    help.addOption('PETSc', '-enable-fortran-kernels', 'Use Fortran for linear algebra kernels', nargs.ArgBool)
    help.addOption('PETSc', 'optionsModule=<module name>', 'The Python module used to determine compiler options and versions')
    help.addOption('PETSc', 'C_VERSION', 'The version of the C compiler')
    help.addOption('PETSc', 'CFLAGS_g', 'Flags for the C compiler with BOPT=g')
    help.addOption('PETSc', 'CFLAGS_O', 'Flags for the C compiler with BOPT=O')
    help.addOption('PETSc', 'CXX_VERSION', 'The version of the C++ compiler')
    help.addOption('PETSc', 'CXXFLAGS_g', 'Flags for the C++ compiler with BOPT=g')
    help.addOption('PETSc', 'CXXFLAGS_O', 'Flags for the C++ compiler with BOPT=O')
    help.addOption('PETSc', 'F_VERSION', 'The version of the Fortran compiler')
    help.addOption('PETSc', 'FFLAGS_g', 'Flags for the Fortran compiler with BOPT=g')
    help.addOption('PETSc', 'FFLAGS_O', 'Flags for the Fortran compiler with BOPT=O')
    help.addOption('PETSc', '-with-libtool', 'Specify that libtool should be used for compiling and linking', nargs.ArgBool)
    help.addOption('PETSc', '-with-make', 'Specify make')
    help.addOption('PETSc', '-with-ranlib', 'Specify ranlib')

    self.framework.argDB['enable-debug']           = 1
    self.framework.argDB['enable-log']             = 1
    self.framework.argDB['enable-stack']           = 1
    self.framework.argDB['enable-dynamic']         = 1
    self.framework.argDB['enable-fortran-kernels'] = 0
    self.framework.argDB['C_VERSION']              = 'Unknown'
    self.framework.argDB['CFLAGS_g']               = '-g'
    self.framework.argDB['CFLAGS_O']               = '-O'
    self.framework.argDB['CXX_VERSION']            = 'Unknown'
    self.framework.argDB['CXXFLAGS_g']             = '-g'
    self.framework.argDB['CXXFLAGS_O']             = '-O'
    self.framework.argDB['F_VERSION']              = 'Unknown'
    self.framework.argDB['FFLAGS_g']               = '-g'
    self.framework.argDB['FFLAGS_O']               = '-O'
    self.framework.argDB['PETSCFLAGS']             = ''
    self.framework.argDB['COPTFLAGS']              = ''
    self.framework.argDB['FOPTFLAGS']              = ''
    self.framework.argDB['with-libtool']           = 0
    self.framework.argDB['with-make']              = 'make'
    self.framework.argDB['with-ranlib']            = 'ranlib'

    self.framework.argDB['BOPT'] = 'O'
    return

  def defineAutoconfMacros(self):
    self.hostMacro = 'dnl Version: 2.13\ndnl Variable: host_cpu\ndnl Variable: host_vendor\ndnl Variable: host_os\nAC_CANONICAL_HOST'
    return

  def checkRequirements(self):
    '''Checking that packages Petsc required are actually here'''
    if not self.blas.found: raise RuntimeError('Petsc requires BLAS!\n Could not link to '+str(self.blas.lib)+'. Check configure.log.')
    if not self.lapack.found: raise RuntimeError('Petsc requires LAPACK!\n Could not link to '+str(self.lapack.lib)+'. Check configure.log.')
    return

  def configureDirectories(self):
    '''Sets PETSC_DIR'''
    if not self.framework.argDB.has_key('PETSC_DIR'):
      self.framework.argDB['PETSC_DIR'] = os.getcwd()
    self.dir = self.framework.argDB['PETSC_DIR']
    # Check for version
    if not os.path.exists(os.path.join(self.dir, 'include', 'petscversion.h')):
      raise RuntimeError('Invalid PETSc directory '+str(self.dir))
    self.addSubstitution('DIR', self.dir)
    self.addDefine('DIR', self.dir)
    return

  def configureArchitecture(self):
    '''Sets PETSC_ARCH'''
    import sys
    # Find auxilliary directory by checking for config.sub
    auxDir = None
    for dir in [os.path.abspath(os.path.join('bin', 'config')), os.path.abspath('config')] + sys.path:
      if os.path.isfile(os.path.join(dir, 'config.sub')):
        auxDir      = dir
        configSub   = os.path.join(auxDir, 'config.sub')
        configGuess = os.path.join(auxDir, 'config.guess')
        break
    if not auxDir: raise RuntimeError('Unable to locate config.sub in order to determine architecture')
    # Try to execute config.sub
    (status, output) = commands.getstatusoutput(self.shell+' '+configSub+' sun4')
    if status: raise RuntimeError('Unable to execute config.sub: '+output)
    # Guess host type (should allow user to specify this
    (status, host) = commands.getstatusoutput(self.shell+' '+configGuess)
    if status: raise RuntimeError('Unable to guess host type using config.guess: '+host)
    # Get full host description
    (status, output) = commands.getstatusoutput(self.shell+' '+configSub+' '+host)
    if status: raise RuntimeError('Unable to determine host type using config.sub: '+output)
    # Parse output
    m = re.match(r'^(?P<cpu>[^-]*)-(?P<vendor>[^-]*)-(?P<os>.*)$', output)
    if not m: raise RuntimeError('Unable to parse output of config.sub: '+output)
    self.host_cpu    = m.group('cpu')
    self.host_vendor = m.group('vendor')
    self.host_os     = m.group('os')

##    results = self.executeShellCode(self.macroToShell(self.hostMacro))
##    self.host_cpu    = results['host_cpu']
##    self.host_vendor = results['host_vendor']
##    self.host_os     = results['host_os']

    if not self.framework.argDB.has_key('PETSC_ARCH'):
      self.arch = self.host_os
    else:
      self.arch = self.framework.argDB['PETSC_ARCH']
    if not self.arch.startswith(self.host_os):
      raise RuntimeError('PETSC_ARCH ('+self.arch+') does not have our guess ('+self.host_os+') as a prefix!')
    self.addSubstitution('ARCH', self.arch)
    self.archBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.arch)
    self.addDefine('ARCH', self.archBase)
    self.addDefine('ARCH_NAME', '"'+self.arch+'"')
    return

  def configureLibraryOptions(self):
    '''Sets PETSC_USE_DEBUG, PETSC_USE_LOG, PETSC_USE_STACK, and PETSC_USE_FORTRAN_KERNELS'''
    self.useDebug = self.framework.argDB['enable-debug']
    self.addDefine('USE_DEBUG', self.useDebug)
    self.useLog   = self.framework.argDB['enable-log']
    self.addDefine('USE_LOG',   self.useLog)
    self.useStack = self.framework.argDB['enable-stack']
    self.addDefine('USE_STACK', self.useStack)
    self.useFortranKernels = self.framework.argDB['enable-fortran-kernels']
    self.addDefine('USE_FORTRAN_KERNELS', self.useFortranKernels)
    return

  def configureCompilerFlags(self):
    '''Get all compiler flags from the Petsc database'''
    options = None
    try:
      mod     = __import__('PETSc.Options', locals(), globals(), ['Options'])
      options = mod.Options()
    except ImportError: print 'Failed to load generic options'
    try:
      if self.framework.argDB.has_key('optionsModule'):
        mod     = __import__(self.framework.argDB['optionsModule'], locals(), globals(), ['Options'])
        options = mod.Options()
    except ImportError: print 'Failed to load custom options'
    if options:
      # We use the framework in order to remove the PETSC_ namespace
      self.framework.argDB['C_VERSION']   = options.getCompilerVersion('C',       self.compilers.CC,  self)
      self.framework.argDB['CXX_VERSION'] = options.getCompilerVersion('Cxx',     self.compilers.CXX, self)
      self.framework.argDB['F_VERSION']   = options.getCompilerVersion('Fortran', self.compilers.FC,  self)

      self.framework.argDB['CFLAGS_g']    = options.getCompilerFlags('C',       self.compilers.CC,  'g', self)
      self.framework.argDB['CFLAGS_O']    = options.getCompilerFlags('C',       self.compilers.CC,  'O', self)
      self.framework.argDB['CXXFLAGS_g']  = options.getCompilerFlags('Cxx',     self.compilers.CXX, 'g', self)
      self.framework.argDB['CXXFLAGS_O']  = options.getCompilerFlags('Cxx',     self.compilers.CXX, 'O', self)
      self.framework.argDB['FFLAGS_g']    = options.getCompilerFlags('Fortran', self.compilers.FC,  'g', self)
      self.framework.argDB['FFLAGS_O']    = options.getCompilerFlags('Fortran', self.compilers.FC,  'O', self)

    self.addSubstitution('C_VERSION',   self.framework.argDB['C_VERSION'])
    self.addSubstitution('CFLAGS_g',    self.framework.argDB['CFLAGS_g'])
    self.addSubstitution('CFLAGS_O',    self.framework.argDB['CFLAGS_O'])
    self.addSubstitution('CXX_VERSION', self.framework.argDB['CXX_VERSION'])
    self.addSubstitution('CXXFLAGS_g',  self.framework.argDB['CXXFLAGS_g'])
    self.addSubstitution('CXXFLAGS_O',  self.framework.argDB['CXXFLAGS_O'])
    self.addSubstitution('F_VERSION',   self.framework.argDB['F_VERSION'])
    self.addSubstitution('FFLAGS_g',    self.framework.argDB['FFLAGS_g'])
    self.addSubstitution('FFLAGS_O',    self.framework.argDB['FFLAGS_O'])

    self.framework.addSubstitution('PETSCFLAGS', self.framework.argDB['PETSCFLAGS'])
    self.framework.addSubstitution('COPTFLAGS',  self.framework.argDB['COPTFLAGS'])
    self.framework.addSubstitution('FOPTFLAGS',  self.framework.argDB['FOPTFLAGS'])
    return

  def checkF77CompilerOption(self, option):
    oldFlags = self.framework.argDB['FFLAGS']
    success  = 0
    self.framework.argDB['FFLAGS'] = option
    self.pushLanguage('F77')
    if self.checkCompile('', ''):
      success = 1
    self.framework.argDB['FFLAGS'] = oldFlags
    self.popLanguage()
    return success

  def configureFortranPIC(self):
    '''Determine the PIC option for the Fortran compiler'''
    # We use the framework in order to remove the PETSC_ namespace
    option = ''
    if self.checkF77CompilerOption('-PIC'):
      option = '-PIC'
    elif self.checkF77CompilerOption('-fPIC'):
      option = '-fPIC'
    elif self.checkF77CompilerOption('-KPIC'):
      option = '-KPIC'
    self.framework.addSubstitution('FC_SHARED_OPT', option)
    return

  def configureFortranStubs(self):
    '''Determine whether the Fortran stubs exist or not'''
    stubDir = os.path.join(self.framework.argDB['PETSC_DIR'], 'src', 'fortran', 'auto')
    if not os.path.exists(os.path.join(stubDir, 'makefile.src')):
      print '  WARNING: Fortran stubs have not been generated in '+stubDir
    return

  def configureDynamicLibraries(self):
    '''Checks whether dynamic libraries should be used, for which you must
      - Specify --enable-dynamic
      - Find dlfcn.h and libdl
    Defines PETSC_USE_DYNAMIC_LIBRARIES is they are used
    Also checks that dlopen() takes RTLD_GLOBAL, and defines PETSC_HAVE_RTLD_GLOBAL if it does'''
    useDynamic = self.framework.argDB['enable-dynamic'] and self.headers.check('dlfcn.h') and self.libraries.haveLib('dl')
    self.addDefine('USE_DYNAMIC_LIBRARIES', useDynamic)
    if useDynamic and self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);\n'):
      self.addDefine('HAVE_RTLD_GLOBAL', 1)
    # This is really bad
    flag = '-L'
    if self.archBase == 'linux':
      flag = '-rdynamic -Wl,-rpath,'
    self.addSubstitution('CLINKER_SLFLAG', flag)
    self.addSubstitution('FLINKER_SLFLAG', flag)
    return

  def configureLibtool(self):
    if self.framework.argDB['with-libtool']:
      self.framework.addSubstitution('LT_CC', '${PETSC_LIBTOOL} ${LIBTOOL} --mode=compile')
      self.framework.addSubstitution('LIBTOOL', '${SHELL} ${top_builddir}/libtool')
      self.framework.addSubstitution('SHARED_TARGET', 'shared_libtool')
    else:
      self.framework.addSubstitution('LT_CC', '')
      self.framework.addSubstitution('LIBTOOL', '')
      self.framework.addSubstitution('SHARED_TARGET', 'shared_'+self.archBase)
    return

  def configureDebuggers(self):
    '''Find a default debugger and determine its arguments'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.getExecutable('gdb', getFullPath = 1)
    self.framework.getExecutable('dbx', getFullPath = 1)
    self.framework.getExecutable('xdb', getFullPath = 1)
    if hasattr(self, 'gdb'):
      self.addDefine('USE_GDB_DEBUGGER', 1)
    elif hasattr(self, 'dbx'):
      self.addDefine('USE_DBX_DEBUGGER', 1)
      f = file('conftest', 'w')
      f.write('quit\n')
      f.close()
      foundOption = 0
      if not foundOption:
        (status, output) = commands.getstatusoutput(self.dbx+' -c conftest -p '+os.getpid())
        for line in output:
          if re.match(r'Process '+os.getpid()):
            self.addDefine('USE_P_FOR_DEBUGGER', 1)
            foundOption = 1
            break
      if not foundOption:
        (status, output) = commands.getstatusoutput(self.dbx+' -c conftest -a '+os.getpid())
        for line in output:
          if re.match(r'Process '+os.getpid()):
            self.addDefine('USE_A_FOR_DEBUGGER', 1)
            foundOption = 1
            break
      if not foundOption:
        (status, output) = commands.getstatusoutput(self.dbx+' -c conftest -pid '+os.getpid())
        for line in output:
          if re.match(r'Process '+os.getpid()):
            self.addDefine('USE_PID_FOR_DEBUGGER', 1)
            foundOption = 1
            break
      os.remove('conftest')
    elif hasattr(self, 'xdb'):
      self.addDefine('USE_XDB_DEBUGGER', 1)
      self.addDefine('USE_LARGEP_FOR_DEBUGGER', 1)
    return

  def configureMake(self):
    '''Check various things about make'''
    self.framework.getExecutable(self.framework.argDB['with-make'], getFullPath = 1)
    # Check for GNU make
    haveGNUMake = 0
    try:
      import commands
      (status, output) = commands.getstatusoutput('strings '+self.framework.make)
      if not status and output.find('GNU Make') >= 0:
        haveGNUMake = 1
    except Exception, e: print 'Make check failed: '+str(e)
    # Setup make flags
    flags = ''
    if haveGNUMake:
      flags += ' --no-print-directory'
    self.framework.addSubstitution('MAKE_FLAGS', flags.strip())
    self.framework.addSubstitution('SET_MAKE', '')
    # Check to see if make allows rules which look inside archives
    if haveGNUMake:
      self.framework.addSubstitution('LIB_C_TARGET', 'libc: ${LIBNAME}(${OBJSC} ${SOBJSC})')
      self.framework.addSubstitution('LIB_F_TARGET', '''
libf: ${OBJSF}
	${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSF}''')
    else:
      self.framework.addSubstitution('LIB_C_TARGET', '''
libc: ${OBJSC}
	${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSC}''')
      self.framework.addSubstitution('LIB_F_TARGET', '''
libf: ${OBJSF}
	${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSF}''')
    return

  def configureMkdir(self):
    '''Make sure we can have mkdir automatically make intermediate directories'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.getExecutable('mkdir', getFullPath = 1)
    if hasattr(self.framework, 'mkdir'):
      self.mkdir = self.framework.mkdir
      if os.path.exists('.conftest'): os.rmdir('.conftest')
      (status, output) = commands.getstatusoutput(self.mkdir+' -p .conftest/.tmp')
      if not status and os.path.isdir('.conftest/.tmp'):
        self.mkdir = self.mkdir+' -p'
        self.framework.addSubstitution('MKDIR', self.mkdir)
      if os.path.exists('.conftest'): os.removedirs('.conftest/.tmp')
    return

  def configurePrograms(self):
    '''Check for the programs needed to build and run PETSc'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.getExecutable('sh',   getFullPath = 1, resultName = 'SHELL')
    self.framework.getExecutable('sed',  getFullPath = 1)
    self.framework.getExecutable('diff', getFullPath = 1)
    self.framework.getExecutable('ar',   getFullPath = 1)
    self.framework.addSubstitution('AR_FLAGS', 'cr')
    self.framework.getExecutable(self.framework.argDB['with-ranlib'])
    self.framework.getExecutable('ps', path = '/usr/ucb:/usr/usb', resultName = 'UCBPS')
    if hasattr(self, 'UCBPS'):
      self.addDefine('HAVE_UCBPS', 1)
    return

  def configureMissingPrototypes(self):
    '''Checks for missing prototypes, which it adds to petscfix.h'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.addSubstitution('MISSING_PROTOTYPES',     '')
    self.framework.addSubstitution('MISSING_PROTOTYPES_CXX', '')
    self.missingPrototypesExternC = ''
    self.framework.addSubstitution('MISSING_PROTOTYPES_EXTERN_C', self.missingPrototypesExternC)
    return

  def configureMissingFunctions(self):
    '''Checks for MISSING_GETPWUID and MISSING_SOCKETS'''
    if not self.functions.haveFunction('getpwuid'):
      self.addDefine('MISSING_GETPWUID', 1)
    if not self.functions.haveFunction('socket'):
      self.addDefine('MISSING_SOCKETS', 1)
    return

  def configureMissingSignals(self):
    '''Check for missing signals, and define MISSING_<signal name> if necessary'''
    if not self.checkCompile('#include <signal.h>\n', 'int i=SIGSYS;\n\nif (i);\n'):
      self.addDefine('MISSING_SIGSYS', 1)
    if not self.checkCompile('#include <signal.h>\n', 'int i=SIGBUS;\n\nif (i);\n'):
      self.addDefine('MISSING_SIGBUS', 1)
    if not self.checkCompile('#include <signal.h>\n', 'int i=SIGQUIT;\n\nif (i);\n'):
      self.addDefine('MISSING_SIGQUIT', 1)
    return

  def configureMemorySize(self):
    '''Try to determine how to measure the memory usage'''
    # Should also check for using procfs and kbytes for size
    if self.functions.haveFunction('sbreak'):
      self.addDefine('USE_SBREAK_FOR_SIZE', 1)
    elif not (self.headers.haveHeader('sys/resource.h') and self.functions.haveFunction('getrusage')):
        self.addDefine('HAVE_NO_GETRUSAGE', 1)
    return

  def checkXMake(self):
    import shutil

    includeDir = ''
    libraryDir = ''
    # Create Imakefile
    dir = os.path.abspath('conftestdir')
    if os.path.exists(dir): shutil.rmtree(dir)
    os.mkdir(dir)
    os.chdir(dir)
    f = file('Imakefile', 'w')
    f.write('''
acfindx:
	@echo \'X_INCLUDE_ROOT = ${INCROOT}\'
	@echo \'X_USR_LIB_DIR = ${USRLIBDIR}\'
	@echo \'X_LIB_DIR = ${LIBDIR}\'
''')
    f.close()
    # Compile makefile
    if not os.system('xmkmf >/dev/null') and os.path.exists('Makefile'):
      import commands
      (status, output) = commands.getstatusoutput(self.framework.make+' acfindx')
      results          = self.parseShellOutput(output)
      # Open Windows xmkmf reportedly sets LIBDIR instead of USRLIBDIR.
      for ext in ['.a', '.so', '.sl']:
        if not os.path.isfile(os.path.join(results['X_USR_LIB_DIR'])) and os.path.isfile(os.path.join(results['X_LIB_DIR'])):
          results['X_USR_LIB_DIR'] = results['X_LIB_DIR']
          break
      # Screen out bogus values from the imake configuration.  They are
      # bogus both because they are the default anyway, and because
      # using them would break gcc on systems where it needs fixed includes.
      if not results['X_INCLUDE_ROOT'] == '/usr/include' and os.path.isfile(os.path.join(results['X_INCLUDE_ROOT'], 'X11', 'Xos.h')):
        includeDir = results['X_INCLUDE_ROOT']
      if not (results['X_USR_LIB_DIR'] == '/lib' or results['X_USR_LIB_DIR'] == '/usr/lib') and os.path.isdir(results['X_USR_LIB_DIR']):
        libraryDir = results['X_USR_LIB_DIR']
    # Cleanup
    os.chdir(os.path.dirname(dir))
    shutil.rmtree(dir)
    return (includeDir, libraryDir)

  def configureX(self):
    '''Checks for X windows, sets PETSC_HAVE_X11 if found, and defines X_CFLAGS, X_PRE_LIBS, X_LIBS, and X_EXTRA_LIBS'''
    foundInclude = 0
    includeDirs  = ['/usr/X11/include',
                   '/usr/X11R6/include',
                   '/usr/X11R5/include',
                   '/usr/X11R4/include',
                   '/usr/include/X11',
                   '/usr/include/X11R6',
                   '/usr/include/X11R5',
                   '/usr/include/X11R4',
                   '/usr/local/X11/include',
                   '/usr/local/X11R6/include',
                   '/usr/local/X11R5/include',
                   '/usr/local/X11R4/include',
                   '/usr/local/include/X11',
                   '/usr/local/include/X11R6',
                   '/usr/local/include/X11R5',
                   '/usr/local/include/X11R4',
                   '/usr/X386/include',
                   '/usr/x386/include',
                   '/usr/XFree86/include/X11',
                   '/usr/include',
                   '/usr/local/include',
                   '/usr/unsupported/include',
                   '/usr/athena/include',
                   '/usr/local/x11r5/include',
                   '/usr/lpp/Xamples/include',
                   '/usr/openwin/include',
                   '/usr/openwin/share/include']
    includeDir   = ''
    foundLibrary = 0
    libraryDirs  = map(lambda s: s.replace('include', 'lib'), includeDirs)
    libraryDir   = ''
    if not self.framework.argDB.has_key('no_x'):
      # Guess X location
      (includeDirGuess, libraryDirGuess) = self.checkXMake()
      # Check for X11 includes
      if self.framework.argDB.has_key('with-x-include'):
        if not os.path.isdir(self.framework.argDB['with-x-include']):
          raise RuntimeError('Invalid X include directory specified: '+os.path.abspath(self.framework.argDB['with-x-include']))
        includeDir = self.framework.argDB['with-x-include']
      else:
        testInclude  = 'X11/Intrinsic.h'

        # Check guess
        if includeDirGuess and os.path.isfile(os.path.join(includeDirGuess, testInclude)):
          foundInclude = 1
          includeDir   = includeDirGuess
        # Check default compiler paths
        if not foundInclude and self.checkPreprocess('#include <'+testInclude+'>\n'):
          foundInclude = 1
        # Check standard paths
        if not foundInclude:
          for dir in includeDirs:
            if os.path.isfile(os.path.join(dir, testInclude)):
              foundInclude = 1
              includeDir   = dir
      # Check for X11 libraries
      if self.framework.argDB.has_key('with-x-library'):
        if not os.path.isfile(self.framework.argDB['with-x-library']):
          raise RuntimeError('Invalid X library specified: '+os.path.abspath(self.framework.argDB['with-x-library']))
        libraryDir = os.path.dirname(self.framework.argDB['with-x-library'])
      else:
        testLibrary  = 'Xt'
        testFunction = 'XtMalloc'

        # Check guess
        if libraryDirGuess:
          for ext in ['.a', '.so', '.sl']:
            if os.path.isfile(os.path.join(libraryDirGuess, 'lib'+testLibrary+ext)):
              foundLibrary = 1
              libraryDir   = libraryDirGuess
              break
        # Check default compiler libraries
        if not foundLibrary:
          oldLibs = self.framework.argDB['LIBS']
          self.framework.argDB['LIBS'] += ' -l'+testLibrary
          self.pushLanguage(self.language[-1])
          if self.checkLink('', testFunction+'();\n'):
            foundLibrary = 1
          self.framework.argDB['LIBS'] = oldLibs
          self.popLanguage()
        # Check standard paths
        if not foundLibrary:
          for dir in libraryDirs:
            for ext in ['.a', '.so', '.sl']:
              if os.path.isfile(os.path.join(dir, 'lib'+testLibrary+ext)):
                foundLibrary = 1
                libraryDir   = dir

    if not foundInclude or not foundLibrary:
      self.addDefine('HAVE_X11', 0)
      self.framework.addSubstitution('X_CFLAGS',     '')
      self.framework.addSubstitution('X_PRE_LIBS',   '')
      self.framework.addSubstitution('X_LIBS',       '')
      self.framework.addSubstitution('X_EXTRA_LIBS', '')
    else:
      self.addDefine('HAVE_X11', 1)
      if includeDir:
        self.framework.addSubstitution('X_CFLAGS',   '-I'+includeDir)
      else:
        self.framework.addSubstitution('X_CFLAGS',   '')
      if libraryDir:
        self.framework.addSubstitution('X_LIBS',     '-L'+libraryDir+' -lX11')
      else:
        self.framework.addSubstitution('X_LIBS',     '-lX11')
      self.framework.addSubstitution('X_PRE_LIBS',   '')
      self.framework.addSubstitution('X_EXTRA_LIBS', '')
    return

  def configureFPTrap(self):
    '''Checking the handling of floating point traps'''
    if self.headers.check('sigfpe.h'):
      if self.functions.check('handle_sigfpes', libraries = 'fpe'):
        self.addDefine('HAVE_IRIX_STYLE_FPTRAP', 1)
    elif self.headers.check('fpxcp.h') and self.headers.check('fptrap.h'):
      if reduce(lambda x,y: x and y, map(self.functions.check, ['fp_sh_trap_info', 'fp_trap', 'fp_enable', 'fp_disable'])):
        self.addDefine('HAVE_RS6000_STYLE_FPTRAP', 1)
    elif self.headers.check('floatingpoint.h'):
      if self.functions.check('ieee_flags') and self.functions.check('ieee_handler'):
        if self.headers.check('sunmath.h'):
          self.addDefine('HAVE_SOLARIS_STYLE_FPTRAP', 1)
        else:
          self.addDefine('HAVE_SUN4_STYLE_FPTRAP', 1)
    return

  def configureLibrarySuffix(self):
    '''(Belongs in config.libraries) Determine the suffix used for libraries'''
    # This is exactly like the libtool check
    if self.archBase.find('win') >= 0 and not self.compilers.CC == 'gcc':
      suffix = 'lib'
    else:
      suffix = 'a'
    self.libraries.addSubstitution('LIB_SUFFIX', suffix)
    return

  def configureIRIX(self):
    '''IRIX specific stuff'''
    if self.archBase == 'irix6.5':
      self.addDefine('USE_KBYTES_FOR_SIZE', 1)
    return

  def configureLinux(self):
    '''Linux specific stuff'''
    if self.archBase == 'linux':
      self.addDefine('HAVE_DOUBLE_ALIGN_MALLOC', 1)
    return

  def configureMacOSX(self):
    '''Mac specific stuff'''
    if self.archBase.startswith('darwin'):
      self.framework.addDefine('MISSING_PROTOTYPES_EXTERN_C', 'void *memalign(size_t, size_t)')
    return

  def configureMachineInfo(self):
    '''Define a string incorporating all configuration data needed for a bug report'''
    self.addDefine('PETSC_MACHINE_INFO', '"Libraries compiled on `date` on `hostname`\\nMachine characteristics: `uname -a`\\n-----------------------------------------\\nUsing C compiler: ${CC} ${COPTFLAGS} ${CCPPFLAGS}\\nC Compiler version: ${C_VERSION}\\nUsing C compiler: ${CXX} ${CXXOPTFLAGS} ${CXXCPPFLAGS}\\nC++ Compiler version: ${CXX_VERSION}\\nUsing Fortran compiler: ${FC} ${FOPTFLAGS} ${FCPPFLAGS}\\nFortran Compiler version: ${F_VERSION}\\n-----------------------------------------\\nUsing PETSc flags: ${PETSCFLAGS} ${PCONF}\\n-----------------------------------------\\nUsing include paths: ${PETSC_INCLUDE}\\n-----------------------------------------\\nUsing PETSc directory: ${PETSC_DIR}\\nUsing PETSc arch: ${PETSC_ARCH}"\\n')
    return

  def configureMPIUNI(self):
    '''If MPI was not found, setup MPIUNI, our uniprocessor version of MPI'''
    if self.mpi.foundInclude and self.mpi.foundLib: return
    raise RuntimeError('Could not find MPI!')
##    if self.mpiuni:
##      print '********** Warning: Using uniprocessor MPI (mpiuni) from Petsc **********'
##      print '**********     Use --with-mpi option to specify a full MPI     **********'
    return

  def configureMisc(self):
    '''Fix up all the things that we currently need to run'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.addSubstitution('CC_SHARED_OPT', '')
    return

  def configure(self):
    self.executeTest(self.checkRequirements)
    self.executeTest(self.configureDirectories)
    self.executeTest(self.configureArchitecture)
    self.framework.header = 'bmake/'+self.arch+'/petscconf.h'
    self.framework.addSubstitutionFile('bmake/config/packages.in',   'bmake/'+self.arch+'/packages')
    self.framework.addSubstitutionFile('bmake/config/rules.in',      'bmake/'+self.arch+'/rules')
    self.framework.addSubstitutionFile('bmake/config/variables.in',  'bmake/'+self.arch+'/variables')
    self.framework.addSubstitutionFile('bmake/config/petscfix.h.in', 'bmake/'+self.arch+'/petscfix.h')
    self.executeTest(self.configureLibraryOptions)
    self.executeTest(self.configureCompilerFlags)
    self.executeTest(self.configureFortranPIC)
    self.executeTest(self.configureFortranStubs)
    self.executeTest(self.configureDynamicLibraries)
    self.executeTest(self.configureLibtool)
    self.executeTest(self.configureDebuggers)
    self.executeTest(self.configureMake)
    self.executeTest(self.configureMkdir)
    self.executeTest(self.configurePrograms)
    self.executeTest(self.configureMissingPrototypes)
    self.executeTest(self.configureMissingFunctions)
    self.executeTest(self.configureMissingSignals)
    self.executeTest(self.configureMemorySize)
    self.executeTest(self.configureX)
    self.executeTest(self.configureFPTrap)
    self.executeTest(self.configureLibrarySuffix)
    self.executeTest(self.configureIRIX)
    self.executeTest(self.configureLinux)
    self.executeTest(self.configureMacOSX)
    self.executeTest(self.configureMachineInfo)
    self.executeTest(self.configureMisc)
    self.executeTest(self.configureMPIUNI)
    return
