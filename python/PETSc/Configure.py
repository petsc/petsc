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
    self.defineAutoconfShell()
    headersC = map(lambda name: name+'.h', ['dos', 'endian', 'fcntl', 'io', 'limits', 'malloc', 'pwd', 'search', 'strings',
                                            'stropts', 'unistd', 'machine/endian', 'sys/param', 'sys/procfs', 'sys/resource',
                                            'sys/stat', 'sys/systeminfo', 'sys/times', 'sys/utsname','string', 'stdlib'])
    functions = ['access', '_access', 'clock', 'drand48', 'getcwd', '_getcwd', 'getdomainname', 'gethostname', 'getpwuid',
                 'gettimeofday', 'getwd', 'memalign', 'memmove', 'mkstemp', 'popen', 'PXFGETARG', 'rand', 'readlink',
                 'realpath', 'sigaction', 'signal', 'sigset', 'sleep', '_sleep', 'socket', 'times', 'uname']
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
    help.addOption('PETSc', 'optionsModule', 'The Python module used to determine compiler options and versions')
    help.addOption('PETSc', 'C_VERSION', 'The version of the C compiler')
    help.addOption('PETSc', 'CFLAGS_g', 'Flags for the C compiler with BOPT=g')
    help.addOption('PETSc', 'CFLAGS_O', 'Flags for the C compiler with BOPT=O')
    help.addOption('PETSc', 'CXX_VERSION', 'The version of the C++ compiler')
    help.addOption('PETSc', 'CXXFLAGS_g', 'Flags for the C++ compiler with BOPT=g')
    help.addOption('PETSc', 'CXXFLAGS_O', 'Flags for the C++ compiler with BOPT=O')
    help.addOption('PETSc', 'F_VERSION', 'The version of the Fortran compiler')
    help.addOption('PETSc', 'FFLAGS_g', 'Flags for the Fortran compiler with BOPT=g')
    help.addOption('PETSc', 'FFLAGS_O', 'Flags for the Fortran compiler with BOPT=O')

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

    self.framework.argDB['BOPT'] = 'O'
    return

  def defineAutoconfMacros(self):
    self.hostMacro = 'dnl Version: 2.13\ndnl Variable: host_cpu\ndnl Variable: host_vendor\ndnl Variable: host_os\nAC_CANONICAL_HOST'
    self.xMacro    = 'dnl Version: 2.13\ndnl Variable: X_CFLAGS\ndnl Variable: X_LIBS\ndnl Variable: X_EXTRA_LIBS\ndnl Variable: X_PRE_LIBS\nAC_PATH_XTRA'
    return

  def checkRequirements(self):
    '''Checking that packages Petsc required are actually here'''
    if not self.blas.found: raise RuntimeError('Petsc requires BLAS!\n Could not link to '+self.blas.fullLib+'. Check configure.log.')
    if not self.lapack.found: raise RuntimeError('Petsc requires LAPACK!\n Could not link to '+self.lapack.fullLib+'. Check configure.log.')
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

  def configureDynamicLibraries(self):
    '''Checks for --enable-dynamic, and defines PETSC_USE_DYNAMIC_LIBRARIES if it is given
    Also checks that dlopen() takes RTLD_GLOBAL, and defines PETSC_HAVE_RTLD_GLOBAL if it does'''
    useDynamic = self.framework.argDB['enable-dynamic']
    self.addDefine('USE_DYNAMIC_LIBRARIES', useDynamic and self.libraries.haveLib('dl'))
    if self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);\n'):
      self.addDefine('HAVE_RTLD_GLOBAL', 1)
    # This is really bad
    flag = '-L'
    if self.archBase == 'linux':
      flag = '-rdynamic -Wl,-rpath,'
    self.addSubstitution('CLINKER_SLFLAG', flag)
    self.addSubstitution('FLINKER_SLFLAG', flag)
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

  def checkMkdir(self):
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
    self.checkMkdir()
    self.framework.getExecutable('sh',   getFullPath = 1, resultName = 'SHELL')
    self.framework.getExecutable('sed',  getFullPath = 1)
    self.framework.getExecutable('diff', getFullPath = 1)
    self.framework.getExecutable('ar',   getFullPath = 1)
    self.framework.getExecutable('make')
    self.framework.addSubstitution('AR_FLAGS', 'cr')
    self.framework.getExecutable('ranlib')
    self.framework.addSubstitution('SET_MAKE', '')
    self.framework.addSubstitution('LIBTOOL', '${SHELL} ${top_builddir}/libtool')
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
    if not self.functions.defines.has_key(self.functions.getDefineName('getpwuid')):
      self.addDefine('MISSING_GETPWUID', 1)
    if not self.functions.defines.has_key(self.functions.getDefineName('socket')):
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

  def configureX(self):
    '''Uses AC_PATH_XTRA, and sets PETSC_HAVE_X11'''
    if not self.framework.argDB.has_key('no_x'):
      os.environ['with_x']      = 'yes'
      os.environ['x_includes']  = 'NONE'
      os.environ['x_libraries'] = 'NONE'
##      results = self.executeShellCode(self.macroToShell(self.xMacro))
      results = self.executeShellCode(self.xShell)
      # LIBS="$LIBS $X_PRE_LIBS $X_LIBS $X_EXTRA_LIBS"
      self.addDefine('HAVE_X11', 1)
      self.framework.addSubstitution('X_CFLAGS',     results['X_CFLAGS'])
      self.framework.addSubstitution('X_PRE_LIBS',   results['X_PRE_LIBS'])
      self.framework.addSubstitution('X_LIBS',       results['X_LIBS']+' -lX11')
      self.framework.addSubstitution('X_EXTRA_LIBS', results['X_EXTRA_LIBS'])
    return

  def configureFPTrap(self):
    '''Checking the handling of flaoting point traps'''
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
    self.framework.addSubstitution('LT_CC', '${PETSC_LIBTOOL} ${LIBTOOL} --mode=compile')
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
    self.executeTest(self.configureDynamicLibraries)
    self.executeTest(self.configureDebuggers)
    self.executeTest(self.configurePrograms)
    self.executeTest(self.configureMissingPrototypes)
    self.executeTest(self.configureMissingFunctions)
    self.executeTest(self.configureMissingSignals)
    self.executeTest(self.configureX)
    self.executeTest(self.configureFPTrap)
    self.executeTest(self.configureIRIX)
    self.executeTest(self.configureLinux)
    self.executeTest(self.configureMachineInfo)
    self.executeTest(self.configureMisc)
    self.executeTest(self.configureMPIUNI)
    return

  def defineAutoconfShell(self):
    # This is here because the long string is screwing up my font coloring
    self.xShell = ('''
# If we find X, set shell vars x_includes and x_libraries to the
# paths, otherwise set no_x=yes.
# Uses ac_ vars as temps to allow command line to override cache and checks.
# --without-x overrides everything else, but does not touch the cache.
echo $ac_n "checking for X""... $ac_c" 1>&2
echo "configure:0: checking for X" >&3

# Check whether --with-x or --without-x was given.
if test "${with_x+set}" = set; then
  withval="$with_x"
  :
fi

# $have_x is yes, no, disabled, or empty when we do not yet know.
if test "x$with_x" = xno; then
  # The user explicitly disabled X.
  have_x=disabled
else
  if test "x$x_includes" != xNONE && test "x$x_libraries" != xNONE; then
    # Both variables are already set.
    have_x=yes
  else
if eval "test \"`echo \'$\'\'{\'ac_cv_have_x\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  # One or both of the vars are not set, and there is no cached value.
ac_x_includes=NO ac_x_libraries=NO
rm -fr conftestdir
if mkdir conftestdir; then
  cd conftestdir
  # Make sure to not put "make" in the Imakefile rules, since we grep it out.
  cat > Imakefile <<\'EOF\'
acfindx:
	@echo \'ac_im_incroot="${INCROOT}"; ac_im_usrlibdir="${USRLIBDIR}"; ac_im_libdir="${LIBDIR}"\'
EOF
  if (xmkmf) >/dev/null 2>/dev/null && test -f Makefile; then
    # GNU make sometimes prints "make[1]: Entering...", which would confuse us.
    eval `${MAKE-make} acfindx 2>/dev/null | grep -v make`
    # Open Windows xmkmf reportedly sets LIBDIR instead of USRLIBDIR.
    for ac_extension in a so sl; do
      if test ! -f $ac_im_usrlibdir/libX11.$ac_extension &&
        test -f $ac_im_libdir/libX11.$ac_extension; then
        ac_im_usrlibdir=$ac_im_libdir; break
      fi
    done
    # Screen out bogus values from the imake configuration.  They are
    # bogus both because they are the default anyway, and because
    # using them would break gcc on systems where it needs fixed includes.
    case "$ac_im_incroot" in
	/usr/include) ;;
	*) test -f "$ac_im_incroot/X11/Xos.h" && ac_x_includes="$ac_im_incroot" ;;
    esac
    case "$ac_im_usrlibdir" in
	/usr/lib | /lib) ;;
	*) test -d "$ac_im_usrlibdir" && ac_x_libraries="$ac_im_usrlibdir" ;;
    esac
  fi
  cd ..
  rm -fr conftestdir
fi

if test "$ac_x_includes" = NO; then
  # Guess where to find include files, by looking for this one X11 .h file.
  test -z "$x_direct_test_include" && x_direct_test_include=X11/Intrinsic.h

  # First, try using that file with no special directory specified.
cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
#include <$x_direct_test_include>
EOF
ac_try="$ac_cpp conftest.$ac_ext >/dev/null 2>conftest.out"
{ (eval echo configure:0: \"$ac_try\") 1>&3; (eval $ac_try) 2>&3; }
ac_err=`grep -v \\'^ *+\\' conftest.out | grep -v "^conftest.${ac_ext}\$"`
if test -z "$ac_err"; then
  rm -rf conftest*
  # We can compile using X headers with no special include directory.
ac_x_includes=
else
  echo "$ac_err" >&3
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  # Look for the header file in a standard set of common directories.
# Check X11 before X11Rn because it is often a symlink to the current release.
  for ac_dir in               \
    /usr/X11/include          \
    /usr/X11R6/include        \
    /usr/X11R5/include        \
    /usr/X11R4/include        \
                              \
    /usr/include/X11          \
    /usr/include/X11R6        \
    /usr/include/X11R5        \
    /usr/include/X11R4        \
                              \
    /usr/local/X11/include    \
    /usr/local/X11R6/include  \
    /usr/local/X11R5/include  \
    /usr/local/X11R4/include  \
                              \
    /usr/local/include/X11    \
    /usr/local/include/X11R6  \
    /usr/local/include/X11R5  \
    /usr/local/include/X11R4  \
                              \
    /usr/X386/include         \
    /usr/x386/include         \
    /usr/XFree86/include/X11  \
                              \
    /usr/include              \
    /usr/local/include        \
    /usr/unsupported/include  \
    /usr/athena/include       \
    /usr/local/x11r5/include  \
    /usr/lpp/Xamples/include  \
                              \
    /usr/openwin/include      \
    /usr/openwin/share/include \
    ; \
  do
    if test -r "$ac_dir/$x_direct_test_include"; then
      ac_x_includes=$ac_dir
      break
    fi
  done
fi
rm -f conftest*
fi # $ac_x_includes = NO

if test "$ac_x_libraries" = NO; then
  # Check for the libraries.

  test -z "$x_direct_test_library" && x_direct_test_library=Xt
  test -z "$x_direct_test_function" && x_direct_test_function=XtMalloc

  # See if we find them without any special options.
  # Do not add to $LIBS permanently.
  ac_save_LIBS="$LIBS"
  LIBS="-l$x_direct_test_library $LIBS"
cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"

int main() {
${x_direct_test_function}()
; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  LIBS="$ac_save_LIBS"
# We can link X programs with no special library path.
ac_x_libraries=
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  LIBS="$ac_save_LIBS"
# First see if replacing the include by lib works.
# Check X11 before X11Rn because it is often a symlink to the current release.
for ac_dir in `echo "$ac_x_includes" | sed s/include/lib/` \
    /usr/X11/lib          \
    /usr/X11R6/lib        \
    /usr/X11R5/lib        \
    /usr/X11R4/lib        \
                          \
    /usr/lib/X11          \
    /usr/lib/X11R6        \
    /usr/lib/X11R5        \
    /usr/lib/X11R4        \
                          \
    /usr/local/X11/lib    \
    /usr/local/X11R6/lib  \
    /usr/local/X11R5/lib  \
    /usr/local/X11R4/lib  \
                          \
    /usr/local/lib/X11    \
    /usr/local/lib/X11R6  \
    /usr/local/lib/X11R5  \
    /usr/local/lib/X11R4  \
                          \
    /usr/X386/lib         \
    /usr/x386/lib         \
    /usr/XFree86/lib/X11  \
                          \
    /usr/lib              \
    /usr/local/lib        \
    /usr/unsupported/lib  \
    /usr/athena/lib       \
    /usr/local/x11r5/lib  \
    /usr/lpp/Xamples/lib  \
    /lib/usr/lib/X11	  \
                          \
    /usr/openwin/lib      \
    /usr/openwin/share/lib \
    ; \
do
  for ac_extension in a so sl; do
    if test -r $ac_dir/lib${x_direct_test_library}.$ac_extension; then
      ac_x_libraries=$ac_dir
      break 2
    fi
  done
done
fi
rm -f conftest*
fi # $ac_x_libraries = NO

if test "$ac_x_includes" = NO || test "$ac_x_libraries" = NO; then
  # Did not find X anywhere.  Cache the known absence of X.
  ac_cv_have_x="have_x=no"
else
  # Record where we found X for the cache.
  ac_cv_have_x="have_x=yes \
	        ac_x_includes=$ac_x_includes ac_x_libraries=$ac_x_libraries"
fi
fi
  fi
  eval "$ac_cv_have_x"
fi # $with_x != no

if test "$have_x" != yes; then
  echo "$ac_t""$have_x" 1>&2
  no_x=yes
else
  # If each of the values was on the command line, it overrides each guess.
  test "x$x_includes" = xNONE && x_includes=$ac_x_includes
  test "x$x_libraries" = xNONE && x_libraries=$ac_x_libraries
  # Update the cache value to reflect the command line values.
  ac_cv_have_x="have_x=yes \
		ac_x_includes=$x_includes ac_x_libraries=$x_libraries"
  echo "$ac_t""libraries $x_libraries, headers $x_includes" 1>&2
fi


ac_aux_dir=
for ac_dir in config bin/config $srcdir/config; do
  if test -f $ac_dir/install-sh; then
    ac_aux_dir=$ac_dir
    ac_install_sh="$ac_aux_dir/install-sh -c"
    break
  elif test -f $ac_dir/install.sh; then
    ac_aux_dir=$ac_dir
    ac_install_sh="$ac_aux_dir/install.sh -c"
    break
  fi
done
if test -z "$ac_aux_dir"; then
  { echo "configure: error: can not find install-sh or install.sh in config bin/config $srcdir/config" 1>&2; exit 1; }
fi
ac_config_guess=$ac_aux_dir/config.guess
ac_config_sub=$ac_aux_dir/config.sub
ac_configure=$ac_aux_dir/configure # This should be Cygnus configure.

ac_help="$ac_help
  --with-x                use the X Window System"
if test "$no_x" = yes; then
  # Not all programs may use this symbol, but it does not hurt to define it.
  cat >> confdefs.h <<\EOF
#define X_DISPLAY_MISSING 1
EOF

  X_CFLAGS= X_PRE_LIBS= X_LIBS= X_EXTRA_LIBS=
else
  if test -n "$x_includes"; then
    X_CFLAGS="$X_CFLAGS -I$x_includes"
  fi

  # It would also be nice to do this for all -L options, not just this one.
  if test -n "$x_libraries"; then
    X_LIBS="$X_LIBS -L$x_libraries"
    # For Solaris; some versions of Sun CC require a space after -R and
    # others require no space.  Words are not sufficient . . . .
    case "`(uname -sr) 2>/dev/null`" in
    "SunOS 5"*)
      echo $ac_n "checking whether -R must be followed by a space""... $ac_c" 1>&2
echo "configure:0: checking whether -R must be followed by a space" >&3
      ac_xsave_LIBS="$LIBS"; LIBS="$LIBS -R$x_libraries"
      cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"

int main() {

; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  ac_R_nospace=yes
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  ac_R_nospace=no
fi
rm -f conftest*
      if test $ac_R_nospace = yes; then
	echo "$ac_t""no" 1>&2
	X_LIBS="$X_LIBS -R$x_libraries"
      else
	LIBS="$ac_xsave_LIBS -R $x_libraries"
	cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"

int main() {

; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  ac_R_space=yes
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  ac_R_space=no
fi
rm -f conftest*
	if test $ac_R_space = yes; then
	  echo "$ac_t""yes" 1>&2
	  X_LIBS="$X_LIBS -R $x_libraries"
	else
	  echo "$ac_t""neither works" 1>&2
	fi
      fi
      LIBS="$ac_xsave_LIBS"
    esac
  fi

  # Check for system-dependent libraries X programs must link with.
  # Do this before checking for the system-independent R6 libraries
  # (-lICE), since we may need -lsocket or whatever for X linking.

  if test "$ISC" = yes; then
    X_EXTRA_LIBS="$X_EXTRA_LIBS -lnsl_s -linet"
  else
    # Martyn.Johnson@cl.cam.ac.uk says this is needed for Ultrix, if the X
    # libraries were built with DECnet support.  And karl@cs.umb.edu says
    # the Alpha needs dnet_stub (dnet does not exist).
    echo $ac_n "checking for dnet_ntoa in -ldnet""... $ac_c" 1>&2
echo "configure:0: checking for dnet_ntoa in -ldnet" >&3
ac_lib_var=`echo dnet\'_\'dnet_ntoa | sed \'y%./+-%__p_%\'`
if eval "test \"`echo \'$\'\'{\'ac_cv_lib_$ac_lib_var\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  ac_save_LIBS="$LIBS"
LIBS="-ldnet  $LIBS"
cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
/* Override any gcc2 internal prototype to avoid an error.  */
#ifdef __cplusplus
extern "C"
#endif
/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char dnet_ntoa();

int main() {
dnet_ntoa()
; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=yes"
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=no"
fi
rm -f conftest*
LIBS="$ac_save_LIBS"

fi
if eval "test \"`echo \'$ac_cv_lib_\'$ac_lib_var`\" = yes"; then
  echo "$ac_t""yes" 1>&2
  X_EXTRA_LIBS="$X_EXTRA_LIBS -ldnet"
else
  echo "$ac_t""no" 1>&2
fi

    if test $ac_cv_lib_dnet_dnet_ntoa = no; then
      echo $ac_n "checking for dnet_ntoa in -ldnet_stub""... $ac_c" 1>&2
echo "configure:0: checking for dnet_ntoa in -ldnet_stub" >&3
ac_lib_var=`echo dnet_stub\'_\'dnet_ntoa | sed \'y%./+-%__p_%\'`
if eval "test \"`echo \'$\'\'{\'ac_cv_lib_$ac_lib_var\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  ac_save_LIBS="$LIBS"
LIBS="-ldnet_stub  $LIBS"
cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
/* Override any gcc2 internal prototype to avoid an error.  */
#ifdef __cplusplus
extern "C"
#endif
/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char dnet_ntoa();

int main() {
dnet_ntoa()
; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=yes"
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=no"
fi
rm -f conftest*
LIBS="$ac_save_LIBS"

fi
if eval "test \"`echo \'$ac_cv_lib_\'$ac_lib_var`\" = yes"; then
  echo "$ac_t""yes" 1>&2
  X_EXTRA_LIBS="$X_EXTRA_LIBS -ldnet_stub"
else
  echo "$ac_t""no" 1>&2
fi

    fi

    # msh@cis.ufl.edu says -lnsl (and -lsocket) are needed for his 386/AT,
    # to get the SysV transport functions.
    # chad@anasazi.com says the Pyramis MIS-ES running DC/OSx (SVR4)
    # needs -lnsl.
    # The nsl library prevents programs from opening the X display
    # on Irix 5.2, according to dickey@clark.net.
    echo $ac_n "checking for gethostbyname""... $ac_c" 1>&2
echo "configure:0: checking for gethostbyname" >&3
if eval "test \"`echo \'$\'\'{\'ac_cv_func_gethostbyname\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
/* System header to define __stub macros and hopefully few prototypes,
    which can conflict with char gethostbyname(); below.  */
#include <assert.h>
/* Override any gcc2 internal prototype to avoid an error.  */
#ifdef __cplusplus
extern "C"
#endif
/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char gethostbyname();

int main() {

/* The GNU C library defines this for functions which it implements
    to always fail with ENOSYS.  Some functions are actually named
    something starting with __ and the normal name is an alias.  */
#if defined (__stub_gethostbyname) || defined (__stub___gethostbyname)
choke me
#else
gethostbyname();
#endif

; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  eval "ac_cv_func_gethostbyname=yes"
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  eval "ac_cv_func_gethostbyname=no"
fi
rm -f conftest*
fi

if eval "test \"`echo \'$ac_cv_func_\'gethostbyname`\" = yes"; then
  echo "$ac_t""yes" 1>&2
  :
else
  echo "$ac_t""no" 1>&2
fi

    if test $ac_cv_func_gethostbyname = no; then
      echo $ac_n "checking for gethostbyname in -lnsl""... $ac_c" 1>&2
echo "configure:0: checking for gethostbyname in -lnsl" >&3
ac_lib_var=`echo nsl\'_\'gethostbyname | sed \'y%./+-%__p_%\'`
if eval "test \"`echo \'$\'\'{\'ac_cv_lib_$ac_lib_var\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  ac_save_LIBS="$LIBS"
LIBS="-lnsl  $LIBS"
cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
/* Override any gcc2 internal prototype to avoid an error.  */
#ifdef __cplusplus
extern "C"
#endif
/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char gethostbyname();

int main() {
gethostbyname()
; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=yes"
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=no"
fi
rm -f conftest*
LIBS="$ac_save_LIBS"

fi
if eval "test \"`echo \'$ac_cv_lib_\'$ac_lib_var`\" = yes"; then
  echo "$ac_t""yes" 1>&2
  X_EXTRA_LIBS="$X_EXTRA_LIBS -lnsl"
else
  echo "$ac_t""no" 1>&2
fi

    fi

    # lieder@skyler.mavd.honeywell.com says without -lsocket,
    # socket/setsockopt and other routines are undefined under SCO ODT
    # 2.0.  But -lsocket is broken on IRIX 5.2 (and is not necessary
    # on later versions), says simon@lia.di.epfl.ch: it contains
    # gethostby* variants that do not use the nameserver (or something).
    # -lsocket must be given before -lnsl if both are needed.
    # We assume that if connect needs -lnsl, so does gethostbyname.
    echo $ac_n "checking for connect""... $ac_c" 1>&2
echo "configure:0: checking for connect" >&3
if eval "test \"`echo \'$\'\'{\'ac_cv_func_connect\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
/* System header to define __stub macros and hopefully few prototypes,
    which can conflict with char connect(); below.  */
#include <assert.h>
/* Override any gcc2 internal prototype to avoid an error.  */
#ifdef __cplusplus
extern "C"
#endif
/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char connect();

int main() {

/* The GNU C library defines this for functions which it implements
    to always fail with ENOSYS.  Some functions are actually named
    something starting with __ and the normal name is an alias.  */
#if defined (__stub_connect) || defined (__stub___connect)
choke me
#else
connect();
#endif

; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  eval "ac_cv_func_connect=yes"
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  eval "ac_cv_func_connect=no"
fi
rm -f conftest*
fi

if eval "test \"`echo \'$ac_cv_func_\'connect`\" = yes"; then
  echo "$ac_t""yes" 1>&2
  :
else
  echo "$ac_t""no" 1>&2
fi

    if test $ac_cv_func_connect = no; then
      echo $ac_n "checking for connect in -lsocket""... $ac_c" 1>&2
echo "configure:0: checking for connect in -lsocket" >&3
ac_lib_var=`echo socket\'_\'connect | sed \'y%./+-%__p_%\'`
if eval "test \"`echo \'$\'\'{\'ac_cv_lib_$ac_lib_var\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  ac_save_LIBS="$LIBS"
LIBS="-lsocket $X_EXTRA_LIBS $LIBS"
cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
/* Override any gcc2 internal prototype to avoid an error.  */
#ifdef __cplusplus
extern "C"
#endif
/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char connect();

int main() {
connect()
; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=yes"
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=no"
fi
rm -f conftest*
LIBS="$ac_save_LIBS"

fi
if eval "test \"`echo \'$ac_cv_lib_\'$ac_lib_var`\" = yes"; then
  echo "$ac_t""yes" 1>&2
  X_EXTRA_LIBS="-lsocket $X_EXTRA_LIBS"
else
  echo "$ac_t""no" 1>&2
fi

    fi

    # gomez@mi.uni-erlangen.de says -lposix is necessary on A/UX.
    echo $ac_n "checking for remove""... $ac_c" 1>&2
echo "configure:0: checking for remove" >&3
if eval "test \"`echo \'$\'\'{\'ac_cv_func_remove\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
/* System header to define __stub macros and hopefully few prototypes,
    which can conflict with char remove(); below.  */
#include <assert.h>
/* Override any gcc2 internal prototype to avoid an error.  */
#ifdef __cplusplus
extern "C"
#endif
/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char remove();

int main() {

/* The GNU C library defines this for functions which it implements
    to always fail with ENOSYS.  Some functions are actually named
    something starting with __ and the normal name is an alias.  */
#if defined (__stub_remove) || defined (__stub___remove)
choke me
#else
remove();
#endif

; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  eval "ac_cv_func_remove=yes"
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  eval "ac_cv_func_remove=no"
fi
rm -f conftest*
fi

if eval "test \"`echo \'$ac_cv_func_\'remove`\" = yes"; then
  echo "$ac_t""yes" 1>&2
  :
else
  echo "$ac_t""no" 1>&2
fi

    if test $ac_cv_func_remove = no; then
      echo $ac_n "checking for remove in -lposix""... $ac_c" 1>&2
echo "configure:0: checking for remove in -lposix" >&3
ac_lib_var=`echo posix\'_\'remove | sed \'y%./+-%__p_%\'`
if eval "test \"`echo \'$\'\'{\'ac_cv_lib_$ac_lib_var\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  ac_save_LIBS="$LIBS"
LIBS="-lposix  $LIBS"
cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
/* Override any gcc2 internal prototype to avoid an error.  */
#ifdef __cplusplus
extern "C"
#endif
/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char remove();

int main() {
remove()
; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=yes"
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=no"
fi
rm -f conftest*
LIBS="$ac_save_LIBS"

fi
if eval "test \"`echo \'$ac_cv_lib_\'$ac_lib_var`\" = yes"; then
  echo "$ac_t""yes" 1>&2
  X_EXTRA_LIBS="$X_EXTRA_LIBS -lposix"
else
  echo "$ac_t""no" 1>&2
fi

    fi

    # BSDI BSD/OS 2.1 needs -lipc for XOpenDisplay.
    echo $ac_n "checking for shmat""... $ac_c" 1>&2
echo "configure:0: checking for shmat" >&3
if eval "test \"`echo \'$\'\'{\'ac_cv_func_shmat\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
/* System header to define __stub macros and hopefully few prototypes,
    which can conflict with char shmat(); below.  */
#include <assert.h>
/* Override any gcc2 internal prototype to avoid an error.  */
#ifdef __cplusplus
extern "C"
#endif
/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char shmat();

int main() {

/* The GNU C library defines this for functions which it implements
    to always fail with ENOSYS.  Some functions are actually named
    something starting with __ and the normal name is an alias.  */
#if defined (__stub_shmat) || defined (__stub___shmat)
choke me
#else
shmat();
#endif

; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  eval "ac_cv_func_shmat=yes"
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  eval "ac_cv_func_shmat=no"
fi
rm -f conftest*
fi

if eval "test \"`echo \'$ac_cv_func_\'shmat`\" = yes"; then
  echo "$ac_t""yes" 1>&2
  :
else
  echo "$ac_t""no" 1>&2
fi

    if test $ac_cv_func_shmat = no; then
      echo $ac_n "checking for shmat in -lipc""... $ac_c" 1>&2
echo "configure:0: checking for shmat in -lipc" >&3
ac_lib_var=`echo ipc\'_\'shmat | sed \'y%./+-%__p_%\'`
if eval "test \"`echo \'$\'\'{\'ac_cv_lib_$ac_lib_var\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  ac_save_LIBS="$LIBS"
LIBS="-lipc  $LIBS"
cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
/* Override any gcc2 internal prototype to avoid an error.  */
#ifdef __cplusplus
extern "C"
#endif
/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char shmat();

int main() {
shmat()
; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=yes"
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=no"
fi
rm -f conftest*
LIBS="$ac_save_LIBS"

fi
if eval "test \"`echo \'$ac_cv_lib_\'$ac_lib_var`\" = yes"; then
  echo "$ac_t""yes" 1>&2
  X_EXTRA_LIBS="$X_EXTRA_LIBS -lipc"
else
  echo "$ac_t""no" 1>&2
fi

    fi
  fi

  # Check for libraries that X11R6 Xt/Xaw programs need.
  ac_save_LDFLAGS="$LDFLAGS"
  test -n "$x_libraries" && LDFLAGS="$LDFLAGS -L$x_libraries"
  # SM needs ICE to (dynamically) link under SunOS 4.x (so we have to
  # check for ICE first), but we must link in the order -lSM -lICE or
  # we get undefined symbols.  So assume we have SM if we have ICE.
  # These have to be linked with before -lX11, unlike the other
  # libraries we check for below, so use a different variable.
  #  --interran@uluru.Stanford.EDU, kb@cs.umb.edu.
  echo $ac_n "checking for IceConnectionNumber in -lICE""... $ac_c" 1>&2
echo "configure:0: checking for IceConnectionNumber in -lICE" >&3
ac_lib_var=`echo ICE\'_\'IceConnectionNumber | sed \'y%./+-%__p_%\'`
if eval "test \"`echo \'$\'\'{\'ac_cv_lib_$ac_lib_var\'+set}\'`\" = set"; then
  echo $ac_n "(cached) $ac_c" 1>&2
else
  ac_save_LIBS="$LIBS"
LIBS="-lICE $X_EXTRA_LIBS $LIBS"
cat > conftest.$ac_ext <<EOF
#line 0 "configure"
#include "confdefs.h"
/* Override any gcc2 internal prototype to avoid an error.  */
#ifdef __cplusplus
extern "C"
#endif
/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char IceConnectionNumber();

int main() {
IceConnectionNumber()
; return 0; }
EOF
if { (eval echo configure:0: \"$ac_link\") 1>&3; (eval $ac_link) 2>&3; } && test -s conftest${ac_exeext}; then
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=yes"
else
  echo "configure: failed program was:" >&3
  cat conftest.$ac_ext >&3
  rm -rf conftest*
  eval "ac_cv_lib_$ac_lib_var=no"
fi
rm -f conftest*
LIBS="$ac_save_LIBS"

fi
if eval "test \"`echo \'$ac_cv_lib_\'$ac_lib_var`\" = yes"; then
  echo "$ac_t""yes" 1>&2
  X_PRE_LIBS="$X_PRE_LIBS -lSM -lICE"
else
  echo "$ac_t""no" 1>&2
fi

  LDFLAGS="$ac_save_LDFLAGS"

fi
    ''', ['X_CFLAGS', 'X_LIBS', 'X_EXTRA_LIBS', 'X_PRE_LIBS'])
    return
