import configure

import commands
import os
import re

class Configure(configure.Configure):
  def __init__(self, framework):
    configure.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.defineAutoconfMacros()
    headersC = map(lambda name: name+'.h', ['dos', 'endian', 'fcntl', 'io', 'limits', 'malloc', 'pwd', 'search',
                                            'strings', 'stropts', 'unistd', 'machine/endian', 'sys/param',
                                            'sys/procfs', 'sys/resource', 'sys/stat', 'sys/systeminfo', 'sys/time',
                                            'sys/times', 'sys/utsname'])
    functions = ['access', '_access', 'clock', 'drand48', 'getcwd', '_getcwd', 'getdomainname', 'gethostname', 'getpwuid',
                 'gettimeofday', 'getwd', 'memalign', 'memmove', 'mkstemp', 'popen', 'PXFGETARG', 'rand',
                 'realpath', 'sigaction', 'signal', 'sigset', 'sleep', '_sleep', 'socket', 'times', 'uname']
    libraries = [('dl', 'dlopen')]
    framework.checkTypes()
    framework.checkHeaders(headersC)
    framework.checkFunctions(functions)
    framework.checkLibraries(libraries)
    framework.checkCompilers()
    return

  def defineAutoconfMacros(self):
    self.hostMacro = 'dnl Version: 2.13\ndnl Variable: host_cpu\ndnl Variable: host_vendor\ndnl Variable: host_os\nAC_CANONICAL_HOST'
    self.xMacro    = 'dnl Version: 2.13\ndnl Variable: X_CFLAGS\ndnl Variable: X_LIBS\ndnl Variable: X_EXTRA_LIBS\ndnl Variable: X_PRE_LIBS\nAC_PATH_XTRA'
    return

  def configureDirectories(self):
    '''Sets PETSC_DIR'''
    if self.framework.argDB.has_key('PETSC_DIR'):
      self.dir = self.framework.argDB['PETSC_DIR']
    else:
      self.dir = os.getcwd()
    self.addSubstitution('DIR', self.dir)
    self.addDefine('DIR', self.dir, 'The root directory of the Petsc installation')
    return

  def configureArchitecture(self):
    '''Sets PETSC_ARCH'''
    results = self.executeShellCode(self.macroToShell(self.hostMacro))
    self.host_cpu    = results['host_cpu']
    self.host_vendor = results['host_vendor']
    self.host_os     = results['host_os']
    if not self.framework.argDB.has_key('PETSC_ARCH'):
      self.arch = results['host_os']
    else:
      self.arch = self.framework.argDB['PETSC_ARCH']
    if not self.arch.startswith(results['host_os']):
      raise RuntimeError('PETSC_ARCH ('+self.arch+') does not have our guess ('+results['host_os']+') as a prefix!')
    self.addSubstitution('ARCH', self.arch)
    self.archBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.arch)
    self.addDefine('ARCH', self.archBase, 'The primary architecture of this machine')
    return

  def configureLibraryOptions(self):
    '''Sets bopt, PETSC_USE_DEBUG, PETSC_USE_LOG, and PETSC_USE_STACK'''
    self.getArgument('bopt', 'g', '-with-')
    self.getArgument('useDebug', 1, '-enable-', int, comment = 'Debugging flag')
    self.addDefine('USE_DEBUG', self.useDebug)
    self.getArgument('useLog',   1, '-enable-', int, comment = 'Logging flag')
    self.addDefine('USE_LOG',   self.useLog)
    self.getArgument('useStack', 1, '-enable-', int, comment = 'Stack tracing flag')
    self.addDefine('USE_STACK', self.useStack)
    return

  def getCompilerFlags(self, langName, compiler):
    optionsCmd = self.shell+' '+os.path.join(self.configAuxDir, 'config.options')+' '+self.host_cpu+'-'+self.host_vendor+'-'+self.host_os+' '+compiler
    # Compiler version
    (status, output) = commands.getstatusoutput(optionsCmd+' v')
    if not status:
      (status, output) = commands.getstatusoutput(compiler+' '+output)
    if not status:
      setattr(self, langName+'Version', output.split('\n')[0])
    else:
      setattr(self, langName+'Version', 'Unknown')
    # Compiler debug options
    var = langName+'FLAGS_g'
    if not self.framework.argDB.has_key('PETSC_'+var):
      (status, output) = commands.getstatusoutput(optionsCmd+' g')
      if not status:
        setattr(self, var, output)
      else:
        raise RuntimeError('Cannot guess debug options: '+output)
    else:
      setattr(self, var, self.framework.argDB['PETSC_'+var])
    # Compiler optimized options
    var = langName+'FLAGS_O'
    if not self.framework.argDB.has_key('PETSC_'+var):
      (status, output) = commands.getstatusoutput(optionsCmd+' O')
      if not status:
        setattr(self, var, output)
      else:
        raise RuntimeError('Cannot guess optimized options')
    else:
      setattr(self, var, self.framework.argDB['PETSC_'+var])
    return

  def configureCompilerFlags(self):
    optionsCmd = os.path.join(self.configAuxDir, 'config.options')+' '+self.host_cpu+'-'+self.host_vendor+'-'+self.host_os
    self.getCompilerFlags('C',   self.framework.compilers.CC)
    self.getCompilerFlags('CXX', self.framework.compilers.CXX)
    self.getCompilerFlags('F',   self.framework.compilers.FC)

    self.addSubstitution('C_VERSION',   self.CVersion)
    self.addSubstitution('CFLAGS_g',    self.CFLAGS_g)
    self.addSubstitution('CFLAGS_O',    self.CFLAGS_O)
    self.addSubstitution('CXX_VERSION', self.CXXVersion)
    self.addSubstitution('CXXFLAGS_g',  self.CXXFLAGS_g)
    self.addSubstitution('CXXFLAGS_O',  self.CXXFLAGS_O)
    self.addSubstitution('F_VERSION',   self.FVersion)
    self.addSubstitution('FFLAGS_g',    self.FFLAGS_g)
    self.addSubstitution('FFLAGS_O',    self.FFLAGS_O)

    self.addSubstitution('PETSCFLAGS', self.framework.argDB['PETSCFLAGS'])
    self.addSubstitution('COPTFLAGS',  self.framework.argDB['COPTFLAGS'])
    self.addSubstitution('FOPTFLAGS',  self.framework.argDB['FOPTFLAGS'])
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
    option = ''
    if self.checkF77CompilerOption('-PIC'):
      option = '-PIC'
    elif self.checkF77CompilerOption('-fPIC'):
      option = '-fPIC'
    elif self.checkF77CompilerOption('-KPIC'):
      option = '-KPIC'
    self.addSubstitution('FC_SHARED_OPT', option)
    return

  def configureDebuggers(self):
    self.getExecutable('gdb', getFullPath = 1, comment = 'GNU debugger')
    self.getExecutable('dbx', getFullPath = 1, comment = 'DBX debugger')
    self.getExecutable('xdb', getFullPath = 1, comment = 'XDB debugger')
    if hasattr(self, 'gdb'):
      self.addDefine('USE_GDB_DEBUGGER', 1, comment = 'Use GDB as the default debugger')
    elif hasattr(self, 'dbx'):
      self.addDefine('USE_DBX_DEBUGGER', 1, comment = 'Use DBX as the default debugger')
      f = file('conftest', 'w')
      f.write('quit\n')
      f.close()
      foundOption = 0
      if not foundOption:
        (status, output) = commands.getstatusoutput(self.dbx+' -c conftest -p '+os.getpid())
        for line in output:
          if re.match(r'Process '+os.getpid()):
            self.addDefine('USE_P_FOR_DEBUGGER', 1, comment = 'Use -p to indicate a process to the debugger')
            foundOption = 1
            break
      if not foundOption:
        (status, output) = commands.getstatusoutput(self.dbx+' -c conftest -a '+os.getpid())
        for line in output:
          if re.match(r'Process '+os.getpid()):
            self.addDefine('USE_A_FOR_DEBUGGER', 1, comment = 'Use -a to indicate a process to the debugger')
            foundOption = 1
            break
      if not foundOption:
        (status, output) = commands.getstatusoutput(self.dbx+' -c conftest -pid '+os.getpid())
        for line in output:
          if re.match(r'Process '+os.getpid()):
            self.addDefine('USE_PID_FOR_DEBUGGER', 1, comment = 'Use -pid to indicate a process to the debugger')
            foundOption = 1
            break
      os.remove('conftest')
    elif hasattr(self, 'xdb'):
      self.addDefine('USE_XDB_DEBUGGER', 1, comment = 'Use XDB as the default debugger')
      self.addDefine('USE_LARGEP_FOR_DEBUGGER', 1, comment = 'Use -P to indicate a process to the debugger')
    return

  def checkMkdir(self):
    self.getExecutable('mkdir', getFullPath = 1, comment = 'Mkdir utility')
    if hasattr(self, 'mkdir'):
      if os.path.exists('.conftest'): os.rmdir('.conftest')
      (status, output) = commands.getstatusoutput(self.mkdir+' -p .conftest/.tmp')
      if not status and os.path.isdir('.conftest/.tmp'):
        self.mkdir = self.mkdir+' -p'
        self.addSubstitution('MKDIR', self.mkdir)
      if os.path.exists('.conftest'): os.removedirs('.conftest/.tmp')
    return

  def configurePrograms(self):
    self.checkMkdir()
    self.getExecutable('sh',   getFullPath = 1, comment = 'Bourne shell', substName = 'SHELL')
    self.getExecutable('sed',  getFullPath = 1, comment = 'Sed utility')
    self.getExecutable('diff', getFullPath = 1, comment = 'Diff utility')
    self.getExecutable('ar',   getFullPath = 1, comment = 'Archive utility')
    self.getExecutable('make', comment = 'Build utility')
    self.addSubstitution('AR_FLAGS', 'cr')
    self.getExecutable('ranlib', comment = 'Ranlib utility')
    self.addSubstitution('SET_MAKE', '', comment = 'Obsolete')
    self.addSubstitution('LIBTOOL', '${SHELL} ${top_builddir}/libtool')
    return

  def configureMissingPrototypes(self):
    '''Checks for missing prototypes, which it adds to petscfix.h'''
    self.addSubstitution('MISSING_PROTOTYPES',     '', comment = 'C compiler')
    self.addSubstitution('MISSING_PROTOTYPES_CXX', '', comment = 'C compiler')
    self.missingPrototypesExternC = ''
    if self.archBase == 'linux':
      self.missingPrototypesExternC += 'extern void *memalign(int, int);'
    self.addSubstitution('MISSING_PROTOTYPES_EXTERN_C', self.missingPrototypesExternC, comment = 'C compiler')
    return

  def configureMissingFunctions(self):
    '''Checks for MISSING_GETPWUID and MISSING_SOCKETS'''
    if not self.framework.functions.defines.has_key(self.framework.functions.getDefineName('getpwuid')):
      self.addDefine('MISSING_GETPWUID', 1)
    if not self.framework.functions.defines.has_key(self.framework.functions.getDefineName('socket')):
      self.addDefine('MISSING_SOCKETS', 1)
    return

  def configureMissingSignals(self):
    '''Check for missing signals, and define MISSING_<signal name> if necessary'''
    if self.checkCompile('#include <signal.h>\n', 'int i=SIGSYS;\n'):
      self.addDefine('MISSING_SIGSYS', 1)
    if self.checkCompile('#include <signal.h>\n', 'int i=SIGBUS;\n'):
      self.addDefine('MISSING_SIGBUS', 1)
    if self.checkCompile('#include <signal.h>\n', 'int i=SIGQUIT;\n'):
      self.addDefine('MISSING_SIGQUIT', 1)
    return

  def configureX(self):
    '''Uses AC_PATH_XTRA, and sets PETSC_HAVE_X11'''
    if not self.framework.argDB.has_key('no_x'):
      os.environ['with_x']      = 'yes'
      os.environ['x_includes']  = 'NONE'
      os.environ['x_libraries'] = 'NONE'
      results = self.executeShellCode(self.macroToShell(self.xMacro))
      # LIBS="$LIBS $X_PRE_LIBS $X_LIBS $X_EXTRA_LIBS"
      self.addDefine('HAVE_X11', 1)
      self.addSubstitution('X_CFLAGS',     results['X_CFLAGS'])
      self.addSubstitution('X_PRE_LIBS',   results['X_PRE_LIBS'])
      self.addSubstitution('X_LIBS',       results['X_LIBS']+' -lX11')
      self.addSubstitution('X_EXTRA_LIBS', results['X_EXTRA_LIBS'])
    return

  def configureFPTrap(self):
    if self.framework.headers.check('sigfpe.h'):
      if self.framework.functions.check(handle_sigfpes, library = 'fpe'):
        self.addDefine('HAVE_IRIX_STYLE_FPTRAP', 1)
    elif self.framework.headers.check('fpxcp.h') and self.framework.headers.check('fptrap.h'):
      if reduce(lambda x,y: x and y, map(self.framework.functions.check, ['fp_sh_trap_info', 'fp_trap', 'fp_enable', 'fp_disable'])):
        self.addDefine('HAVE_RS6000_STYLE_FPTRAP', 1)
    elif self.framework.headers.check('floatingpoint.h'):
      if self.framework.functions.check('ieee_flags') and self.framework.functions.check('ieee_handler'):
        if self.framework.headers.check('sunmath.h'):
          self.addDefine('HAVE_SOLARIS_STYLE_FPTRAP', 1)
        else:
          self.addDefine('HAVE_SUN4_STYLE_FPTRAP', 1)
    return

  def configureIRIX(self):
    if self.archBase == 'irix6.5':
      self.addDefine('USE_KBYTES_FOR_SIZE', 1)
    return

  def configureLinux(self):
    if self.archBase == 'linux':
      self.addDefine('HAVE_DOUBLE_ALIGN_MALLOC', 1)
    return

  def configureMachineInfo(self):
    self.addDefine('PETSC_MACHINE_INFO', '''
    "Libraries compiled on `date` on `hostname`
    Machine characteristics: `uname -a`
    -----------------------------------------
    Using C compiler: ${CC} ${COPTFLAGS} ${CCPPFLAGS}
    C Compiler version: ${C_VERSION}
    Using C compiler: ${CXX} ${CXXOPTFLAGS} ${CXXCPPFLAGS}
    C++ Compiler version: ${CXX_VERSION}
    Using Fortran compiler: ${FC} ${FOPTFLAGS} ${FCPPFLAGS}
    Fortran Compiler version: ${F_VERSION}
    -----------------------------------------
    Using PETSc flags: ${PETSCFLAGS} ${PCONF}
    -----------------------------------------
    Using include paths: ${PETSC_INCLUDE}
    -----------------------------------------
    Using PETSc directory: ${PETSC_DIR}
    Using PETSc arch: ${PETSC_ARCH}"
    ''')
    return

  def mpiuniWarning(self):
##    if self.mpiuni:
##      print '********** Warning: Using uniprocessor MPI (mpiuni) from Petsc **********'
##      print '**********     Use --with-mpi option to specify a full MPI     **********'
    return

  def configureMisc(self):
    '''Fix up all the things that we currently need to run'''
    self.addSubstitution('LT_CC', '${PETSC_LIBTOOL} ${LIBTOOL} --mode=compile')
    self.addSubstitution('CC_SHARED_OPT', '')
    self.addSubstitution('LDFLAGS', self.framework.argDB['LDFLAGS'])
    self.addSubstitution('LIBS',    self.framework.argDB['LIBS'])
    return

  def configure(self):
    self.configureDirectories()
    self.configureArchitecture()
    self.framework.header = 'bmake/'+self.arch+'-test/petscconfig.h'
    self.framework.addSubstitutionFile('bmake/config-test/packages.in',   'bmake/'+self.arch+'-test/packages')
    self.framework.addSubstitutionFile('bmake/config-test/rules.in',      'bmake/'+self.arch+'-test/rules')
    self.framework.addSubstitutionFile('bmake/config-test/variables.in',  'bmake/'+self.arch+'-test/variables')
    self.framework.addSubstitutionFile('bmake/config-test/petscfix.h.in', 'bmake/'+self.arch+'-test/petscfix.h')
    self.configureLibraryOptions()
    self.configureCompilerFlags()
    self.configureFortranPIC()
    self.configureDebuggers()
    self.configurePrograms()
    self.configureMissingPrototypes()
    self.configureMissingFunctions()
    self.configureMissingSignals()
    self.configureX()
    self.configureFPTrap()
    self.configureIRIX()
    self.configureLinux()
    self.configureMachineInfo()
    self.configureMisc()
    self.mpiuniWarning()
    return
