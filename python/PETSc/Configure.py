import config.base

import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.usingMPIUni              = 0
    self.missingPrototypes        = []
    self.missingPrototypesC       = []
    self.missingPrototypesCxx     = []
    self.missingPrototypesExternC = []
    self.defineAutoconfMacros()
    headersC = map(lambda name: name+'.h', ['dos', 'endian', 'fcntl', 'io', 'limits', 'malloc', 'pwd', 'search', 'strings',
                                            'stropts', 'unistd', 'machine/endian', 'sys/param', 'sys/procfs', 'sys/resource',
                                            'sys/stat', 'sys/systeminfo', 'sys/times', 'sys/utsname','string', 'stdlib',
                                            'sys/socket'])
    functions = ['access', '_access', 'clock', 'drand48', 'getcwd', '_getcwd', 'getdomainname', 'gethostname', 'getpwuid',
                 'gettimeofday', 'getrusage', 'getwd', 'memalign', 'memmove', 'mkstemp', 'popen', 'PXFGETARG', 'rand',
                 'readlink', 'realpath', 'sbreak', 'sigaction', 'signal', 'sigset', 'sleep', '_sleep', 'socket', 'times',
                 'uname','_snprintf']
    libraries = [('dl', 'dlopen'),(['socket','nsl'],'socket')]
    self.setCompilers = self.framework.require('config.setCompilers', self)
    self.compilers    = self.framework.require('config.compilers',    self)
    self.types        = self.framework.require('config.types',        self)
    self.headers      = self.framework.require('config.headers',      self)
    self.functions    = self.framework.require('config.functions',    self)
    self.libraries    = self.framework.require('config.libraries',    self)
    self.compilers.headerPrefix = self.headerPrefix
    self.types.headerPrefix     = self.headerPrefix
    self.headers.headerPrefix   = self.headerPrefix
    self.functions.headerPrefix = self.headerPrefix
    self.libraries.headerPrefix = self.headerPrefix
    self.headers.headers.extend(headersC)
    self.functions.functions.extend(functions)
    self.libraries.libraries.extend(libraries)
    # Check for packages
    import PETSc.packages

    for package in os.listdir(os.path.dirname(PETSc.packages.__file__)):
      (packageName, ext) = os.path.splitext(package)
      if ext == '.py' and not packageName == '__init__':
        packageObj              = self.framework.require('PETSc.packages.'+packageName, self)
        packageObj.headerPrefix = self.headerPrefix
        setattr(self, packageName.lower(), packageObj)
    # Put in dependencies
    self.framework.require('PETSc.packages.update',        self.setCompilers)
    self.framework.require('PETSc.packages.compilerFlags', self.compilers)
    self.framework.require('PETSc.packages.fortranstubs',  self.blaslapack)
    return

  def configureHelp(self, help):
    import nargs

    help.addArgument('PETSc', 'PETSC_DIR',                   nargs.Arg(None, None, 'The root directory of the PETSc installation'))
    help.addArgument('PETSc', 'PETSC_ARCH',                  nargs.Arg(None, None, 'The machine architecture'))
    help.addArgument('PETSc', '-enable-debug',               nargs.ArgBool(None, 1, 'Activate debugging code in PETSc'))
    help.addArgument('PETSc', '-enable-log',                 nargs.ArgBool(None, 1, 'Activate logging code in PETSc'))
    help.addArgument('PETSc', '-enable-stack',               nargs.ArgBool(None, 1, 'Activate manual stack tracing code in PETSc'))
    help.addArgument('PETSc', '-enable-dynamic',             nargs.ArgBool(None, 1, 'Build dynamic libraries for PETSc'))
    help.addArgument('PETSc', '-enable-etags',               nargs.ArgBool(None, 1, 'Build etags if they do not exist'))
    help.addArgument('PETSc', '-enable-fortran-kernels',     nargs.ArgBool(None, 0, 'Use Fortran for linear algebra kernels'))
    help.addArgument('PETSc', '-with-mpi',                   nargs.ArgBool(None, 1, 'If this is false, MPIUNI will be used as a uniprocessor substitute'))
    help.addArgument('PETSc', '-with-libtool',               nargs.ArgBool(None, 0, 'Specify that libtool should be used for compiling and linking'))
    help.addArgument('PETSc', '-with-make',                  nargs.Arg(None, 'make', 'Specify make'))
    help.addArgument('PETSc', '-with-ar',                    nargs.Arg(None, 'ar',   'Specify the archiver'))
    help.addArgument('PETSc', 'AR_FLAGS',                    nargs.Arg(None, 'cr',   'Specify the archiver flags'))
    help.addArgument('PETSc', '-with-ranlib',                nargs.Arg(None, None,   'Specify ranlib'))
    help.addArgument('PETSc', '-with-default-language=<c,c++,c++-complex,0(zero for no default)>', nargs.Arg(None, 'c', 'Specifiy default language of libraries'))
    help.addArgument('PETSc', '-with-default-optimization=<g,O,0(zero for no default)>',           nargs.Arg(None, 'g', 'Specifiy default optimization of libraries'))
    help.addArgument('PETSc', '-with-default-arch',          nargs.ArgBool(None, 1, 'Allow using the most recently configured arch without setting PETSC_ARCH'))
    return

  def defineAutoconfMacros(self):
    self.hostMacro = 'dnl Version: 2.13\ndnl Variable: host_cpu\ndnl Variable: host_vendor\ndnl Variable: host_os\nAC_CANONICAL_HOST'
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

  def configureFortranPIC(self):
    '''Determine the PIC option for the Fortran compiler'''
    # We use the framework in order to remove the PETSC_ namespace
    self.compilers.pushLanguage('F77')
    option = ''
    for opt in ['-PIC', '-fPIC', '-KPIC']:
      if self.compilers.checkCompilerFlag(opt):
        option = opt
        break
    self.framework.addSubstitution('FC_SHARED_OPT', option)
    self.compilers.popLanguage()
    return

  def configureFortranCPP(self):
    '''Handle case where Fortran cannot preprocess properly'''
    if 'FC' in self.framework.argDB:
      # IBM xlF chokes on this
      if not self.compilers.fortranPreprocess:
        if self.compilers.isGCC:
          traditional = 'TRADITIONAL_CPP = -traditional-cpp\n'
        else:
          traditional = 'TRADITIONAL_CPP = \n'
        self.framework.addSubstitution('F_to_o_TARGET', traditional+'include ${PETSC_DIR}/bmake/common/rules.fortran.nocpp')
      else:
        self.framework.addSubstitution('F_to_o_TARGET', 'include ${PETSC_DIR}/bmake/common/rules.fortran.cpp')
    else:
      self.framework.addSubstitution('F_to_o_TARGET', 'include ${PETSC_DIR}/bmake/common/rules.fortran.none')
    return


  def configureDynamicLibraries(self):
    '''Checks whether dynamic libraries should be used, for which you must
      - Specify --enable-dynamic
      - Find dlfcn.h and libdl
    Defines PETSC_USE_DYNAMIC_LIBRARIES is they are used
    Also checks that dlopen() takes RTLD_GLOBAL, and defines PETSC_HAVE_RTLD_GLOBAL if it does'''
    if not (self.framework.archBase.startswith('aix') or (self.framework.archBase.startswith('darwin') and not (self.usingMPIUni and not self.framework.argDB.has_key('FC')))):
      useDynamic = self.framework.argDB['enable-dynamic'] and self.headers.check('dlfcn.h') and self.libraries.haveLib('dl')
      self.addDefine('USE_DYNAMIC_LIBRARIES', useDynamic)
      if useDynamic and self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);\n'):
        self.addDefine('HAVE_RTLD_GLOBAL', 1)

    #  can only get dynamic shared libraries on Mac X with no g77 and no MPICH (maybe LAM?)
    if self.framework.archBase.startswith('darwin') and self.usingMPIUni and not self.framework.argDB.has_key('FC'):
      if self.framework.sharedBlasLapack: bls = 'BLASLAPACK_LIB_SHARED=${BLASLAPACK_LIB}\n'
      else:                               bls = ''
      self.framework.addSubstitution('DYNAMIC_SHARED_TARGET', bls+'MPI_LIB_SHARED=${MPI_LIB}\ninclude ${PETSC_DIR}/bmake/common/rules.shared.darwin7')
    else:
      self.framework.addSubstitution('DYNAMIC_SHARED_TARGET', 'include ${PETSC_DIR}/bmake/common/rules.shared.basic')

    # This is really bad
    flag = '-L'
    if self.framework.archBase == 'linux':
      flag = '-Wl,-rpath,'
    elif self.framework.archBase.startswith('irix'):
      flag = '-rpath '
    elif self.framework.archBase.startswith('osf'):
      flag = '-Wl,-rpath,'
    elif self.framework.archBase.startswith('freebsd'):
      flag = '-Wl,-R,'
    elif self.framework.archBase.startswith('solaris'):
      flag = '-R'
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
      self.framework.addSubstitution('SHARED_TARGET', 'shared_'+self.framework.archBase)
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
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.dbx+' -c conftest -p '+os.getpid(), log = self.framework.log)
          if not status:
            for line in output:
              if re.match(r'Process '+os.getpid()):
                self.addDefine('USE_P_FOR_DEBUGGER', 1)
                foundOption = 1
                break
        except RuntimeError: pass
      if not foundOption:
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.dbx+' -c conftest -a '+os.getpid(), log = self.framework.log)
          if not status:
            for line in output:
              if re.match(r'Process '+os.getpid()):
                self.addDefine('USE_A_FOR_DEBUGGER', 1)
                foundOption = 1
                break
        except RuntimeError: pass
      if not foundOption:
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.dbx+' -c conftest -pid '+os.getpid(), log = self.framework.log)
          if not status:
            for line in output:
              if re.match(r'Process '+os.getpid()):
                self.addDefine('USE_PID_FOR_DEBUGGER', 1)
                foundOption = 1
                break
        except RuntimeError: pass
      os.remove('conftest')
    elif hasattr(self, 'xdb'):
      self.addDefine('USE_XDB_DEBUGGER', 1)
      self.addDefine('USE_LARGEP_FOR_DEBUGGER', 1)
    return

  def configureMkdir(self):
    '''Make sure we can have mkdir automatically make intermediate directories'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.getExecutable('mkdir', getFullPath = 1)
    if hasattr(self.framework, 'mkdir'):
      self.mkdir = self.framework.mkdir
      if os.path.exists('.conftest'): os.rmdir('.conftest')
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(self.mkdir+' -p .conftest/.tmp', log = self.framework.log)
        if not status and os.path.isdir('.conftest/.tmp'):
          self.mkdir = self.mkdir+' -p'
          self.framework.addSubstitution('MKDIR', self.mkdir)
      except RuntimeError: pass
      if os.path.exists('.conftest'): os.removedirs('.conftest/.tmp')
    return

  def configureArchiver(self):
    '''Check the archiver'''
    self.framework.getExecutable(self.framework.argDB['with-ar'], getFullPath = 1, resultName = 'AR')
    self.framework.addArgumentSubstitution('AR_FLAGS', 'AR_FLAGS')
    return

  def configureRanlib(self):
    '''Check for ranlib, using "true" if it is not found'''
    if 'with-ranlib' in self.framework.argDB:
      found = self.framework.getExecutable(self.framework.argDB['with-ranlib'], resultName = 'RANLIB')
      if not found:
         raise RuntimeError('You set a value for --with-ranlib, but '+self.framework.argDB['with-ranlib']+' does not exist')
    else:
      found = self.framework.getExecutable('ranlib')
      if not found:
        self.framework.addSubstitution('RANLIB', 'true')
    return

  def configurePrograms(self):
    '''Check for the programs needed to build and run PETSc'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.getExecutable('sh',   getFullPath = 1, resultName = 'SHELL')
    self.framework.getExecutable('sed',  getFullPath = 1)
    self.framework.getExecutable('diff', getFullPath = 1)
    self.framework.getExecutable('ps',   path = '/usr/ucb:/usr/usb', resultName = 'UCBPS')
    if hasattr(self.framework, 'UCBPS'):
      self.addDefine('HAVE_UCBPS', 1)
    return

  def configureMissingFunctions(self):
    '''Checks for MISSING_GETPWUID and MISSING_SOCKETS'''
    if not self.functions.haveFunction('getpwuid'):
      self.addDefine('MISSING_GETPWUID', 1)
    if not self.functions.haveFunction('socket'):
      # solaris requires these two libraries for socket()
      if self.libraries.haveLib('socket') and self.libraries.haveLib('nsl'):
        self.addDefine('HAVE_SOCKET', 1)
        self.framework.argDB['LIBS'] += ' -lsocket -lnsl'
      else:
        self.addDefine('MISSING_SOCKETS', 1)
    return

  def configureMissingSignals(self):
    '''Check for missing signals, and define MISSING_<signal name> if necessary'''
    for signal in ['ABRT', 'ALRM', 'BUS',  'CHLD', 'CONT', 'FPE',  'HUP',  'ILL', 'INT',  'KILL', 'PIPE', 'QUIT', 'SEGV',
                   'STOP', 'SYS',  'TERM', 'TRAP', 'TSTP', 'URG',  'USR1', 'USR2']:
      if not self.checkCompile('#include <signal.h>\n', 'int i=SIG'+signal+';\n\nif (i);\n'):
        self.addDefine('MISSING_SIG'+signal, 1)
    return

  def configureMemorySize(self):
    '''Try to determine how to measure the memory usage'''
    # Should also check for using procfs and kbytes for size
    if self.functions.haveFunction('sbreak'):
      self.addDefine('USE_SBREAK_FOR_SIZE', 1)
    elif not (self.headers.haveHeader('sys/resource.h') and self.functions.haveFunction('getrusage')):
        self.addDefine('HAVE_NO_GETRUSAGE', 1)
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
    # If MS Windows's kernel32.lib is available then use lib for the suffix, otherwise use a.
    # Cygwin w32api uses libkernel32.a for this symbol.
    oldLibs = self.framework.argDB['LIBS']
    found = self.libraries.check(['kernel32.lib'],'GetCurrentProcess',prototype='int __stdcall GetCurrentProcess(void);\n')
    if found:
      suffix = 'lib'
    else:
      suffix = 'a'
    self.libraries.addSubstitution('LIB_SUFFIX', suffix)
    self.framework.argDB['LIBS'] = oldLibs
    return

  def configureGetDomainName(self):
    if not self.checkLink('#include <unistd.h>\n','char test[10]; int err = getdomainname(test,10);'):
      self.missingPrototypesC.append('int getdomainname(char *, int);')
    if 'CXX' in self.framework.argDB:
      self.pushLanguage('C++')
      if not self.checkLink('#include <unistd.h>\n','char test[10]; int err = getdomainname(test,10);'):
        self.missingPrototypesExternC.append('int getdomainname(char *, int);')
      self.popLanguage()  
    return
 
  def configureIRIX(self):
    '''IRIX specific stuff'''
    if self.framework.archBase.startswith('irix'):
      self.addDefine('USE_KBYTES_FOR_SIZE', 1)
    return

  def configureSolaris(self):
    '''Solaris specific stuff'''
    if self.framework.archBase.startswith('solaris'):
      if os.path.isdir(os.path.join('/usr','ucblib')):
        self.framework.argDB['LIBS'] += ' ${CLINKER_SLFLAG}/usr/ucblib'
    return

  def configureLinux(self):
    '''Linux specific stuff'''
    if self.framework.archBase == 'linux':
      self.addDefine('HAVE_DOUBLE_ALIGN_MALLOC', 1)
    return

  def configureWin32NonCygwin(self):
    '''Win32 non-cygwin specific stuff'''
    wfe = self.framework.argDB['CC'].split()[0]
    import os
    if os.path.splitext(os.path.basename(wfe))[0] == 'win32fe':
      self.framework.addDefine('PARCH_win32',1)
      self.framework.argDB['LIBS'] += ' kernel32.lib user32.lib  gdi32.lib advapi32.lib'
      self.addDefine('CANNOT_START_DEBUGGER',1)
      self.addDefine('USE_NT_TIME',1)
      self.missingPrototypes.append('typedef int uid_t;')
      self.missingPrototypes.append('typedef int gid_t;')
      self.missingPrototypes.append('#define R_OK 04')
      self.missingPrototypes.append('#define W_OK 02')
      self.missingPrototypes.append('#define X_OK 01')
      self.missingPrototypes.append('#define S_ISREG(a) (((a)&_S_IFMT) == _S_IFREG)')
      self.missingPrototypes.append('#define S_ISDIR(a) (((a)&_S_IFMT) == _S_IFDIR)')
    return
    
  def configureMPIUNI(self):
    '''If MPI was not found, setup MPIUNI, our uniprocessor version of MPI'''
    if self.framework.argDB['with-mpi']:
      if self.mpi.foundMPI:
        return
      else:
        raise RuntimeError('********** Error: Unable to locate a functional MPI. Please consult configure.log. **********')
    self.framework.addDefine('HAVE_MPI', 1)
    self.framework.addSubstitution('MPI_INCLUDE', '-I'+'${PETSC_DIR}/src/sys/src/mpiuni')
    self.framework.addSubstitution('MPI_LIB',     '-L${PETSC_DIR}/lib/lib${BOPT}/${PETSC_ARCH} -lmpiuni')
    self.framework.addSubstitution('MPIRUN',      '${PETSC_DIR}/src/sys/src/mpiuni/mpirun')
    self.framework.addSubstitution('MPE_INCLUDE', '')
    self.framework.addSubstitution('MPE_LIB',     '')
    self.mpi.addDefine('HAVE_MPI_COMM_F2C', 1)
    self.mpi.addDefine('HAVE_MPI_COMM_C2F', 1)
    self.mpi.addDefine('HAVE_MPI_FINT', 1)
    self.usingMPIUni = 1
    return

  def configureMissingPrototypes(self):
    '''Checks for missing prototypes, which it adds to petscfix.h'''
    if not 'HAVE_MPI_FINT' in self.mpi.defines:
      self.missingPrototypes.append('typedef int MPI_Fint;')
    if not 'HAVE_MPI_COMM_F2C' in self.mpi.defines:
      self.missingPrototypes.append('#define MPI_Comm_f2c(a) (a)')
    if not 'HAVE_MPI_COMM_C2F' in self.mpi.defines:
      self.missingPrototypes.append('#define MPI_Comm_c2f(a) (a)')
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.addSubstitution('MISSING_PROTOTYPES',     '\n'.join(self.missingPrototypes))
    self.framework.addSubstitution('MISSING_PROTOTYPES_C', '\n'.join(self.missingPrototypesC))
    self.framework.addSubstitution('MISSING_PROTOTYPES_CXX', '\n'.join(self.missingPrototypesCxx))
    self.framework.addSubstitution('MISSING_PROTOTYPES_EXTERN_C', '\n'.join(self.missingPrototypesExternC))
    return

  def configureMachineInfo(self):
    '''Define a string incorporating all configuration data needed for a bug report'''
    self.addDefine('MACHINE_INFO', '"Libraries compiled on `date` on `hostname`\\nMachine characteristics: `uname -a`\\n-----------------------------------------\\nUsing C compiler: ${CC} ${COPTFLAGS} ${CCPPFLAGS}\\nC Compiler version: ${C_VERSION}\\nUsing C compiler: ${CXX} ${CXXOPTFLAGS} ${CXXCPPFLAGS}\\nC++ Compiler version: ${CXX_VERSION}\\nUsing Fortran compiler: ${FC} ${FOPTFLAGS} ${FCPPFLAGS}\\nFortran Compiler version: ${F_VERSION}\\n-----------------------------------------\\nUsing PETSc flags: ${PETSCFLAGS} ${PCONF}\\n-----------------------------------------\\nUsing include paths: ${PETSC_INCLUDE}\\n-----------------------------------------\\nUsing PETSc directory: ${PETSC_DIR}\\nUsing PETSc arch: ${PETSC_ARCH}"\\n')
    return

  def configureMisc(self):
    '''Fix up all the things that we currently need to run'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.addSubstitution('CC_SHARED_OPT', '')

    # if BOPT is not set determines what libraries to use
    bopt = self.framework.argDB['with-default-optimization']
    if self.framework.argDB['with-default-language'] == '0' or self.framework.argDB['with-default-optimization'] == '0':
      fd = open('bmake/common/bopt_','w')
      fd.write('PETSC_LANGUAGE  = CONLY\nPETSC_SCALAR    = real\nPETSC_PRECISION = double\n')
      fd.close()
    elif not ((bopt == 'O') or (bopt == 'g')):
      raise RuntimeError('Unknown option given with --with-default-optimization='+self.framework.argDB['with-default-optimization'])
    else:
      if self.framework.argDB['with-default-language'] == 'c': pass
      elif self.framework.argDB['with-default-language'] == 'c++': bopt += '_c++'
      elif self.framework.argDB['with-default-language'].find('complex') >= 0: bopt += '_complex'
      else:
        raise RuntimeError('Unknown option given with --with-default-language='+self.framework.argDB['with-default-language'])
      fd = open(os.path.join('bmake','common','bopt_'),'w')
      fd.write('BOPT='+bopt+'\n')
      fd.write('include ${PETSC_DIR}/bmake/common/bopt_'+bopt+'\n')
      fd.close()

    # if PETSC_ARCH is not set use one last created with configure
    if self.framework.argDB['with-default-arch']:
      fd = open(os.path.join('bmake','variables'),'w')
      fd.write('PETSC_ARCH='+self.framework.arch+'\n')
      fd.write('include ${PETSC_DIR}/bmake/'+self.framework.arch+'/variables\n')
      fd.close()
    else:
      os.unlink(os.path.join('bmake','variables'))

    return

  def configureETags(self):
    '''Determine if etags files exist and try to create otherwise'''
    if not os.path.exists(os.path.join(self.framework.argDB['PETSC_DIR'], 'TAGS')):
      self.framework.log.write('WARNING: ETags files have not been created\n')
      self.framework.getExecutable('etags', getFullPath = 1)
      if hasattr(self.framework, 'etags'):
        pd = self.framework.argDB['PETSC_DIR']
        self.framework.log.write('           Running '+self.framework.etags+' to generate TAGS files\n')
        try:
          (output, error, status) = config.base.Configure.executeShellCommand('export PETSC_ARCH=linux;make PETSC_DIR='+pd+' TAGSDIR='+pd+' etags', timeout = 15*60.0, log = self.framework.log)
          # filter out the normal messages
          cnt = 0
          for i in output.split('\n'):
            if not (i.startswith('etags_') or i.find('TAGS') >= 0):
              if not cnt:
                self.framework.log.write('*******Error generating etags files****\n')
              cnt = cnt + 1
              self.framework.log.write(i+'\n')
          if not cnt:
            self.framework.log.write('           Completed generating etags files\n')
          else:
            self.framework.log.write('*******End of error messages from generating etags files*******\n')
        except RuntimeError, e:
          self.framework.log.write('*******Error generating etags files: '+str(e)+'*******\n')
      else:
        self.framework.log.write('           The etags command is not in your path, cannot build etags files\n')
    else:
      self.framework.log.write('Found etags file \n')
    return

  def configureDocs(self):
    '''Determine if the docs are built, if not, warn the user'''
    if not os.path.exists(os.path.join(self.framework.argDB['PETSC_DIR'], 'include','petscvec.h.html')):
      self.framework.log.write('WARNING: document files have not been created\n')
      self.framework.getExecutable('doctext', getFullPath = 1)
      self.framework.getExecutable('mapnames', getFullPath = 1)
      self.framework.getExecutable('c2html', getFullPath = 1)
      self.framework.getExecutable('pdflatex', getFullPath = 1)
      if hasattr(self.framework, 'doctext') and hasattr(self.framework, 'mapnames') and hasattr(self.framework, 'c2html') and hasattr(self.framework, 'pdflatex'):
        self.framework.log.write('           You can run "make alldoc LOC=${PETSC_DIR}" to generate all the documentation\n')
        self.framework.log.write('           WARNING!!! This will take several HOURS to run\n')
      else:
        self.framework.log.write('           You are missing')
        if not hasattr(self.framework, 'doctext'): self.framework.log.write(' doctext')
        if not hasattr(self.framework, 'mapnames'):self.framework.log.write(' mapnames')
        if not hasattr(self.framework, 'c2html'):  self.framework.log.write(' c2html')
        if not hasattr(self.framework, 'pdflatex'):self.framework.log.write(' pdflatex')
        self.framework.log.write('\n')
        self.framework.log.write('           from your PATH. See http:/www.mcs.anl.gov/petsc/petsc-2/developers for how\n')
        self.framework.log.write('           install them and compile the documentation\n')
        self.framework.log.write('      Or view the docs on line at http://www.mcs.anl.gov/petsc/petsc-2/snapshots/petsc-dev/docs\n')
    else:
      self.framework.log.write('Document files found\n')
    return

  def configureRegression(self):
    '''Output a file listing the jobs that should be run by the PETSc buildtest'''
    jobs = []
    if self.usingMPIUni:
      jobs.append('4')
      if 'FC' in self.framework.argDB:
        jobs.append('9')
    else:
      jobs.append('1')
      if self.x11.foundX11:
        jobs.append('2')
      if 'FC' in self.framework.argDB:
        jobs.append('3')
    jobsFile  = file(os.path.abspath(os.path.join(self.bmakeDir, 'jobs')), 'w')
    jobsFile.write(' '.join(jobs)+'\n')
    jobsFile.close()
    ejobsFile = file(os.path.abspath(os.path.join(self.bmakeDir, 'ejobs')), 'w')
    ejobsFile.write(' ')
    ejobsFile.close()
    return

  def configureScript(self):
    '''Output a script in the bmake directory which will reproduce the configuration'''
    scriptName = os.path.join(self.bmakeDir, 'configure.py')
    f = file(scriptName, 'w')
    f.write('#!/usr/bin/env python\n')
    f.write('if __name__ == \'__main__\':\n')
    f.write('  import sys\n')
    f.write('  sys.path.insert(0, '+repr(os.path.join(self.framework.argDB['PETSC_DIR'], 'config'))+')\n')
    f.write('  import configure\n')
    f.write('  configure_options = '+repr(self.framework.clArgs)+'\n')
    f.write('  configure.petsc_configure(configure_options)\n')
    f.close()
    os.chmod(scriptName, 0775)
    return

  def configure(self):
    self.framework.header = 'bmake/'+self.framework.arch+'/petscconf.h'
    self.framework.addSubstitutionFile('bmake/config/packages.in',   'bmake/'+self.framework.arch+'/packages')
    self.framework.addSubstitutionFile('bmake/config/rules.in',      'bmake/'+self.framework.arch+'/rules')
    self.framework.addSubstitutionFile('bmake/config/variables.in',  'bmake/'+self.framework.arch+'/variables')
    self.framework.addSubstitutionFile('bmake/config/petscfix.h.in', 'bmake/'+self.framework.arch+'/petscfix.h')
    self.executeTest(self.configureLibraryOptions)
    if 'FC' in self.framework.argDB:
      self.executeTest(self.configureFortranPIC)
    else:
      self.framework.addSubstitution('FC_SHARED_OPT', '')
    self.executeTest(self.configureFortranCPP)
    self.executeTest(self.configureMPIUNI)
    self.executeTest(self.configureDynamicLibraries)
    self.executeTest(self.configureLibtool)
    self.executeTest(self.configureDebuggers)
    self.executeTest(self.configureMkdir)
    self.executeTest(self.configureArchiver)
    self.executeTest(self.configureRanlib)
    self.executeTest(self.configurePrograms)
    self.executeTest(self.configureMissingFunctions)
    self.executeTest(self.configureMissingSignals)
    self.executeTest(self.configureMemorySize)
    self.executeTest(self.configureFPTrap)
    self.executeTest(self.configureLibrarySuffix)
    self.executeTest(self.configureGetDomainName)
    self.executeTest(self.configureIRIX)
    self.executeTest(self.configureSolaris)
    self.executeTest(self.configureLinux)
    self.executeTest(self.configureWin32NonCygwin)
    self.executeTest(self.configureMissingPrototypes)
    self.executeTest(self.configureMachineInfo)
    self.executeTest(self.configureMisc)
    if self.framework.argDB['enable-etags']:                                    
      self.executeTest(self.configureETags)
    self.executeTest(self.configureDocs)
    self.bmakeDir = os.path.join('bmake', self.framework.argDB['PETSC_ARCH'])
    if not os.path.exists(self.bmakeDir):
      os.makedirs(self.bmakeDir)
    self.executeTest(self.configureRegression)
    self.executeTest(self.configureScript)
    self.startLine()
    return
