import config.base

import commands
import os
import os.path
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.usingMPIUni  = 0
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
    self.compilers   = self.framework.require('config.compilers', self)
    self.update      = self.framework.require('PETSc.packages.update', self.compilers)
    self.types       = self.framework.require('config.types',     self)
    self.headers     = self.framework.require('config.headers',   self)
    self.functions   = self.framework.require('config.functions', self)
    self.libraries   = self.framework.require('config.libraries', self)
    self.blaslapack  = self.framework.require('PETSc.packages.BlasLapack',  self)
    self.mpi         = self.framework.require('PETSc.packages.MPI',         self)
    self.mpe         = self.framework.require('PETSc.packages.MPE',         self)
    self.adic        = self.framework.require('PETSc.packages.ADIC',        self)
    self.matlab      = self.framework.require('PETSc.packages.Matlab',      self)
    self.mathematica = self.framework.require('PETSc.packages.Mathematica', self)
    self.triangle    = self.framework.require('PETSc.packages.Triangle',    self)
    self.parmetis    = self.framework.require('PETSc.packages.ParMetis',    self)
    self.plapack     = self.framework.require('PETSc.packages.PLAPACK',     self)
    self.pvode       = self.framework.require('PETSc.packages.PVODE',       self)
    self.blocksolve  = self.framework.require('PETSc.packages.BlockSolve',  self)
    self.netcdf      = self.framework.require('PETSc.packages.NetCDF',      self)
    self.headers.headers.extend(headersC)
    self.functions.functions.extend(functions)
    self.libraries.libraries.extend(libraries)
    # Put all defines in the PETSc namespace
    self.compilers.headerPrefix   = self.headerPrefix
    self.types.headerPrefix       = self.headerPrefix
    self.headers.headerPrefix     = self.headerPrefix
    self.functions.headerPrefix   = self.headerPrefix
    self.libraries.headerPrefix   = self.headerPrefix
    self.blaslapack.headerPrefix  = self.headerPrefix
    self.mpi.headerPrefix         = self.headerPrefix
    self.mpe.headerPrefix         = self.headerPrefix
    self.adic.headerPrefix        = self.headerPrefix
    self.matlab.headerPrefix      = self.headerPrefix
    self.mathematica.headerPrefix = self.headerPrefix
    self.triangle.headerPrefix    = self.headerPrefix
    self.parmetis.headerPrefix    = self.headerPrefix
    self.plapack.headerPrefix     = self.headerPrefix
    self.pvode.headerPrefix       = self.headerPrefix
    self.blocksolve.headerPrefix  = self.headerPrefix
    self.netcdf.headerPrefix      = self.headerPrefix
    return


  def configureHelp(self, help):
    import nargs

    help.addArgument('PETSc', 'PETSC_DIR',                   nargs.Arg(None, None, 'The root directory of the PETSc installation'))
    help.addArgument('PETSc', 'PETSC_ARCH',                  nargs.Arg(None, None, 'The machine architecture'))
    help.addArgument('PETSc', '-enable-debug',               nargs.ArgBool(None, 1, 'Activate debugging code in PETSc'))
    help.addArgument('PETSc', '-enable-log',                 nargs.ArgBool(None, 1, 'Activate logging code in PETSc'))
    help.addArgument('PETSc', '-enable-stack',               nargs.ArgBool(None, 1, 'Activate manual stack tracing code in PETSc'))
    help.addArgument('PETSc', '-enable-dynamic',             nargs.ArgBool(None, 1, 'Build dynamic libraries for PETSc'))
    help.addArgument('PETSc', '-enable-fortran-kernels',     nargs.ArgBool(None, 0, 'Use Fortran for linear algebra kernels'))
    help.addArgument('PETSc', 'optionsModule=<module name>', nargs.Arg(None, None, 'The Python module used to determine compiler options and versions'))
    help.addArgument('PETSc', 'C_VERSION',                   nargs.Arg(None, 'Unknown', 'The version of the C compiler'))
    help.addArgument('PETSc', 'CFLAGS_g',                    nargs.Arg(None, 'Unknown', 'Flags for the C compiler with BOPT=g'))
    help.addArgument('PETSc', 'CFLAGS_O',                    nargs.Arg(None, 'Unknown', 'Flags for the C compiler with BOPT=O'))
    help.addArgument('PETSc', 'CXX_VERSION',                 nargs.Arg(None, 'Unknown', 'The version of the C++ compiler'))
    help.addArgument('PETSc', 'CXXFLAGS_g',                  nargs.Arg(None, 'Unknown', 'Flags for the C++ compiler with BOPT=g'))
    help.addArgument('PETSc', 'CXXFLAGS_O',                  nargs.Arg(None, 'Unknown', 'Flags for the C++ compiler with BOPT=O'))
    help.addArgument('PETSc', 'F_VERSION',                   nargs.Arg(None, 'Unknown', 'The version of the Fortran compiler'))
    help.addArgument('PETSc', 'FFLAGS_g',                    nargs.Arg(None, 'Unknown', 'Flags for the Fortran compiler with BOPT=g'))
    help.addArgument('PETSc', 'FFLAGS_O',                    nargs.Arg(None, 'Unknown', 'Flags for the Fortran compiler with BOPT=O'))
    help.addArgument('PETSc', '-with-mpi',                   nargs.ArgBool(None, 1, 'If this is false, MPIUNI will be used as a uniprocessor substitute'))
    help.addArgument('PETSc', '-with-libtool',               nargs.ArgBool(None, 0, 'Specify that libtool should be used for compiling and linking'))
    help.addArgument('PETSc', '-with-make',                  nargs.Arg(None, 'make',   'Specify make'))
    help.addArgument('PETSc', '-with-ranlib',                nargs.Arg(None, None, 'Specify ranlib'))
    help.addArgument('PETSc', '-with-default-language=<c,c++,c++-complex,0(zero for no default)>', nargs.Arg(None, 'c', 'Specifiy default language of libraries'))
    help.addArgument('PETSc', '-with-default-optimization=<g,O,0(zero for no default)>',           nargs.Arg(None, 'g', 'Specifiy default optimization of libraries'))

    self.framework.argDB['PETSCFLAGS'] = ''
    self.framework.argDB['COPTFLAGS']  = ''
    self.framework.argDB['FOPTFLAGS']  = ''
    self.framework.argDB['BOPT']       = 'O'
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
      if self.framework.argDB['C_VERSION']   == 'Unknown':
        self.framework.argDB['C_VERSION']   = options.getCompilerVersion('C',       self.compilers.CC,  self)
      if self.framework.argDB['CXX']:
        if self.framework.argDB['CXX_VERSION'] == 'Unknown':
          self.framework.argDB['CXX_VERSION'] = options.getCompilerVersion('Cxx',     self.compilers.CXX, self)
      else:
          self.framework.argDB['CXX_VERSION'] = ''

      if self.framework.argDB['CFLAGS_g']   == 'Unknown':
        self.framework.argDB['CFLAGS_g']    = options.getCompilerFlags('C',       self.compilers.CC,  'g', self)
      if self.framework.argDB['CFLAGS_O']   == 'Unknown':
        self.framework.argDB['CFLAGS_O']    = options.getCompilerFlags('C',       self.compilers.CC,  'O', self)
      if self.framework.argDB['CXX']:
        if self.framework.argDB['CXXFLAGS_g'] == 'Unknown':
          self.framework.argDB['CXXFLAGS_g']  = options.getCompilerFlags('Cxx',     self.compilers.CXX, 'g', self)
        if self.framework.argDB['CXXFLAGS_O'] == 'Unknown':
          self.framework.argDB['CXXFLAGS_O']  = options.getCompilerFlags('Cxx',     self.compilers.CXX, 'O', self)
      else:
        self.framework.argDB['CXXFLAGS_g']  = ''
        self.framework.argDB['CXXFLAGS_O']  = ''

      if hasattr(self.compilers,'FC'):
        if self.framework.argDB['F_VERSION']   == 'Unknown':
          self.framework.argDB['F_VERSION']   = options.getCompilerVersion('Fortran', self.compilers.FC,  self)
        if self.framework.argDB['FFLAGS_g']   == 'Unknown':
          self.framework.argDB['FFLAGS_g']    = options.getCompilerFlags('Fortran', self.compilers.FC,  'g', self)
        if self.framework.argDB['FFLAGS_O']   == 'Unknown':
          self.framework.argDB['FFLAGS_O']    = options.getCompilerFlags('Fortran', self.compilers.FC,  'O', self)

    # does C++ compiler (IBM's xlC) need special for .c files as c++?
    if self.framework.argDB['CXX']:
      self.pushLanguage('C++')
      self.sourceExtension = '.c'
      if not self.checkCompile('class somename { int i; };'):
        oldFlags = self.framework.argDB['CXXFLAGS']
        self.framework.argDB['CXXFLAGS'] = oldFlags+' -+'
        if not self.checkCompile('class somename { int i; };'):
          self.framework.argDB['CXXFLAGS'] = oldFlags
      self.popLanguage()


    self.addSubstitution('C_VERSION',   self.framework.argDB['C_VERSION'])
    self.addSubstitution('CFLAGS_g',    self.framework.argDB['CFLAGS_g'])
    self.addSubstitution('CFLAGS_O',    self.framework.argDB['CFLAGS_O'])
    self.framework.addSubstitution('CFLAGS',self.framework.argDB['CFLAGS'])
    self.addSubstitution('CXX_VERSION', self.framework.argDB['CXX_VERSION'])
    self.addSubstitution('CXXFLAGS_g',  self.framework.argDB['CXXFLAGS_g'])
    self.addSubstitution('CXXFLAGS_O',  self.framework.argDB['CXXFLAGS_O'])
    self.framework.addSubstitution('CXXFLAGS',self.framework.argDB['CXXFLAGS'])
    self.addSubstitution('F_VERSION',   self.framework.argDB['F_VERSION'])
    self.addSubstitution('FFLAGS_g',    self.framework.argDB['FFLAGS_g'])
    self.addSubstitution('FFLAGS_O',    self.framework.argDB['FFLAGS_O'])
    self.framework.addSubstitution('FFLAGS',self.framework.argDB['FFLAGS'])

    self.framework.addSubstitution('PETSCFLAGS', self.framework.argDB['PETSCFLAGS'])
    self.framework.addSubstitution('COPTFLAGS',  self.framework.argDB['COPTFLAGS'])
    self.framework.addSubstitution('FOPTFLAGS',  self.framework.argDB['FOPTFLAGS'])
    return

  def checkFortranCompilerOption(self, option, lang = 'F77'):
    self.pushLanguage(lang)
    self.sourceExtension = '.F'
    oldFlags = self.framework.argDB['FFLAGS']
    success  = 0

    (output, returnCode) = self.outputCompile('', '')
    if returnCode: raise RuntimeError('Could not compile anything with '+lang+' compiler:\n'+output)

    self.framework.argDB['FFLAGS'] = option
    (newOutput, returnCode) = self.outputCompile('', '')
    if not returnCode and output == newOutput:
      success = 1

    self.framework.argDB['FFLAGS'] = oldFlags
    self.popLanguage()
    return success

  def configureFortranPIC(self):
    '''Determine the PIC option for the Fortran compiler'''
    # We use the framework in order to remove the PETSC_ namespace
    option = ''
    if self.checkFortranCompilerOption('-PIC'):
      option = '-PIC'
    elif self.checkFortranCompilerOption('-fPIC'):
      option = '-fPIC'
    elif self.checkFortranCompilerOption('-KPIC'):
      option = '-KPIC'
    self.framework.addSubstitution('FC_SHARED_OPT', option)
    return

  def configureFortranCPP(self):
    '''Determine if Fortran handles CPP properly'''
    if 'FC' in self.framework.argDB:
      # IBM xlF chokes on this
      if not self.checkFortranCompilerOption('-DPTesting'):
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

  def configureFortranStubs(self):
    '''Determine whether the Fortran stubs exist or not'''
    stubDir = os.path.join(self.framework.argDB['PETSC_DIR'], 'src', 'fortran', 'auto')
    if not os.path.exists(os.path.join(stubDir, 'makefile.src')):
      self.framework.log.write('WARNING: Fortran stubs have not been generated in '+stubDir+'\n')
      self.framework.getExecutable('bfort', getFullPath = 1)
      if hasattr(self.framework, 'bfort'):
        self.framework.log.write('           Running '+self.framework.bfort+' to generate Fortran stubs\n')
        (status,output) = commands.getstatusoutput('export PETSC_ARCH=linux;make allfortranstubs')
        # filter out the normal messages, user has to cope with error messages
        cnt = 0
        for i in output.split('\n'):
          if not (i.startswith('fortranstubs in:') or i.startswith('Fixing pointers') or i.find('ACTION=') >= 0):
            if not cnt:
              self.framework.log.write('*******Error generating Fortran stubs****\n')
            cnt = cnt + 1
            self.framework.log.write(i+'\n')
        if not cnt:
          self.framework.log.write('           Completed generating Fortran stubs\n')
        else:
          self.framework.log.write('*******End of error messages from generating Fortran stubs****\n')
      else:
        self.framework.log.write('           See http:/www.mcs.anl.gov/petsc/petsc-2/developers for how\n')
        self.framework.log.write('           to obtain bfort to generate the Fortran stubs or make sure\n')
        self.framework.log.write('           bfort is in your path\n')
        self.framework.log.write('WARNING: Turning off Fortran interfaces for PETSc')
        del self.framework.argDB['FC']
        self.compilers.addSubstitution('FC', '')
    else:
      self.framework.log.write('Fortran stubs do exist in '+stubDir+'\n')
    return

  def configureDynamicLibraries(self):
    '''Checks whether dynamic libraries should be used, for which you must
      - Specify --enable-dynamic
      - Find dlfcn.h and libdl
    Defines PETSC_USE_DYNAMIC_LIBRARIES is they are used
    Also checks that dlopen() takes RTLD_GLOBAL, and defines PETSC_HAVE_RTLD_GLOBAL if it does'''
    if not self.framework.archBase.startswith('darwin') or  (self.usingMPIUni and not self.framework.argDB.has_key('FC')):
      useDynamic = self.framework.argDB['enable-dynamic'] and self.headers.check('dlfcn.h') and self.libraries.haveLib('dl')
      self.addDefine('USE_DYNAMIC_LIBRARIES', useDynamic)
      if useDynamic and self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);\n'):
        self.addDefine('HAVE_RTLD_GLOBAL', 1)

    # This is really bad
    flag = '-L'
    if self.framework.archBase == 'linux':
      flag = '-Wl,-rpath,'
    elif self.framework.archBase.startswith('irix'):
      flag = '-rpath '
    elif self.framework.archBase.startswith('osf'):
      flag = '-Wl,-rpath,'
    elif self.framework.archBase.startswith('solaris'):
      flag = '-R'
    #  can only get dynamic shared libraries on Mac X with no g77 and no MPICH (maybe LAM?)
    if self.framework.archBase.startswith('darwin') and self.usingMPIUni and not self.framework.argDB.has_key('FC'):
      if self.framework.sharedBlasLapack: bls = 'BLASLAPACK_LIB_SHARED=${BLASLAPACK_LIB}\n'
      else:                               bls = ''
      self.framework.addSubstitution('DYNAMIC_SHARED_TARGET', bls+'MPI_LIB_SHARED=${MPI_LIB}\ninclude ${PETSC_DIR}/bmake/common/rules.shared.darwin7')
    else:
      self.framework.addSubstitution('DYNAMIC_SHARED_TARGET', 'include ${PETSC_DIR}/bmake/common/rules.shared.basic')
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
    if 'with-ranlib' in self.framework.argDB:
      found = self.framework.getExecutable(self.framework.argDB['with-ranlib'], resultName = 'RANLIB')
      if not found:
         raise RuntimeError('You set a value for --with-ranlib, but '+self.framework.argDB['with-ranlib']+' does not exist')
    else:
      found = self.framework.getExecutable('ranlib')
      if not found:
        self.framework.addSubstitution('RANLIB', 'true')
    self.framework.getExecutable('ps', path = '/usr/ucb:/usr/usb', resultName = 'UCBPS')
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
          raise RuntimeError('Invalid X include directory specified by --with-x-include='+os.path.abspath(self.framework.argDB['with-x-include']))
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
          raise RuntimeError('Invalid X library specified by --with-x-libary='+os.path.abspath(self.framework.argDB['with-x-library']))
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
      self.missingPrototypesExternC.append('int getdomainname(char *, int);')
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
      setattr(self.framework,'suppressGuard',1)
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
    self.usingMPIUni = 1
    return

  def configureMissingPrototypes(self):
    '''Checks for missing prototypes, which it adds to petscfix.h'''
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
    fd = open(os.path.join('bmake','variables'),'w')
    fd.write('PETSC_ARCH='+self.framework.arch+'\n')
    fd.write('include ${PETSC_DIR}/bmake/'+self.framework.arch+'/variables\n')
    fd.close()

    return

  def configureETags(self):
    '''Determine if etags files exist and try to create otherwise'''
    if not os.path.exists(os.path.join(self.framework.argDB['PETSC_DIR'], 'TAGS')):
      self.framework.log.write('WARNING: ETags files have not been created\n')
      self.framework.getExecutable('etags', getFullPath = 1)
      if hasattr(self.framework, 'etags'):
        pd = self.framework.argDB['PETSC_DIR']
        self.framework.log.write('           Running '+self.framework.etags+' to generate TAGS files\n')
        (status,output) = commands.getstatusoutput('export PETSC_ARCH=linux;make PETSC_DIR='+pd+' TAGSDIR='+pd+' etags')
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
          self.framework.log.write('*******End of error messages from generating etags files****\n')
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

  def configureScript(self):
    '''Output a script in the bmake directory which will reproduce the configuration'''
    scriptName = os.path.join('bmake', self.framework.arch, 'configure.py')
    if not os.path.exists(os.path.dirname(scriptName)):
      os.makedirs(os.path.dirname(scriptName))
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
    self.executeTest(self.configureCompilerFlags)
    if 'FC' in self.framework.argDB:
      self.executeTest(self.configureFortranStubs)
    if 'FC' in self.framework.argDB:
      self.executeTest(self.configureFortranPIC)
    else:
      self.framework.addSubstitution('FC_SHARED_OPT', '')
    self.executeTest(self.configureFortranCPP)
    self.executeTest(self.configureMPIUNI)
    self.executeTest(self.configureDynamicLibraries)
    self.executeTest(self.configureLibtool)
    self.executeTest(self.configureDebuggers)
    self.executeTest(self.configureMake)
    self.executeTest(self.configureMkdir)
    self.executeTest(self.configurePrograms)
    self.executeTest(self.configureMissingFunctions)
    self.executeTest(self.configureMissingSignals)
    self.executeTest(self.configureMemorySize)
    self.executeTest(self.configureX)
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
    self.executeTest(self.configureETags)
    self.executeTest(self.configureDocs)
    self.executeTest(self.configureScript)
    self.startLine()
    return
