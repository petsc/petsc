import config.base

import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.usingMPIUni  = 0
    self.defineAutoconfMacros()
    headersC = map(lambda name: name+'.h', ['dos', 'endian', 'fcntl', 'float', 'io', 'limits', 'malloc', 'pwd', 'search', 'strings',
                                            'stropts', 'unistd', 'machine/endian', 'sys/param', 'sys/procfs', 'sys/resource',
                                            'sys/systeminfo', 'sys/times', 'sys/utsname','string', 'stdlib',
                                            'sys/socket','sys/wait','netinet/in','netdb','Direct','time','Ws2tcpip','sys/types',
                                            'WindowsX'])
    functions = ['access', '_access', 'clock', 'drand48', 'getcwd', '_getcwd', 'getdomainname', 'gethostname', 'getpwuid',
                 'gettimeofday', 'getrusage', 'getwd', 'memalign', 'memmove', 'mkstemp', 'popen', 'PXFGETARG', 'rand',
                 'readlink', 'realpath', 'sbreak', 'sigaction', 'signal', 'sigset', 'sleep', '_sleep', 'socket', 'times',
                 'uname','snprintf','_snprintf','_fullpath','lseek','_lseek','time','fork','stricmp','bzero','dlopen','dlsym','erf']
    libraries1 = [(['socket', 'nsl'], 'socket')]
    self.setCompilers = self.framework.require('config.setCompilers', self)
    self.compilers    = self.framework.require('config.compilers',    self)
    self.types        = self.framework.require('config.types',        self)
    self.headers      = self.framework.require('config.headers',      self)
    self.functions    = self.framework.require('config.functions',    self)
    self.libraries    = self.framework.require('config.libraries',    self)
    self.update       = self.framework.require('PETSc.packages.update', self)
    self.x11          = self.framework.require('PETSc.packages.X11', self)
    self.compilers.headerPrefix = self.headerPrefix
    self.types.headerPrefix     = self.headerPrefix
    self.headers.headerPrefix   = self.headerPrefix
    self.functions.headerPrefix = self.headerPrefix
    self.libraries.headerPrefix = self.headerPrefix
    self.headers.headers.extend(headersC)
    self.functions.functions.extend(functions)
    self.libraries.libraries.extend(libraries1)
    # Check for packages
    import PETSc.packages

    for package in os.listdir(os.path.dirname(PETSc.packages.__file__)):
      (packageName, ext) = os.path.splitext(package)
      if not packageName.startswith('.') and not packageName.startswith('#') and ext == '.py' and not packageName == '__init__':
        packageObj              = self.framework.require('PETSc.packages.'+packageName, self)
        packageObj.headerPrefix = self.headerPrefix
        setattr(self, packageName.lower(), packageObj)
    # Put in dependencies
    self.framework.require('PETSc.packages.update', self.setCompilers)
    self.framework.require('PETSc.packages.compilerFlags', self.compilers)

    # List of packages actually found
    self.framework.packages = []
    return

  def __str__(self):
    desc = ['PETSc:']
    carch = self.framework.argDB['PETSC_ARCH']
    cdir  = self.framework.argDB['PETSC_DIR']
    envarch = os.getenv('PETSC_ARCH')
    envdir  = os.getenv('PETSC_DIR')
    change=0
    if not carch == envarch :
      change=1
      desc.append('  **\n  ** Configure has determined that your PETSC_ARCH must be specified as:')
      desc.append('  **  ** PETSC_ARCH: '+str(self.framework.argDB['PETSC_ARCH']+'\n  **'))
    else:
      desc.append('  PETSC_ARCH: '+str(self.framework.argDB['PETSC_ARCH']))
    if not cdir == envdir :
      change=1
      desc.append('  **\n  ** Configure has determined that your PETSC_DIR must be specified as:')
      desc.append('  **  **  PETSC_DIR: '+str(self.framework.argDB['PETSC_DIR']+'\n  **'))
    else:
      desc.append('  PETSC_DIR: '+str(self.framework.argDB['PETSC_DIR']))
    if change:
      desc.append('  ** Please make the above changes to your environment or on the command line for make.\n  **')
    
    return '\n'.join(desc)+'\n'
                              
  def setupHelp(self, help):
    import nargs

    help.addArgument('PETSc', 'PETSC_DIR',                   nargs.Arg(None, None, 'The root directory of the PETSc installation'))
    help.addArgument('PETSc', 'PETSC_ARCH',                  nargs.Arg(None, None, 'The machine architecture'))
    help.addArgument('PETSc', '-with-debug=<bool>',          nargs.ArgBool(None, 1, 'Activate debugging code in PETSc'))
    help.addArgument('PETSc', '-with-log=<bool>',            nargs.ArgBool(None, 1, 'Activate logging code in PETSc'))
    help.addArgument('PETSc', '-with-stack=<bool>',          nargs.ArgBool(None, 1, 'Activate manual stack tracing code in PETSc'))
    help.addArgument('PETSc', '-with-ctable=<bool>',         nargs.ArgBool(None, 1, 'Use CTABLE hashing for certain search functions - to conserve memory'))
    help.addArgument('PETSc', '-with-dynamic=<bool>',        nargs.ArgBool(None, 1, 'Build dynamic libraries for PETSc'))
    help.addArgument('PETSc', '-with-shared=<bool>',         nargs.ArgBool(None, 1, 'Build shared libraries for PETSc'))
    help.addArgument('PETSc', '-with-etags=<bool>',          nargs.ArgBool(None, 1, 'Build etags if they do not exist'))
    help.addArgument('PETSc', '-with-fortran-kernels=<bool>',nargs.ArgBool(None, 0, 'Use Fortran for linear algebra kernels'))
    help.addArgument('PETSc', '-with-libtool=<bool>',        nargs.ArgBool(None, 0, 'Specify that libtool should be used for compiling and linking'))
    help.addArgument('PETSc', '-with-make',                  nargs.Arg(None, 'make', 'Specify make'))
    help.addArgument('PETSc', '-prefix=<path>',              nargs.Arg(None, '',     'Specifiy location to install PETSc (eg. /usr/local)'))
    help.addArgument('PETSc', '-with-gcov=<bool>',           nargs.ArgBool(None, 0, 'Specify that GNUs coverage tool gcov is used'))
    help.addArgument('PETSc', '-with-64-bit-ints=<bool>',     nargs.ArgBool(None, 0, 'Use 64 bit integers (long long) for indexing in vectors and matrices'))    
    return

  def defineAutoconfMacros(self):
    self.hostMacro = 'dnl Version: 2.13\ndnl Variable: host_cpu\ndnl Variable: host_vendor\ndnl Variable: host_os\nAC_CANONICAL_HOST'
    return

  def configureLibraryOptions(self):
    '''Sets PETSC_USE_DEBUG, PETSC_USE_LOG, PETSC_USE_STACK, PETSC_USE_CTABLE and PETSC_USE_FORTRAN_KERNELS'''
    self.useDebug = self.framework.argDB['with-debug']
    self.addDefine('USE_DEBUG', self.useDebug)
    self.useLog   = self.framework.argDB['with-log']
    self.addDefine('USE_LOG',   self.useLog)
    self.useStack = self.framework.argDB['with-stack']
    self.addDefine('USE_STACK', self.useStack)
    self.useCtable = self.framework.argDB['with-ctable']
    self.addDefine('USE_CTABLE', self.useCtable)
    self.useFortranKernels = self.framework.argDB['with-fortran-kernels']
    self.addDefine('USE_FORTRAN_KERNELS', self.useFortranKernels)
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

  def configureFortranCommandline(self):
    '''Check for the mechanism to retrieve command line arguments in Fortran'''
    self.pushLanguage('C')
    if self.functions.check('ipxfargc_', libraries = self.compilers.flibs):
      self.addDefine('HAVE_PXFGETARG_NEW',1)
    elif self.functions.check('f90_unix_MP_iargc', libraries = self.compilers.flibs):
      self.addDefine('HAVE_NAGF90',1)
    elif self.functions.check('PXFGETARG', libraries = self.compilers.flibs):
      self.addDefine('HAVE_PXFGETARG',1)
    elif self.functions.check('GETARG@16', libraries = self.compilers.flibs): 
      self.addDefine('USE_NARGS',1)
      self.addDefine('HAVE_IARG_COUNT_PROGNAME',1)
    return

  def configureDynamicLibraries(self):
    '''Checks whether dynamic libraries should be used, for which you must
      - Specify --with-dynamic
      - Find dlfcn.h and libdl
    Defines PETSC_USE_DYNAMIC_LIBRARIES is they are used
    Also checks that dlopen() takes RTLD_GLOBAL, and defines PETSC_HAVE_RTLD_GLOBAL if it does'''
    self.useDynamic = 0
    if not (self.framework.argDB['PETSC_ARCH_BASE'].startswith('aix') or (self.framework.argDB['PETSC_ARCH_BASE'].startswith('darwin') and not (self.usingMPIUni and not self.framework.argDB.has_key('FC')))):
      self.useDynamic = self.framework.argDB['with-shared'] and self.framework.argDB['with-dynamic'] and self.headers.check('dlfcn.h') and self.libraries.haveLib('dl')
      self.addDefine('USE_DYNAMIC_LIBRARIES', self.useDynamic)
      if self.useDynamic and self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);\n'):
        self.addDefine('HAVE_RTLD_GLOBAL', 1)

    #  can only get dynamic shared libraries on Mac X with no g77 and no MPICH (maybe LAM?)
    if self.useDynamic and self.framework.argDB['PETSC_ARCH_BASE'].startswith('darwin') and self.usingMPIUni and not self.framework.argDB.has_key('FC'):
      if self.blaslapack.sharedBlasLapack: bls = 'BLASLAPACK_LIB_SHARED=${BLASLAPACK_LIB}\n'
      else:                                bls = ''
      self.framework.addSubstitution('DYNAMIC_SHARED_TARGET', bls+'MPI_LIB_SHARED=${MPI_LIB}\ninclude ${PETSC_DIR}/bmake/common/rules.shared.darwin7')
    else:
      self.framework.addSubstitution('DYNAMIC_SHARED_TARGET', 'include ${PETSC_DIR}/bmake/common/rules.shared.basic')

    if self.setCompilers.CSharedLinkerFlag is None:
      self.addSubstitution('C_LINKER_SLFLAG', '-L')
    else:
      self.addSubstitution('C_LINKER_SLFLAG', self.setCompilers.CSharedLinkerFlag)
    if 'CXX' in self.framework.argDB:
      if self.setCompilers.CxxSharedLinkerFlag is None:
        self.addSubstitution('CXX_LINKER_SLFLAG', '-L')
      else:
        self.addSubstitution('CXX_LINKER_SLFLAG', self.setCompilers.CxxSharedLinkerFlag)
    if 'FC' in self.framework.argDB:
      if self.setCompilers.F77SharedLinkerFlag is None:
        self.addSubstitution('F77_LINKER_SLFLAG', '-L')
      else:
        self.addSubstitution('F77_LINKER_SLFLAG', self.setCompilers.F77SharedLinkerFlag)
    return


  def configureLibtool(self):
    if self.framework.argDB['with-libtool']:
      self.framework.addSubstitution('LT_CC', '${PETSC_LIBTOOL} ${LIBTOOL} --mode=compile')
      self.framework.addSubstitution('LIBTOOL', '${SHELL} ${top_builddir}/libtool')
      self.framework.addSubstitution('SHARED_TARGET', 'shared_libtool')
    else:
      self.framework.addSubstitution('LT_CC', '')
      self.framework.addSubstitution('LIBTOOL', '')
      # OSF/alpha cannot handle multiple -rpath, therefor current configure cannot do shared on alpha
      if self.framework.argDB['with-shared'] and not self.framework.argDB['PETSC_ARCH_BASE'].startswith('osf'):
        self.framework.addSubstitution('SHARED_TARGET', 'shared_'+self.framework.argDB['PETSC_ARCH_BASE'])
      else:
        self.framework.addSubstitution('SHARED_TARGET', 'shared_none')
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
    '''Check that the archiver exists and can make a library usable by the compiler'''
    def checkArchive(command, status, output, error):
      if error or status:
        self.framework.log.write('Possible ERROR while running archiver: '+output)
        if status: self.framework.log.write('ret = '+str(status)+'\n')
        if error: self.framework.log.write('error message = {'+error+'}\n')
        os.remove('conf1.o')
        raise RuntimeError('Archiver is not functional')
      return
    self.framework.getExecutable(self.framework.argDB['with-ar'], getFullPath = 1, resultName = 'AR')
    self.framework.addArgumentSubstitution('AR_FLAGS', 'AR_FLAGS')
    self.pushLanguage('C')
    if not self.checkCompile('', 'int foo(int a) {\n  return a+1;\n}\n\n', cleanup = 0, codeBegin = '', codeEnd = ''):
      raise RuntimeError('Compiler is not functional')
    os.rename(self.compilerObj, 'conf1.o')
    (output, error, status) = config.base.Configure.executeShellCommand(self.framework.AR+' '+self.framework.argDB['AR_FLAGS']+' conf1.a conf1.o', checkCommand = checkArchive, log = self.framework.log)
    os.remove('conf1.o')
    oldLibs = self.framework.argDB['LIBS']
    self.framework.argDB['LIBS'] = 'conf1.a'
    if not self.checkLink('extern int foo(int);', '  int b = foo(1);  if (b);\n', cleanup = 0):
      self.framework.argDB['LIBS'] = oldLibs
      os.remove('conf1.a')
      raise RuntimeError('Compiler cannot use libaries made by archiver')
    self.framework.argDB['LIBS'] = oldLibs
    os.remove('conf1.a')
    self.popLanguage()
    return

  def configureRanlib(self):
    '''Check for ranlib, using "true" if it is not found. If found, test it on a library.'''
    if 'with-ranlib' in self.framework.argDB:
      found = self.framework.getExecutable(self.framework.argDB['with-ranlib'], resultName = 'RANLIB')
      if not found:
         raise RuntimeError('You set a value for --with-ranlib, but '+self.framework.argDB['with-ranlib']+' does not exist')
    else:
      found = self.framework.getExecutable('ranlib', resultName = 'RANLIB')
      if not found:
        self.framework.addSubstitution('RANLIB', 'true')
    if found:
      def checkRanlib(command, status, output, error):
        if error or status:
          self.framework.log.write('Possible ERROR while running ranlib: '+output)
          if status: self.framework.log.write('ret = '+str(status)+'\n')
          if error: self.framework.log.write('error message = {'+error+'}\n')
          os.remove('conf1.a')
          raise RuntimeError('Ranlib is not functional')
        return
      self.pushLanguage('C')
      if not self.checkCompile('', 'int foo(int a) {\n  return a+1;\n}\n\n', cleanup = 0, codeBegin = '', codeEnd = ''):
        raise RuntimeError('Compiler is not functional')
      os.rename(self.compilerObj, 'conf1.o')
      (output, error, status) = config.base.Configure.executeShellCommand(self.framework.AR+' '+self.framework.argDB['AR_FLAGS']+' conf1.a conf1.o', log = self.framework.log)
      os.remove('conf1.o')
      self.popLanguage()
      config.base.Configure.executeShellCommand(self.framework.RANLIB+' conf1.a', checkCommand = checkRanlib, log = self.framework.log)
      os.remove('conf1.a')
    return

  def configurePrograms(self):
    '''Check for the programs needed to build and run PETSc'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.getExecutable('sh',   getFullPath = 1, resultName = 'SHELL')
    self.framework.getExecutable('sed',  getFullPath = 1)
    self.framework.getExecutable('mv',   getFullPath = 1)
    self.framework.getExecutable('diff', getFullPath = 1)
    # check if diff supports -w option for ignoring whitespace
    f = file('diff1', 'w')
    f.write('diff\n')
    f.close()
    f = file('diff2', 'w')
    f.write('diff  \n')
    f.close()
    (out,err,status) = Configure.executeShellCommand(getattr(self.framework, 'diff')+' -w diff1 diff2')
    os.unlink('diff1')
    os.unlink('diff2')
    if not status:    
      self.framework.addSubstitution('DIFF',getattr(self.framework, 'diff')+' -w')
    
    self.framework.getExecutable('ps',   path = '/usr/ucb:/usr/usb', resultName = 'UCBPS')
    if hasattr(self.framework, 'UCBPS'):
      self.addDefine('HAVE_UCBPS', 1)
    self.framework.getExecutable('gzip',getFullPath=1, resultName = 'GZIP')
    if hasattr(self.framework, 'GZIP'):
      self.addDefine('HAVE_GZIP',1)
    return

  def configureMissingDefines(self):
    '''Checks for limits'''
    if not self.checkCompile('#ifdef PETSC_HAVE_LIMITS_H\n  #include <limits.h>\n#endif\n', 'int i=INT_MAX;\n\nif (i);\n'):
      self.addDefine('INT_MIN', '(-INT_MAX - 1)')
      self.addDefine('INT_MAX', 2147483647)
    if not self.checkCompile('#ifdef PETSC_HAVE_FLOAT_H\n  #include <float.h>\n#endif\n', 'double d=DBL_MAX;\n\nif (d);\n'):
      self.addDefine('DBL_MIN', 2.2250738585072014e-308)
      self.addDefine('DBL_MAX', 1.7976931348623157e+308)
    return

  def configureMissingFunctions(self):
    '''Checks for SOCKETS'''
    if not self.functions.haveFunction('socket'):
      # solaris requires these two libraries for socket()
      if self.libraries.haveLib('socket') and self.libraries.haveLib('nsl'):
        self.addDefine('HAVE_SOCKET', 1)
        self.framework.argDB['LIBS'] += ' -lsocket -lnsl'
      # Windows requires Ws2_32.lib for socket(), uses stdcall, and declspec prototype decoration
      if self.libraries.check('Ws2_32.lib','socket',prototype='#include <Winsock2.h>',call='socket(0,0,0);'):
        self.addDefine('HAVE_WINSOCK2_H',1)
        self.addDefine('HAVE_SOCKET', 1)
        if self.checkLink('#include <Winsock2.h>','closesocket(0)'):
          self.addDefine('HAVE_CLOSESOCKET',1)
        if self.checkLink('#include <Winsock2.h>','WSAGetLastError()'):
          self.addDefine('HAVE_WSAGETLASTERROR',1)
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

  def configureMissingErrnos(self):
    '''Check for missing errno values, and define MISSING_<errno value> if necessary'''
    for errnoval in ['EINTR']:
      if not self.checkCompile('#include <errno.h>','int i='+errnoval+';\n\nif (i);\n'):
        self.addDefine('MISSING_ERRNO_'+errnoval, 1)
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

  def checkPrototype(self, includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None):
    (output, error, status) = self.outputCompile(includes, body, cleanup, codeBegin, codeEnd)
    output += error
    if output.find('implicit') >= 0 or output.find('Implicit') >= 0:
      return 0
    return 1

  def configureGetDomainName(self):
    if not self.checkPrototype('#include <unistd.h>\n','char test[10]; int err = getdomainname(test,10);'):
      self.addPrototype('int getdomainname(char *, int);', 'C')
    if 'CXX' in self.framework.argDB:
      self.pushLanguage('C++')
      if not self.checkLink('#include <unistd.h>\n','char test[10]; int err = getdomainname(test,10);'):
        self.addPrototype('int getdomainname(char *, int);', 'extern C')
      self.popLanguage()  
    return
 
  def configureIRIX(self):
    '''IRIX specific stuff'''
    if self.framework.argDB['PETSC_ARCH_BASE'].startswith('irix'):
      self.addDefine('USE_KBYTES_FOR_SIZE', 1)
    return

  def configureSolaris(self):
    '''Solaris specific stuff'''
    if self.framework.argDB['PETSC_ARCH_BASE'].startswith('solaris'):
      if os.path.isdir(os.path.join('/usr','ucblib')):
        try:
          flag = getattr(self.setCompilers, self.language[-1].replace('+', 'x')+'SharedLinkerFlag')
        except AttributeError:
          flag = None
        if flag is None:
          self.framework.argDB['LIBS'] += ' -L/usr/ucblib'
        else:
          self.framework.argDB['LIBS'] += ' '+flag+'/usr/ucblib'
    return

  def configureLinux(self):
    '''Linux specific stuff'''
    if self.framework.argDB['PETSC_ARCH_BASE'] == 'linux':
      self.addDefine('HAVE_DOUBLE_ALIGN_MALLOC', 1)
    return

  def configureWin32(self):
    '''Win32 non-cygwin specific stuff'''
    kernel32=0
    if self.libraries.check('Kernel32.lib','GetComputerName',prototype='#include <Windows.h>',
                            call='GetComputerName(NULL,NULL);'):
      self.addDefine('HAVE_WINDOWS_H',1)
      self.addDefine('HAVE_GETCOMPUTERNAME',1)
      kernel32=1
    elif self.libraries.check('kernel32','GetComputerName',prototype='#include <Windows.h>',
                              call='GetComputerName(NULL,NULL);'):
      self.addDefine('HAVE_WINDOWS_H',1)
      self.addDefine('HAVE_GETCOMPUTERNAME',1)
      kernel32=1
    if kernel32:  
      if self.checkLink('#include <Windows.h>','GetProcAddress(0,0)'):
        self.addDefine('HAVE_GETPROCADDRESS',1)
      if self.checkLink('#include <Windows.h>','LoadLibrary(0)'):
        self.addDefine('HAVE_LOADLIBRARY',1)
      if self.checkLink('#include <Windows.h>\n','QueryPerformanceCounter(0);\n'):
        self.addDefine('USE_NT_TIME',1)
    if self.libraries.check('Advapi32.lib','GetUserName',prototype='#include <Windows.h>',
                            call='GetUserName(NULL,NULL);'):
      self.addDefine('HAVE_GET_USER_NAME',1)
    elif self.libraries.check('advapi32','GetUserName',prototype='#include <Windows.h>',
                              call='GetUserName(NULL,NULL);'):
      self.addDefine('HAVE_GET_USER_NAME',1)
        
    if not self.libraries.check('User32.lib','GetDC',prototype='#include <Windows.h>',call='GetDC(0);'):
      self.libraries.check('user32','GetDC',prototype='#include <Windows.h>',call='GetDC(0);')
    if not self.libraries.check('Gdi32.lib','CreateCompatibleDC',prototype='#include <Windows.h>',call='CreateCompatibleDC(0);'):
      self.libraries.check('gdi32','CreateCompatibleDC',prototype='#include <Windows.h>',call='CreateCompatibleDC(0);')
      
    if not self.checkCompile('#include <sys/types.h>\n','uid_t u;\n'):
      self.addTypedef('int', 'uid_t')
      self.addTypedef('int', 'gid_t')
    if not self.checkLink('#if defined(PETSC_HAVE_UNISTD_H)\n#include <unistd.h>\n#endif\n','int a=R_OK;\n'):
      self.framework.addDefine('R_OK', '04')
      self.framework.addDefine('W_OK', '02')
      self.framework.addDefine('X_OK', '01')
    if not self.checkLink('#include <sys/stat.h>\n','int a=0;\nif (S_ISDIR(a)){}\n'):
      self.framework.addDefine('S_ISREG(a)', '(((a)&_S_IFMT) == _S_IFREG)')
      self.framework.addDefine('S_ISDIR(a)', '(((a)&_S_IFMT) == _S_IFDIR)')
    if self.checkCompile('#include <Windows.h>\n','LARGE_INTEGER a;\nDWORD b=a.u.HighPart;\n'):
      self.addDefine('HAVE_LARGE_INTEGER_U',1)

    # Windows requires a Binary file creation flag when creating/opening binary files.  Is a better test in order?
    if self.checkCompile('#include <Windows.h>\n',''):
      self.addDefine('HAVE_O_BINARY',1)

    if self.framework.argDB['CC'].find('win32fe') >= 0:
      self.addDefine('PATH_SEPARATOR','\';\'')
      self.addDefine('DIR_SEPARATOR','\'\\\\\'')
      self.addDefine('REPLACE_DIR_SEPARATOR','\'/\'')
      self.addDefine('CANNOT_START_DEBUGGER',1)
    else:
      self.addDefine('PATH_SEPARATOR','\':\'')
      self.addDefine('REPLACE_DIR_SEPARATOR','\'\\\\\'')
      self.addDefine('DIR_SEPARATOR','\'/\'')
    return
    
  def configureMPIUNI(self):
    '''If MPI was not found, setup MPIUNI, our uniprocessor version of MPI'''
    if self.framework.argDB['with-mpi']:
      if self.mpi.foundMPI:
        return
      else:
        raise RuntimeError('********** Error: Unable to locate a functional MPI. Please consult configure.log. **********')
    self.framework.addDefine('HAVE_MPI', 1)
    if 'STDCALL' in self.compilers.defines:
      self.framework.addSubstitution('MPI_INCLUDE', '-I'+'${PETSC_DIR}/include/mpiuni'+' -D'+'MPIUNI_USE_STDCALL')
    else:
      self.framework.addSubstitution('MPI_INCLUDE', '-I'+'${PETSC_DIR}/include/mpiuni')
    self.framework.addSubstitution('MPI_LIB',     '-L${PETSC_DIR}/lib/lib${BOPT}/${PETSC_ARCH} -lmpiuni')
    self.framework.addSubstitution('MPIRUN',      '${PETSC_DIR}/bin/mpirun.uni')
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
      self.addPrototype('typedef int MPI_Fint;')
    if not 'HAVE_MPI_COMM_F2C' in self.mpi.defines:
      self.addPrototype('#define MPI_Comm_f2c(a) (a)')
    if not 'HAVE_MPI_COMM_C2F' in self.mpi.defines:
      self.addPrototype('#define MPI_Comm_c2f(a) (a)')
    return

  def configureMachineInfo(self):
    '''Define a string incorporating all configuration data needed for a bug report'''
    #self.addDefine('MACHINE_INFO', '"Libraries compiled on `date` on `hostname`\\nMachine characteristics: `uname -a`\\n-----------------------------------------\\nUsing C compiler: ${CC} ${COPTFLAGS} ${CCPPFLAGS}\\nC Compiler version: ${C_VERSION}\\nUsing C compiler: ${CXX} ${CXXOPTFLAGS} ${CXXCPPFLAGS}\\nC++ Compiler version: ${CXX_VERSION}\\nUsing Fortran compiler: ${FC} ${FOPTFLAGS} ${FCPPFLAGS}\\nFortran Compiler version: ${F_VERSION}\\n-----------------------------------------\\nUsing PETSc flags: ${PETSCFLAGS} ${PCONF}\\n-----------------------------------------\\nUsing include paths: ${PETSC_INCLUDE}\\n-----------------------------------------\\nUsing PETSc directory: ${PETSC_DIR}\\nUsing PETSc arch: ${PETSC_ARCH}"\\n')
    return

  def configureETags(self):
    '''Determine if etags files exist and try to create otherwise'''
    if not os.path.exists(os.path.join(self.framework.argDB['PETSC_DIR'], 'TAGS')):
      self.framework.log.write('WARNING: ETags files have not been created\n')
      self.framework.getExecutable('etags', getFullPath = 1)
      if hasattr(self.framework, 'etags'):
        pd = self.framework.argDB['PETSC_DIR']
        if pd[-1]=='/': pd = pd[:-1] # etags chokes if there's a trailing /
        self.framework.log.write('           Running '+self.framework.etags+' to generate TAGS files\n')
        try:
          (output, error, status) = config.base.Configure.executeShellCommand('make PETSC_ARCH=solaris BOPT=g PETSC_DIR='+pd+' TAGSDIR='+pd+' etags', timeout = 15*60.0, log = self.framework.log)
          # filter out the normal messages
          cnt = 0
          for i in output.split('\n'):
            if not (i.startswith('etags_') or i.find('TAGS') >= 0 or i.find('Entering') >= 0 or i.find('Leaving') >= 0 or i==''):
              if not cnt:
                self.framework.log.write('*******Error generating etags files****\n')
              cnt = cnt + 1
              self.framework.log.write(i+'\n')
          if not cnt:
            self.framework.log.write('           Completed generating etags files\n')
            self.framework.actions.addArgument('PETSc', 'File creation', 'Generated etags files in '+pd)
          else:
            self.framework.log.write('*******End of error messages from generating etags files*******\n')
        except RuntimeError, e:
          self.framework.log.write('*******Error generating etags files: '+str(e)+'*******\n')
      else:
        self.framework.log.write('           The etags command is not in your path, cannot build etags files\n')
    else:
      self.framework.log.write('Found etags file \n')
    return

  def configureRegression(self):
    '''Output a file listing the jobs that should be run by the PETSc buildtest'''
    jobs  = []    # Jobs can be run on with all BOPTs
    rjobs = []    # Jobs can only be run with real numbers; i.e. NOT BOPT=g_complex or BOPT=O_complex
    ejobs = []    # Jobs that require an external package install (also cannot work with complex)
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
        rjobs.append('8')
      if self.update.hasdatafiles:
        rjobs.append('6')
      # add jobs for each external package (except X11, already done)
      for i in self.framework.packages:
        ejobs.append(i.name.upper())
    if os.path.isfile(os.path.join(self.bmakeDir, 'jobs')):
      try:
        os.unlink(os.path.join(self.bmakeDir, 'jobs'))
      except:
        raise RuntimeError('Unable to remove file '+os.path.join(self.bmakeDir, 'jobs')+'. Did a different user create it?')
    jobsFile  = file(os.path.abspath(os.path.join(self.bmakeDir, 'jobs')), 'w')
    jobsFile.write(' '.join(jobs)+'\n')
    jobsFile.close()
    self.framework.actions.addArgument('PETSc', 'File creation', 'Generated list of jobs for testing in '+os.path.join(self.bmakeDir,'jobs'))
    if os.path.isfile(os.path.join(self.bmakeDir, 'ejobs')):
      try:
        os.unlink(os.path.join(self.bmakeDir, 'ejobs'))
      except:
        raise RuntimeError('Unable to remove file '+os.path.join(self.bmakeDir, 'ejobs')+'. Did a different user create it?')
    ejobsFile = file(os.path.abspath(os.path.join(self.bmakeDir, 'ejobs')), 'w')
    ejobsFile.write(' '.join(ejobs)+'\n')
    ejobsFile.close()
    self.framework.actions.addArgument('PETSc', 'File creation', 'Generated list of jobs for testing in '+os.path.join(self.bmakeDir,'ejobs'))
    if os.path.isfile(os.path.join(self.bmakeDir, 'rjobs')):
      try:
        os.unlink(os.path.join(self.bmakeDir, 'rjobs'))
      except:
        raise RuntimeError('Unable to remove file '+os.path.join(self.bmakeDir, 'rjobs')+'. Did a different user create it?')
    rjobsFile = file(os.path.abspath(os.path.join(self.bmakeDir, 'rjobs')), 'w')
    rjobsFile.write(' '.join(rjobs)+'\n')
    rjobsFile.close()
    self.framework.actions.addArgument('PETSc', 'File creation', 'Generated list of jobs for testing in '+os.path.join(self.bmakeDir,'rjobs'))
    return

  def configureScript(self):
    '''Output a script in the bmake directory which will reproduce the configuration'''
    import nargs

    scriptName = os.path.join(self.bmakeDir, 'configure.py')
    args = filter(lambda a: not a.endswith('-configModules=PETSc.Configure') , self.framework.clArgs)
    if not nargs.Arg.findArgument('PETSC_ARCH', args):
      args.append('-PETSC_ARCH='+self.framework.argDB['PETSC_ARCH'])
    f = file(scriptName, 'w')
    f.write('#!/usr/bin/env python\n')
    f.write('if __name__ == \'__main__\':\n')
    f.write('  import sys\n')
    f.write('  sys.path.insert(0, '+repr(os.path.join(self.framework.argDB['PETSC_DIR'], 'config'))+')\n')
    f.write('  import configure\n')
    f.write('  configure_options = '+repr(args)+'\n')
    f.write('  configure.petsc_configure(configure_options)\n')
    f.close()
    os.chmod(scriptName, 0775)
    self.framework.actions.addArgument('PETSc', 'File creation', 'Created '+scriptName+' for automatic reconfiguration')
    return

  def configureInstall(self):
    '''Setup the directories for installation'''
    if self.framework.argDB['prefix']:
      self.framework.addSubstitution('INSTALL_DIR', os.path.join(self.framework.argDB['prefix'], os.path.basename(os.getcwd())))
    else:
      self.framework.addSubstitution('INSTALL_DIR', self.framework.argDB['PETSC_DIR'])
    return

  def configure(self):
    self.framework.header  = 'bmake/'+self.framework.argDB['PETSC_ARCH']+'/petscconf.h'
    self.framework.cHeader = 'bmake/'+self.framework.argDB['PETSC_ARCH']+'/petscfix.h'
    self.framework.addSubstitutionFile('bmake/config/packages.in',   'bmake/'+self.framework.argDB['PETSC_ARCH']+'/packages')
    self.framework.addSubstitutionFile('bmake/config/rules.in',      'bmake/'+self.framework.argDB['PETSC_ARCH']+'/rules')
    self.framework.addSubstitutionFile('bmake/config/variables.in',  'bmake/'+self.framework.argDB['PETSC_ARCH']+'/variables')
    if self.framework.argDB['with-64-bit-ints']:
      self.addDefine('USE_64BIT_INT', 1)
    else:
      self.addDefine('USE_32BIT_INT', 1)
    self.executeTest(self.configureLibraryOptions)
    self.executeTest(self.configureFortranCPP)
    self.executeTest(self.configureFortranCommandline)
    self.executeTest(self.configureMPIUNI)
    self.executeTest(self.configureDynamicLibraries)
    self.executeTest(self.configureLibtool)
    self.executeTest(self.configureDebuggers)
    self.executeTest(self.configureMkdir)
    self.executeTest(self.configurePrograms)
    self.executeTest(self.configureMissingDefines)
    self.executeTest(self.configureMissingFunctions)
    self.executeTest(self.configureMissingSignals)
    self.executeTest(self.configureMemorySize)
    self.executeTest(self.configureFPTrap)
    self.executeTest(self.configureGetDomainName)
    self.executeTest(self.configureIRIX)
    self.executeTest(self.configureSolaris)
    self.executeTest(self.configureLinux)
    self.executeTest(self.configureWin32)
    self.executeTest(self.configureMissingPrototypes)
    self.executeTest(self.configureMachineInfo)
    if self.framework.argDB['with-etags']:                                    
      self.executeTest(self.configureETags)
    self.bmakeDir = os.path.join('bmake', self.framework.argDB['PETSC_ARCH'])
    if not os.path.exists(self.bmakeDir):
      os.makedirs(self.bmakeDir)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created '+self.bmakeDir+' for configuration data')
    self.executeTest(self.configureRegression)
    self.executeTest(self.configureScript)
    self.executeTest(self.configureInstall)
    self.framework.log.write('================================================================================\n')
    self.logClear()
    return
