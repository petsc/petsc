import config.base

import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.defineAutoconfMacros()
    headersC = map(lambda name: name+'.h', ['dos', 'endian', 'fcntl', 'float', 'io', 'limits', 'malloc', 'pwd', 'search', 'strings',
                                            'stropts', 'unistd', 'machine/endian', 'sys/param', 'sys/procfs', 'sys/resource',
                                            'sys/stat', 'sys/systeminfo', 'sys/times', 'sys/utsname','string', 'stdlib',
                                            'sys/socket','sys/wait','netinet/in','netdb','Direct','time','Ws2tcpip','sys/types',
                                            'WindowsX'])
    functions = ['access', '_access', 'clock', 'drand48', 'getcwd', '_getcwd', 'getdomainname', 'gethostname', 'getpwuid',
                 'gettimeofday', 'getwd', 'memalign', 'memmove', 'mkstemp', 'popen', 'PXFGETARG', 'rand',
                 'readlink', 'realpath',  'sigaction', 'signal', 'sigset', 'sleep', '_sleep', 'socket', 'times',
                 'uname','snprintf','_snprintf','_fullpath','lseek','_lseek','time','fork','stricmp','bzero','dlopen','dlsym','erf']
    libraries1 = [(['socket', 'nsl'], 'socket')]
    self.setCompilers = self.framework.require('config.setCompilers',      self)
    self.framework.require('PETSc.utilities.arch', self.setCompilers)
    self.compilers    = self.framework.require('config.compilers',         self)
    self.framework.require('PETSc.utilities.compilerFlags', self.compilers)
    self.types        = self.framework.require('config.types',             self)
    self.headers      = self.framework.require('config.headers',           self)
    self.functions    = self.framework.require('config.functions',         self)
    self.libraries    = self.framework.require('config.libraries',         self)
    self.arch         = self.framework.require('PETSc.utilities.arch',     self)
    self.bmake        = self.framework.require('PETSc.utilities.bmakeDir', self)
    self.dynamic      = self.framework.require('PETSc.utilities.dynamicLibraries', self)        
    self.x11          = self.framework.require('PETSc.packages.X11',       self)
    self.compilers.headerPrefix = self.headerPrefix
    self.types.headerPrefix     = self.headerPrefix
    self.headers.headerPrefix   = self.headerPrefix
    self.functions.headerPrefix = self.headerPrefix
    self.libraries.headerPrefix = self.headerPrefix
    self.headers.headers.extend(headersC)
    self.functions.functions.extend(functions)
    self.libraries.libraries.extend(libraries1)

    import PETSc.packages
    import PETSc.utilities    

    for utility in os.listdir(os.path.join('python','PETSc','utilities')):
      (utilityName, ext) = os.path.splitext(utility)
      if not utilityName.startswith('.') and not utilityName.startswith('#') and ext == '.py' and not utilityName == '__init__':
        utilityObj              = self.framework.require('PETSc.utilities.'+utilityName, self)
        utilityObj.headerPrefix = self.headerPrefix
        setattr(self, utilityName.lower(), utilityObj)

    for package in os.listdir(os.path.join('python','PETSc','packages')):
      (packageName, ext) = os.path.splitext(package)
      if not packageName.startswith('.') and not packageName.startswith('#') and ext == '.py' and not packageName == '__init__':
        packageObj              = self.framework.require('PETSc.packages.'+packageName, self)
        packageObj.headerPrefix = self.headerPrefix
        setattr(self, packageName.lower(), packageObj)

    # List of packages actually found
    self.framework.packages = []
    return

  def __str__(self):
    return ''
                              
  def setupHelp(self, help):
    import nargs

    help.addArgument('PETSc', '-prefix=<path>',                nargs.Arg(None, '',     'Specifiy location to install PETSc (eg. /usr/local)'))
    help.addArgument('PETSc', '-with-external-packages=<bool>',nargs.ArgBool(None, 1, 'Allow external packages like Spooles, ParMetis, etc'))        
    return

  def defineAutoconfMacros(self):
    self.hostMacro = 'dnl Version: 2.13\ndnl Variable: host_cpu\ndnl Variable: host_vendor\ndnl Variable: host_os\nAC_CANONICAL_HOST'
    return



  def configurePIC(self):
    '''Determine the PIC option for each compiler
       - There needs to be a test that checks that the functionality is actually working'''
    if not self.dynamic.useDynamic:
      return
    if self.framework.argDB['PETSC_ARCH_BASE'].startswith('hpux') and not config.setCompilers.Configure.isGNU(self.framework.argDB['CC']):
      return
    languages = ['C']
    if 'CXX' in self.framework.argDB:
      languages.append('C++')
    if 'FC' in self.framework.argDB:
      languages.append('F77')
    for language in languages:
      self.pushLanguage(language)
      for testFlag in ['-PIC', '-fPIC', '-KPIC']:
        try:
          self.framework.log.write('Trying '+language+' compiler flag '+testFlag+'\n')
          self.addCompilerFlag(testFlag)
          break
        except RuntimeError:
          self.framework.log.write('Rejected '+language+' compiler flag '+testFlag+'\n')
      self.popLanguage()
    return

    
  def configureBmake(self):
    ''' Actually put the values into the bmake files '''
    # eventually this will be gone
    
    # archive management tools
    self.addMakeMacro('AR_FLAGS  ',      self.setCompilers.AR_FLAGS)
    self.addMakeMacro('AR_LIB_SUFFIX ',    self.libraries.suffix)
    self.addMakeMacro('RANLIB ',         self.setCompilers.RANLIB)

    # C preprocessor values
    self.addMakeMacro('CPP_FLAGS',self.setCompilers.CPPFLAGS)
    
    # compiler values
    self.setCompilers.pushLanguage('C')
    self.addMakeMacro('CC',self.setCompilers.getCompiler())
    self.addMakeMacro('CC_FLAGS',self.setCompilers.getCompilerFlags())    
    self.setCompilers.popLanguage()
    # .o or .obj 
    self.addMakeMacro('CC_SUFFIX','o')

    # executable linker values
    self.setCompilers.pushLanguage('C')
    self.addMakeMacro('CC_LINKER',self.setCompilers.getLinker())
    self.addMakeMacro('CC_LINKER',self.setCompilers.getLinker())
    self.addMakeMacro('CC_LINKER_FLAGS',self.setCompilers.getLinkerFlags())
    self.setCompilers.popLanguage()
    # -rpath or -R or -L etc
    if not self.setCompilers.CSharedLinkerFlag: value = '-L'
    else: value = self.setCompilers.CSharedLinkerFlag
    self.addMakeMacro('CC_LINKER_SLFLAG',value)    
    # '' for Unix, .exe for Windows
    self.addMakeMacro('CC_LINKER_SUFFIX','')
    self.addMakeMacro('CC_LINKER_LIBS',self.framework.argDB['LIBS']+' '+self.compilers.flibs)    

    # shared library linker values
    self.setCompilers.pushLanguage('C')
    # need to fix BuildSystem to collect these seperately
    self.addMakeMacro('SL_LINKER',self.setCompilers.getLinker())
    self.addMakeMacro('SL_LINKER_FLAGS',self.setCompilers.getLinkerFlags())
    self.setCompilers.popLanguage()
    # '' for Unix, .exe for Windows
    self.addMakeMacro('SL_LINKER_SUFFIX','.so')
    self.addMakeMacro('SL_LINKER_LIBS',self.framework.argDB['LIBS']+' '+self.compilers.flibs)    
    
    # CONLY or CPP
    self.addMakeMacro('PETSC_LANGUAGE','CONLY')
    # real or complex
    self.addMakeMacro('PETSC_SCALAR','real')
    # double or float
    self.addMakeMacro('PETSC_PRECISION','double')

    # print include and lib for external packages
    for i in self.framework.packages:
      self.addDefine('HAVE_'+i.PACKAGE,1)
      if not isinstance(i.lib,list): i.lib = [i.lib]
      self.addMakeMacro(i.PACKAGE+'_LIB',' '.join(map(self.libraries.getLibArgument, i.lib)))
      if hasattr(i,'include'):
        if not isinstance(i.include,list): i.include = [i.include]      
        self.addMakeMacro(i.PACKAGE+'_INCLUDE',' '.join(map(self.libraries.getIncludeArgument, i.include)))
    text = ''
    for i in self.framework.packages:
      text += '${'+i.PACKAGE+'_LIB} '
    self.addMakeMacro('PACKAGES_LIBS',text)
    
    self.addMakeMacro('INSTALL_DIR',self.installdir)
    self.addMakeMacro('top_builddir',self.installdir)                

      
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
      except RuntimeError: pass
      if os.path.exists('.conftest'): os.removedirs('.conftest/.tmp')
    return

  def configurePrograms(self):
    '''Check for the programs needed to build and run PETSc'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.getExecutable('sh',   getFullPath = 1, resultName = 'SHELL')
    self.framework.getExecutable('sed',  getFullPath = 1)
    self.framework.getExecutable('mv',   getFullPath = 1)
    self.framework.getExecutable('diff', getFullPath = 1)
    self.framework.getExecutable('rm -f',getFullPath = 1)
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
      self.framework.diff = self.framework.diff + ' -w'
      
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
    


  def configureMachineInfo(self):
    '''Define a string incorporating all configuration data needed for a bug report'''
    #self.addDefine('MACHINE_INFO', '"Libraries compiled on `date` on `hostname`\\nMachine characteristics: `uname -a`\\n-----------------------------------------\\nUsing C compiler: ${CC} ${COPTFLAGS} ${CCPPFLAGS}\\nC Compiler version: ${C_VERSION}\\nUsing C compiler: ${CXX} ${CXXOPTFLAGS} ${CXXCPPFLAGS}\\nC++ Compiler version: ${CXX_VERSION}\\nUsing Fortran compiler: ${FC} ${FOPTFLAGS} ${FCPPFLAGS}\\nFortran Compiler version: ${F_VERSION}\\n-----------------------------------------\\nUsing PETSc flags: ${PETSCFLAGS} ${PCONF}\\n-----------------------------------------\\nUsing include paths: ${PETSC_INCLUDE}\\n-----------------------------------------\\nUsing PETSc directory: ${PETSC_DIR}\\nUsing PETSc arch: ${PETSC_ARCH}"\\n')
    return



  def configureScript(self):
    '''Output a script in the bmake directory which will reproduce the configuration'''
    import nargs

    scriptName = os.path.join(self.bmake.bmakeDir, 'configure.py')
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
      self.installdir = os.path.join(self.framework.argDB['prefix'], os.path.basename(os.getcwd()))
    else:
      self.installdir = self.framework.argDB['PETSC_DIR']
    return

  def configure(self):
    self.framework.header          = 'bmake/'+self.framework.argDB['PETSC_ARCH']+'/petscconf.h'
    self.framework.cHeader         = 'bmake/'+self.framework.argDB['PETSC_ARCH']+'/petscfix.h'
    self.framework.makeMacroHeader = 'bmake/'+self.framework.argDB['PETSC_ARCH']+'/petscconf'
    self.framework.makeRuleHeader  = 'bmake/'+self.framework.argDB['PETSC_ARCH']+'/petscrules'        
    self.executeTest(self.configurePIC)
    self.executeTest(self.configureDebuggers)
    self.executeTest(self.configureMkdir)
    self.executeTest(self.configurePrograms)
    self.executeTest(self.configureMissingDefines)
    self.executeTest(self.configureMissingFunctions)
    self.executeTest(self.configureMissingSignals)
    self.executeTest(self.configureFPTrap)
    self.executeTest(self.configureGetDomainName)
    self.executeTest(self.configureSolaris)
    self.executeTest(self.configureLinux)
    self.executeTest(self.configureWin32)
    self.executeTest(self.configureMachineInfo)
    self.executeTest(self.configureScript)
    self.executeTest(self.configureInstall)
    self.executeTest(self.configureBmake)    
    self.framework.log.write('================================================================================\n')
    self.logClear()
    return
