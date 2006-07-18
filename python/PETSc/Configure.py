import config.base

import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.defineAutoconfMacros()
    # List of packages actually found
    self.framework.packages = []
    return

  def __str__(self):
    return ''
                              
  def setupHelp(self, help):
    import nargs

    help.addArgument('PETSc', '-prefix=<path>',            nargs.Arg(None, '', 'Specifiy location to install PETSc (eg. /usr/local)'))
    help.addArgument('PETSc', '-with-default-arch=<bool>', nargs.ArgBool(None, 1, 'Allow using the last configured arch without setting PETSC_ARCH'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers  = framework.require('config.setCompilers',      self)
    self.arch          = framework.require('PETSc.utilities.arch',     self.setCompilers)
    self.petscdir      = framework.require('PETSc.utilities.petscdir', self.setCompilers)
    self.languages     = framework.require('PETSc.utilities.languages',self.setCompilers)
    self.debugging     = framework.require('PETSc.utilities.debugging',self.setCompilers)        
    self.compilers     = framework.require('config.compilers',         self)
    self.types         = framework.require('config.types',             self)
    self.headers       = framework.require('config.headers',           self)
    self.functions     = framework.require('config.functions',         self)
    self.libraries     = framework.require('config.libraries',         self)
    if os.path.isdir(os.path.join('python', 'PETSc')):
      for d in ['utilities', 'packages']:
        for utility in os.listdir(os.path.join('python', 'PETSc', d)):
          (utilityName, ext) = os.path.splitext(utility)
          if not utilityName.startswith('.') and not utilityName.startswith('#') and ext == '.py' and not utilityName == '__init__':
            utilityObj              = self.framework.require('PETSc.'+d+'.'+utilityName, self)
            utilityObj.headerPrefix = self.headerPrefix
            ##utilityObj.languageProvider = self.languages
            setattr(self, utilityName.lower(), utilityObj)
    self.blaslapack    = framework.require('config.packages.BlasLapack', self)
    self.blaslapack.archProvider      = self.arch
    self.blaslapack.precisionProvider = self.scalartypes
    self.mpi           = framework.require('config.packages.MPI',        self)
    self.mpi.archProvider             = self.arch
    self.mpi.languageProvider         = self.languages

    self.compilers.headerPrefix = self.headerPrefix
    self.types.headerPrefix     = self.headerPrefix
    self.headers.headerPrefix   = self.headerPrefix
    self.functions.headerPrefix = self.headerPrefix
    self.libraries.headerPrefix = self.headerPrefix
    headersC = map(lambda name: name+'.h', ['dos', 'endian', 'fcntl', 'float', 'io', 'limits', 'malloc', 'pwd', 'search', 'strings',
                                            'unistd', 'machine/endian', 'sys/param', 'sys/procfs', 'sys/resource',
                                            'sys/systeminfo', 'sys/times', 'sys/utsname','string', 'stdlib',
                                            'sys/socket','sys/wait','netinet/in','netdb','Direct','time','Ws2tcpip','sys/types',
                                            'WindowsX', 'cxxabi'])
    functions = ['access', '_access', 'clock', 'drand48', 'getcwd', '_getcwd', 'getdomainname', 'gethostname', 'getpwuid',
                 'gettimeofday', 'getwd', 'memalign', 'memmove', 'mkstemp', 'popen', 'PXFGETARG', 'rand', 'getpagesize',
                 'readlink', 'realpath',  'sigaction', 'signal', 'sigset', 'sleep', '_sleep', 'socket', 'times', 'gethostbyname',
                 'uname','snprintf','_snprintf','_fullpath','lseek','_lseek','time','fork','stricmp','bzero','dlerror',
                 '_intel_fast_memcpy','_intel_fast_memset']
    libraries1 = [(['socket', 'nsl'], 'socket'), (['fpe'], 'handle_sigfpes')]
    self.headers.headers.extend(headersC)
    self.functions.functions.extend(functions)
    self.libraries.libraries.extend(libraries1)
    return

  def defineAutoconfMacros(self):
    self.hostMacro = 'dnl Version: 2.13\ndnl Variable: host_cpu\ndnl Variable: host_vendor\ndnl Variable: host_os\nAC_CANONICAL_HOST'
    return
    
  def Dump(self):
    ''' Actually put the values into the bmake files '''
    # eventually everything between -- should be gone
#-----------------------------------------------------------------------------------------------------    

    # Sometimes we need C compiler, even if built with C++
    self.setCompilers.pushLanguage('C')
    self.addMakeMacro('CC_FLAGS',self.setCompilers.getCompilerFlags())    
    self.setCompilers.popLanguage()

    # C preprocessor values
    self.addMakeMacro('CPP_FLAGS',self.setCompilers.CPPFLAGS)
    
    # compiler values
    self.setCompilers.pushLanguage(self.languages.clanguage)
    self.addMakeMacro('PCC',self.setCompilers.getCompiler())
    self.addMakeMacro('PCC_FLAGS',self.setCompilers.getCompilerFlags())    
    self.setCompilers.popLanguage()
    # .o or .obj 
    self.addMakeMacro('CC_SUFFIX','o')

    # executable linker values
    self.setCompilers.pushLanguage(self.languages.clanguage)
    pcc_linker = self.setCompilers.getLinker()
    self.addMakeMacro('PCC_LINKER',pcc_linker)
    self.addMakeMacro('PCC_LINKER_FLAGS',self.setCompilers.getLinkerFlags())
    self.setCompilers.popLanguage()
    # '' for Unix, .exe for Windows
    self.addMakeMacro('CC_LINKER_SUFFIX','')
    self.addMakeMacro('PCC_LINKER_LIBS',self.libraries.toString(self.compilers.flibs)+' '+self.libraries.toString(self.compilers.cxxlibs)+' '+self.compilers.LIBS)

    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      # need FPPFLAGS in config/setCompilers
      self.addDefine('HAVE_FORTRAN','1')
      self.addMakeMacro('FPP_FLAGS',self.setCompilers.CPPFLAGS)
    
      # compiler values
      self.addMakeMacro('FC_FLAGS',self.setCompilers.getCompilerFlags())    
      self.setCompilers.popLanguage()
      # .o or .obj 
      self.addMakeMacro('FC_SUFFIX','o')

      # executable linker values
      self.setCompilers.pushLanguage('FC')
      # Cannot have NAG f90 as the linker - so use pcc_linker as fc_linker
      fc_linker = self.setCompilers.getLinker()
      if config.setCompilers.Configure.isNAG(fc_linker):
        self.addMakeMacro('FC_LINKER',pcc_linker)
      else:
        self.addMakeMacro('FC_LINKER',fc_linker)
      self.addMakeMacro('FC_LINKER_FLAGS',self.setCompilers.getLinkerFlags())
      self.setCompilers.popLanguage()
      # '' for Unix, .exe for Windows
      self.addMakeMacro('FC_LINKER_SUFFIX','')
      self.addMakeMacro('FC_LINKER_LIBS',' '.join([self.libraries.getLibArgument(lib) for lib in self.compilers.flibs])+' '+self.compilers.LIBS)
    else:
      self.addMakeMacro('FC','')

    # shared library linker values
    self.setCompilers.pushLanguage(self.languages.clanguage)
    # need to fix BuildSystem to collect these separately
    self.addMakeMacro('SL_LINKER',self.setCompilers.getLinker())
    self.addMakeMacro('SL_LINKER_FLAGS',self.setCompilers.getLinkerFlags())
    self.setCompilers.popLanguage()
    # One of 'a', 'so', 'lib', 'dll', 'dylib' (perhaps others also?) depending on the library generator and architecture
    # Note: . is not included in this macro, consistent with AR_LIB_SUFFIX
    if self.setCompilers.sharedLibraryExt == self.setCompilers.AR_LIB_SUFFIX:
      self.addMakeMacro('SL_LINKER_SUFFIX', '')
    else:
      self.addMakeMacro('SL_LINKER_SUFFIX', self.setCompilers.sharedLibraryExt)
    if self.setCompilers.isDarwin() and self.languages.clanguage == 'Cxx':
      self.addMakeMacro('SL_LINKER_LIBS',' '.join([self.libraries.getLibArgument(lib) for lib in self.compilers.flibs])+' '+' '.join([self.libraries.getLibArgument(lib) for lib in self.compilers.cxxlibs])+' '+self.compilers.LIBS)
    else:
      self.addMakeMacro('SL_LINKER_LIBS',' '.join([self.libraries.getLibArgument(lib) for lib in self.compilers.flibs])+' '+ self.compilers.LIBS)
#-----------------------------------------------------------------------------------------------------

    # CONLY or CPP. We should change the PETSc makefiles to do this better
    if self.languages.clanguage == 'C': lang = 'CONLY'
    else: lang = 'CXXONLY'
    self.addMakeMacro('PETSC_LANGUAGE',lang)

    if self.python.usePython:
      self.addMakeMacro('PYTHON_INCLUDE', ' '.join([self.headers.getIncludeArgument(inc) for inc in self.python.python.include]))
      self.addMakeMacro('PYTHON_LIB', ' '.join([self.libraries.getLibArgument(lib) for lib in self.python.python.lib]))
    
    # real or complex
    self.addMakeMacro('PETSC_SCALAR',self.scalartypes.scalartype)
    # double or float
    self.addMakeMacro('PETSC_PRECISION',self.languages.precision)

    if self.framework.argDB['with-batch']:
      self.addMakeMacro('PETSC_WITH_BATCH','1')

    # Test for compiler-specific macros that need to be defined.
    if self.setCompilers.isCray('CC'):
      self.addDefine('HAVE_CRAYC','1')

#-----------------------------------------------------------------------------------------------------
    if self.functions.haveFunction('gethostbyname') and self.functions.haveFunction('socket'):
      self.addDefine('USE_SOCKET_VIEWER','1')

#-----------------------------------------------------------------------------------------------------
    # print include and lib for external packages
    self.framework.packages.reverse()
    for i in self.framework.packages:
      self.addDefine('HAVE_'+i.PACKAGE, 1)
      if not isinstance(i.lib, list):
        i.lib = [i.lib]
      self.addMakeMacro(i.PACKAGE+'_LIB', ' '.join([self.libraries.getLibArgument(l) for l in i.lib]))
      if hasattr(i,'include'):
        if not isinstance(i.include,list):
          i.include = [i.include]
        self.addMakeMacro(i.PACKAGE+'_INCLUDE', ' '.join([self.headers.getIncludeArgument(inc) for inc in i.include]))
    self.addMakeMacro('PACKAGES_LIBS',' '.join(['${'+package.PACKAGE+'_LIB}' for package in self.framework.packages]+[self.libraries.getLibArgument(l) for l in self.libraries.math]))
    self.addMakeMacro('PACKAGES_INCLUDES',' '.join(['${'+p.PACKAGE+'_INCLUDE}' for p in self.framework.packages if hasattr(p,'include')]))
    
    self.addMakeMacro('INSTALL_DIR',self.installdir)
    self.addMakeMacro('top_builddir',self.installdir)                

    if not os.path.exists(os.path.join(self.petscdir.dir,'lib')):
      os.makedirs(os.path.join(self.petscdir.dir,'lib'))

    # add a makefile entry for configure options
    self.addMakeMacro('CONFIGURE_OPTIONS', self.framework.getOptionsString(['configModules', 'optionsModule']).replace('\"','\\"'))
    return

  def dumpConfigInfo(self):
    import time
    fd = file(os.path.join('bmake',self.arch.arch,'petscconfiginfo.h'),'w')
    fd.write('static const char *petscconfigureruntime = "'+time.ctime(time.time())+'";\n')
    fd.write('static const char *petscconfigureoptions = "'+self.framework.getOptionsString(['configModules', 'optionsModule']).replace('\"','\\"')+'";\n')
    fd.close()
    return

  def configureInline(self):
    '''Get a generic inline keyword, depending on the language'''
    if self.languages.clanguage == 'C':
      self.addDefine('STATIC_INLINE', self.compilers.cStaticInlineKeyword)
      self.addDefine('RESTRICT', self.compilers.cRestrict)
    elif self.languages.clanguage == 'Cxx':
      self.addDefine('STATIC_INLINE', self.compilers.cxxStaticInlineKeyword)
      self.addDefine('RESTRICT', self.compilers.cxxRestrict)
    return

  def configureSolaris(self):
    '''Solaris specific stuff'''
    if self.arch.hostOsBase.startswith('solaris'):
      if os.path.isdir(os.path.join('/usr','ucblib')):
        try:
          flag = getattr(self.setCompilers, self.language[-1]+'SharedLinkerFlag')
        except AttributeError:
          flag = None
        if flag is None:
          self.compilers.LIBS += ' -L/usr/ucblib'
        else:
          self.compilers.LIBS += ' '+flag+'/usr/ucblib'
    return

  def configureLinux(self):
    '''Linux specific stuff'''
    if self.arch.hostOsBase == 'linux':
      self.addDefine('HAVE_DOUBLE_ALIGN_MALLOC', 1)
    return

  def configureWin32(self):
    '''Win32 non-cygwin specific stuff'''
    kernel32=0
    if self.libraries.add('Kernel32.lib','GetComputerName',prototype='#include <Windows.h>', call='GetComputerName(NULL,NULL);'):
      self.addDefine('HAVE_WINDOWS_H',1)
      self.addDefine('HAVE_GETCOMPUTERNAME',1)
      kernel32=1
    elif self.libraries.add('kernel32','GetComputerName',prototype='#include <Windows.h>', call='GetComputerName(NULL,NULL);'):
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
    if self.libraries.add('Advapi32.lib','GetUserName',prototype='#include <Windows.h>', call='GetUserName(NULL,NULL);'):
      self.addDefine('HAVE_GET_USER_NAME',1)
    elif self.libraries.add('advapi32','GetUserName',prototype='#include <Windows.h>', call='GetUserName(NULL,NULL);'):
      self.addDefine('HAVE_GET_USER_NAME',1)
        
    if not self.libraries.add('User32.lib','GetDC',prototype='#include <Windows.h>',call='GetDC(0);'):
      self.libraries.add('user32','GetDC',prototype='#include <Windows.h>',call='GetDC(0);')
    if not self.libraries.add('Gdi32.lib','CreateCompatibleDC',prototype='#include <Windows.h>',call='CreateCompatibleDC(0);'):
      self.libraries.add('gdi32','CreateCompatibleDC',prototype='#include <Windows.h>',call='CreateCompatibleDC(0);')
      
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

    if self.compilers.CC.find('win32fe') >= 0:
      self.addDefine('PATH_SEPARATOR','\';\'')
      self.addDefine('DIR_SEPARATOR','\'\\\\\'')
      self.addDefine('REPLACE_DIR_SEPARATOR','\'/\'')
      self.addDefine('CANNOT_START_DEBUGGER',1)
    else:
      self.addDefine('PATH_SEPARATOR','\':\'')
      self.addDefine('REPLACE_DIR_SEPARATOR','\'\\\\\'')
      self.addDefine('DIR_SEPARATOR','\'/\'')
    return

#-----------------------------------------------------------------------------------------------------
  def configureDefaults(self):
    if self.framework.argDB['with-default-arch']:
      fd = file(os.path.join('bmake', 'petscconf'), 'w')
      fd.write('PETSC_ARCH='+self.arch.arch+'\n')
      fd.write('include '+os.path.join('${PETSC_DIR}','bmake',self.arch.arch,'petscconf')+'\n')
      fd.close()
      self.framework.actions.addArgument('PETSc', 'Build', 'Set default architecture to '+self.arch.arch+' in bmake/petscconf')
    elif os.path.isfile(os.path.join('bmake', 'petscconf')):
      try:
        os.unlink(os.path.join('bmake', 'petscconf'))
      except:
        raise RuntimeError('Unable to remove file '+os.path.join('bmake', 'petscconf')+'. Did a different user create it?')
    return

  def configureScript(self):
    '''Output a script in the bmake directory which will reproduce the configuration'''
    import nargs

    scriptName = os.path.join(self.bmakedir.bmakeDir, 'configure.py')
    args = dict([(nargs.Arg.parseArgument(arg)[0], arg) for arg in self.framework.clArgs])
    if 'configModules' in args:
      del args['configModules']
    if 'optionsModule' in args:
      del args['optionsModule']
    if not 'PETSC_ARCH' in args:
      args['PETSC_ARCH'] = '-PETSC_ARCH='+str(self.arch.arch)
    f = file(scriptName, 'w')
    f.write('#!/usr/bin/env python\n')
    f.write('if __name__ == \'__main__\':\n')
    f.write('  import sys\n')
    f.write('  sys.path.insert(0, '+repr(os.path.join(self.petscdir.dir, 'config'))+')\n')
    f.write('  import configure\n')
    f.write('  configure_options = '+repr(args.values())+'\n')
    f.write('  configure.petsc_configure(configure_options)\n')
    f.close()
    try:
      os.chmod(scriptName, 0775)
    except OSError, e:
      self.framework.logPrint('Unable to make reconfigure script executable:\n'+str(e))
    self.framework.actions.addArgument('PETSc', 'File creation', 'Created '+scriptName+' for automatic reconfiguration')
    return

  def configureInstall(self):
    '''Setup the directories for installation'''
    if self.framework.argDB['prefix']:
      self.installdir = self.framework.argDB['prefix']
    else:
      self.installdir = self.petscdir.dir
    return

  def configureGCOV(self):
    if self.framework.argDB['with-gcov']:
      self.addDefine('USE_GCOV','1')
    return

  def configure(self):
    if not os.path.samefile(self.petscdir.dir, os.getcwd()):
      raise RuntimeError('Wrong PETSC_DIR option specified: '+str(self.petscdir.dir) + '\n  Configure invoked in: '+os.path.realpath(os.getcwd()))
    self.framework.header          = 'bmake/'+self.arch.arch+'/petscconf.h'
    self.framework.cHeader         = 'bmake/'+self.arch.arch+'/petscfix.h'
    self.framework.makeMacroHeader = 'bmake/'+self.arch.arch+'/petscconf'
    self.framework.makeRuleHeader  = 'bmake/'+self.arch.arch+'/petscrules'
    if self.libraries.math is None:
      raise RuntimeError('PETSc requires a functional math library. Please send configure.log to petsc-maint@mcs.anl.gov.')
    if self.languages.clanguage == 'Cxx' and not hasattr(self.compilers, 'CXX'):
      raise RuntimeError('Cannot set C language to C++ without a functional C++ compiler.')
    self.executeTest(self.configureInline)
    self.executeTest(self.configureSolaris)
    self.executeTest(self.configureLinux)
    self.executeTest(self.configureWin32)
    self.executeTest(self.configureDefaults)
    self.executeTest(self.configureScript)
    self.executeTest(self.configureInstall)
    self.executeTest(self.configureGCOV)
    self.Dump()
    self.dumpConfigInfo()
    self.framework.log.write('================================================================================\n')
    self.logClear()
    return
