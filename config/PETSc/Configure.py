import config.base

import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.defineAutoconfMacros()
    return

  def __str__(self):
    return ''
                              
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc',  '-prefix=<path>',                  nargs.Arg(None, '', 'Specifiy location to install PETSc (eg. /usr/local)'))
    help.addArgument('Windows','-with-windows-graphics=<bool>',   nargs.ArgBool(None, 1,'Enable check for Windows Graphics'))
    help.addArgument('PETSc','-with-single-library=<bool>',       nargs.ArgBool(None, 0,'Put all PETSc code into the single -lpetsc library'))

    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers  = framework.require('config.setCompilers',      self)
    self.arch          = framework.require('PETSc.utilities.arch',     self.setCompilers)
    self.petscdir      = framework.require('PETSc.utilities.petscdir', self.setCompilers)
    self.languages     = framework.require('PETSc.utilities.languages',self.setCompilers)
    self.debugging     = framework.require('PETSc.utilities.debugging',self.setCompilers)
    self.make          = framework.require('PETSc.utilities.Make',     self)
    self.CHUD          = framework.require('PETSc.utilities.CHUD',     self)        
    self.compilers     = framework.require('config.compilers',         self)
    self.types         = framework.require('config.types',             self)
    self.headers       = framework.require('config.headers',           self)
    self.functions     = framework.require('config.functions',         self)
    self.libraries     = framework.require('config.libraries',         self)
    if os.path.isdir(os.path.join('config', 'PETSc')):
      for d in ['utilities', 'packages']:
        for utility in os.listdir(os.path.join('config', 'PETSc', d)):
          (utilityName, ext) = os.path.splitext(utility)
          if not utilityName.startswith('.') and not utilityName.startswith('#') and ext == '.py' and not utilityName == '__init__':
            utilityObj              = self.framework.require('PETSc.'+d+'.'+utilityName, self)
            utilityObj.headerPrefix = self.headerPrefix
            ##utilityObj.languageProvider = self.languages
            setattr(self, utilityName.lower(), utilityObj)
    self.blaslapack    = framework.require('config.packages.BlasLapack', self)
    self.blaslapack.archProvider      = self.arch
    self.blaslapack.precisionProvider = self.scalartypes
    self.blaslapack.installDirProvider= self.petscdir
    self.mpi           = framework.require('config.packages.MPI',        self)
    self.mpi.archProvider             = self.arch
    self.mpi.languageProvider         = self.languages
    self.mpi.installDirProvider       = self.petscdir
    self.umfpack       = framework.require('config.packages.UMFPACK',    self)
    self.umfpack.archProvider         = self.arch
    self.umfpack.languageProvider     = self.languages
    self.umfpack.installDirProvider   = self.petscdir
    self.Boost         = framework.require('config.packages.Boost',      self)
    self.Boost.archProvider           = self.arch
    self.Boost.languageProvider       = self.languages
    self.Boost.installDirProvider     = self.petscdir
    self.Fiat          = framework.require('config.packages.Fiat',       self)
    self.Fiat.archProvider            = self.arch
    self.Fiat.languageProvider        = self.languages
    self.Fiat.installDirProvider      = self.petscdir

    self.compilers.headerPrefix = self.headerPrefix
    self.types.headerPrefix     = self.headerPrefix
    self.headers.headerPrefix   = self.headerPrefix
    self.functions.headerPrefix = self.headerPrefix
    self.libraries.headerPrefix = self.headerPrefix
    self.blaslapack.headerPrefix = self.headerPrefix
    self.mpi.headerPrefix        = self.headerPrefix
    headersC = map(lambda name: name+'.h', ['dos', 'endian', 'fcntl', 'float', 'io', 'limits', 'malloc', 'pwd', 'search', 'strings',
                                            'unistd', 'machine/endian', 'sys/param', 'sys/procfs', 'sys/resource',
                                            'sys/systeminfo', 'sys/times', 'sys/utsname','string', 'stdlib','memory',
                                            'sys/socket','sys/wait','netinet/in','netdb','Direct','time','Ws2tcpip','sys/types',
                                            'WindowsX', 'cxxabi','float.h','ieeefp','xmmintrin.h'])
    functions = ['access', '_access', 'clock', 'drand48', 'getcwd', '_getcwd', 'getdomainname', 'gethostname', 'getpwuid',
                 'gettimeofday', 'getwd', 'memalign', 'memmove', 'mkstemp', 'popen', 'PXFGETARG', 'rand', 'getpagesize',
                 'readlink', 'realpath',  'sigaction', 'signal', 'sigset', 'sleep', '_sleep', 'socket', 'times', 'gethostbyname',
                 'uname','snprintf','_snprintf','_fullpath','lseek','_lseek','time','fork','stricmp','strcasecmp','bzero',
                 'dlopen', 'dlsym', 'dlclose', 'dlerror',
                 '_intel_fast_memcpy','_intel_fast_memset','isinf','isnan','_finite','_isnan']
    libraries1 = [(['socket', 'nsl'], 'socket'), (['fpe'], 'handle_sigfpes')]
    self.headers.headers.extend(headersC)
    self.functions.functions.extend(functions)
    self.libraries.libraries.extend(libraries1)
    return

  def defineAutoconfMacros(self):
    self.hostMacro = 'dnl Version: 2.13\ndnl Variable: host_cpu\ndnl Variable: host_vendor\ndnl Variable: host_os\nAC_CANONICAL_HOST'
    return
    
  def Dump(self):
    ''' Actually put the values into the configuration files '''
    # eventually everything between -- should be gone
#-----------------------------------------------------------------------------------------------------    

    # Sometimes we need C compiler, even if built with C++
    self.setCompilers.pushLanguage('C')
    self.addMakeMacro('CC_FLAGS',self.setCompilers.getCompilerFlags())    
    self.setCompilers.popLanguage()

    # C preprocessor values
    self.addMakeMacro('CPP_FLAGS',self.setCompilers.CPPFLAGS+self.CHUD.CPPFLAGS)
    
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
    self.addMakeMacro('PCC_LINKER_LIBS',self.libraries.toStringNoDupes(self.compilers.flibs+self.compilers.cxxlibs+self.compilers.LIBS.split(' '))+self.CHUD.LIBS)

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
      # apple requires this shared library linker flag on SOME versions of the os
      if self.setCompilers.getLinkerFlags().find('-Wl,-commons,use_dylibs') > -1:
        self.addMakeMacro('DARWIN_COMMONS_USE_DYLIBS',' -Wl,-commons,use_dylibs ')
      self.setCompilers.popLanguage()
    else:
      self.addMakeMacro('FC','')

    # shared library linker values
    self.setCompilers.pushLanguage(self.languages.clanguage)
    # need to fix BuildSystem to collect these separately
    self.addMakeMacro('SL_LINKER',self.setCompilers.getLinker())
    self.addMakeMacro('SL_LINKER_FLAGS','${PCC_LINKER_FLAGS}')
    self.setCompilers.popLanguage()
    # One of 'a', 'so', 'lib', 'dll', 'dylib' (perhaps others also?) depending on the library generator and architecture
    # Note: . is not included in this macro, consistent with AR_LIB_SUFFIX
    if self.setCompilers.sharedLibraryExt == self.setCompilers.AR_LIB_SUFFIX:
      self.addMakeMacro('SL_LINKER_SUFFIX', '')
    else:
      self.addMakeMacro('SL_LINKER_SUFFIX', self.setCompilers.sharedLibraryExt)
    
    #SL_LINKER_LIBS is currently same as PCC_LINKER_LIBS - so simplify
    self.addMakeMacro('SL_LINKER_LIBS','${PCC_LINKER_LIBS}')
    #self.addMakeMacro('SL_LINKER_LIBS',self.libraries.toStringNoDupes(self.compilers.flibs+self.compilers.cxxlibs+self.compilers.LIBS.split(' ')))

#-----------------------------------------------------------------------------------------------------

    # CONLY or CPP. We should change the PETSc makefiles to do this better
    if self.languages.clanguage == 'C': lang = 'CONLY'
    else: lang = 'CXXONLY'
    self.addMakeMacro('PETSC_LANGUAGE',lang)

    # real or complex
    self.addMakeMacro('PETSC_SCALAR',self.scalartypes.scalartype)
    # double or float
    self.addMakeMacro('PETSC_PRECISION',self.scalartypes.precision)

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
    includes = []
    libs = []
    for i in self.framework.packages:
      self.addDefine('HAVE_'+i.PACKAGE, 1)
      if not isinstance(i.lib, list):
        i.lib = [i.lib]
      libs.extend(i.lib)
      self.addMakeMacro(i.PACKAGE+'_LIB', self.libraries.toStringNoDupes(i.lib))
      if hasattr(i,'include'):
        if not isinstance(i.include,list):
          i.include = [i.include]
        includes.extend(i.include)
        self.addMakeMacro(i.PACKAGE+'_INCLUDE',self.headers.toStringNoDupes(i.include))
    self.addMakeMacro('PACKAGES_LIBS',self.libraries.toStringNoDupes(libs+self.libraries.math))
    self.addMakeMacro('PACKAGES_INCLUDES',self.headers.toStringNoDupes(includes))
    
    self.addMakeMacro('INSTALL_DIR',self.installdir)

    if self.framework.argDB['with-single-library']:
      # overrides the values set in conf/variables
      self.addMakeMacro('LIBNAME','${INSTALL_LIB_DIR}/libpetsc.${AR_LIB_SUFFIX}')
      self.addMakeMacro('PETSC_SYS_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_VEC_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_MAT_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_DM_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_KSP_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_SNES_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_TS_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_CONTRIB_BASIC','-lpetsc')
      self.addDefine('USE_SINGLE_LIBRARY', '1')
      
    if not os.path.exists(os.path.join(self.petscdir.dir,self.arch.arch,'lib')):
      os.makedirs(os.path.join(self.petscdir.dir,self.arch.arch,'lib'))

    # add a makefile entry for configure options
    self.addMakeMacro('CONFIGURE_OPTIONS', self.framework.getOptionsString(['configModules', 'optionsModule']).replace('\"','\\"'))
    return

  def dumpConfigInfo(self):
    import time
    fd = file(os.path.join(self.arch.arch,'include','petscconfiginfo.h'),'w')
    fd.write('static const char *petscconfigureruntime = "'+time.ctime(time.time())+'";\n')
    fd.write('static const char *petscconfigureoptions = "'+self.framework.getOptionsString(['configModules', 'optionsModule']).replace('\"','\\"')+'";\n')
    fd.close()
    return


  def configurePrefetch(self):
    '''Sees if there are any prefetch functions supported'''
    if self.checkLink('#include <xmmintrin.h>', 'void *v = 0;_mm_prefetch(v,0);\n'):
      self.addDefine('Prefetch(a,b,c)', '_mm_prefetch((void*)(a),c)')
    elif self.checkLink('', 'void *v = 0;__builtin_prefetch(v,0,0);\n'):
      self.addDefine('Prefetch(a,b,c)', '__builtin_prefetch(a,b,c)')
    else:
      self.addDefine('Prefetch(a,b,c)', '')
      
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
      if self.framework.argDB['with-windows-graphics']:
        self.addDefine('USE_WINDOWS_GRAPHICS',1)
      if self.checkLink('#include <Windows.h>','LoadLibrary(0)'):
        self.addDefine('HAVE_LOADLIBRARY',1)
      if self.checkLink('#include <Windows.h>','GetProcAddress(0,0)'):
        self.addDefine('HAVE_GETPROCADDRESS',1)
      if self.checkLink('#include <Windows.h>','FreeLibrary(0)'):
        self.addDefine('HAVE_FREELIBRARY',1)
      if self.checkLink('#include <Windows.h>','GetLastError()'):
        self.addDefine('HAVE_GETLASTERROR',1)
      if self.checkLink('#include <Windows.h>','SetLastError(0)'):
        self.addDefine('HAVE_SETLASTERROR',1)
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
      
    self.types.check('int32_t', 'int')
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
  def configureScript(self):
    '''Output a script in the conf directory which will reproduce the configuration'''
    import nargs
    import sys
    scriptName = os.path.join(self.arch.arch,'conf', 'reconfigure-'+self.arch.arch+'.py')
    args = dict([(nargs.Arg.parseArgument(arg)[0], arg) for arg in self.framework.clArgs])
    if 'configModules' in args:
      if nargs.Arg.parseArgument(args['configModules'])[1] == ['PETSc.Configure']:
        del args['configModules']
    if 'optionsModule' in args:
      if nargs.Arg.parseArgument(args['optionsModule'])[1] == 'PETSc.compilerOptions':
        del args['optionsModule']
    if not 'PETSC_ARCH' in args:
      args['PETSC_ARCH'] = '-PETSC_ARCH='+str(self.arch.arch)
    f = file(scriptName, 'w')
    f.write('#!'+sys.executable+'\n')
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
      self.addMakeRule('shared_nomesg_noinstall','')
      self.addMakeRule('shared_install','',['-@echo "Now to install the libraries do: make install"',\
                                              '-@echo "========================================="'])
    else:
      self.installdir = os.path.join(self.petscdir.dir,self.arch.arch)
      self.addMakeRule('shared_nomesg_noinstall','shared_nomesg')            
      self.addMakeRule('shared_install','',['-@echo "Now to check if the libraries are working do: make test"',\
                                              '-@echo "========================================="'])
      return

  def configureGCOV(self):
    if self.framework.argDB['with-gcov']:
      self.addDefine('USE_GCOV','1')
    return

  def configureFortranFlush(self):
    if hasattr(self.compilers, 'FC'):
      for baseName in ['flush','flush_']:
        if self.libraries.check('', baseName, otherLibs = self.compilers.flibs, fortranMangle = 1):
          self.addDefine('HAVE_'+baseName.upper(), 1)
          return


  def configure(self):
    if not os.path.samefile(self.petscdir.dir, os.getcwd()):
      raise RuntimeError('Wrong PETSC_DIR option specified: '+str(self.petscdir.dir) + '\n  Configure invoked in: '+os.path.realpath(os.getcwd()))
    self.framework.header          = self.arch.arch+'/include/petscconf.h'
    self.framework.cHeader         = self.arch.arch+'/include/petscfix.h'
    self.framework.makeMacroHeader = self.arch.arch+'/conf/petscvariables'
    self.framework.makeRuleHeader  = self.arch.arch+'/conf/petscrules'
    if self.libraries.math is None:
      raise RuntimeError('PETSc requires a functional math library. Please send configure.log to petsc-maint@mcs.anl.gov.')
    if self.languages.clanguage == 'Cxx' and not hasattr(self.compilers, 'CXX'):
      raise RuntimeError('Cannot set C language to C++ without a functional C++ compiler.')
    self.executeTest(self.configureInline)
    self.executeTest(self.configurePrefetch)
    self.executeTest(self.configureSolaris)
    self.executeTest(self.configureLinux)
    self.executeTest(self.configureWin32)
    self.executeTest(self.configureScript)
    self.executeTest(self.configureInstall)
    self.executeTest(self.configureGCOV)
    self.executeTest(self.configureFortranFlush)
    # dummy rules, always needed except for remote builds
    self.addMakeRule('remote','')
    self.addMakeRule('remoteclean','')
    
    self.Dump()
    self.dumpConfigInfo()
    self.framework.log.write('================================================================================\n')
    self.logClear()
    return
