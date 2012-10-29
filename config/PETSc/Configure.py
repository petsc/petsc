import config.base

import os
import re

# The sorted() builtin is not available with python-2.3
try: sorted
except NameError:
  def sorted(lst):
    lst.sort()
    return lst

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    return

  def __str2__(self):
    desc = []
    desc.append('xxx=========================================================================xxx')
    if self.getMakeMacro('PETSC_BUILD_USING_CMAKE'):
      build_type = 'cmake build'
    else:
      build_type = 'legacy build'
    desc.append(' Configure stage complete. Now build PETSc libraries with (%s):' % build_type)
    desc.append('   make PETSC_DIR='+self.petscdir.dir+' PETSC_ARCH='+self.arch.arch+' all')
    desc.append(' or (experimental with python):')
    desc.append('   PETSC_DIR='+self.petscdir.dir+' PETSC_ARCH='+self.arch.arch+' ./config/builder.py')
    desc.append('xxx=========================================================================xxx')
    return '\n'.join(desc)+'\n'

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc',  '-prefix=<dir>',                  nargs.Arg(None, '', 'Specifiy location to install PETSc (eg. /usr/local)'))
    help.addArgument('Windows','-with-windows-graphics=<bool>',   nargs.ArgBool(None, 1,'Enable check for Windows Graphics'))
    help.addArgument('PETSc', '-with-default-arch=<bool>',        nargs.ArgBool(None, 1, 'Allow using the last configured arch without setting PETSC_ARCH'))
    help.addArgument('PETSc','-with-single-library=<bool>',       nargs.ArgBool(None, 1,'Put all PETSc code into the single -lpetsc library'))
    help.addArgument('PETSc', '-with-ios=<bool>',              nargs.ArgBool(None, 0, 'Build an iPhone/iPad version of PETSc library'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers  = framework.require('config.setCompilers',      self)
    self.arch          = framework.require('PETSc.utilities.arch',     self.setCompilers)
    self.petscdir      = framework.require('PETSc.utilities.petscdir', self.setCompilers)
    self.languages     = framework.require('PETSc.utilities.languages',self.setCompilers)
    self.debugging     = framework.require('PETSc.utilities.debugging',self.setCompilers)
    self.CHUD          = framework.require('PETSc.utilities.CHUD',     self)        
    self.compilers     = framework.require('config.compilers',         self)
    self.types         = framework.require('config.types',             self)
    self.headers       = framework.require('config.headers',           self)
    self.functions     = framework.require('config.functions',         self)
    self.libraries     = framework.require('config.libraries',         self)
    self.atomics       = framework.require('config.atomics',           self)
    if os.path.isdir(os.path.join('config', 'PETSc')):
      for d in ['utilities', 'packages']:
        for utility in os.listdir(os.path.join('config', 'PETSc', d)):
          (utilityName, ext) = os.path.splitext(utility)
          if not utilityName.startswith('.') and not utilityName.startswith('#') and ext == '.py' and not utilityName == '__init__':
            utilityObj                    = self.framework.require('PETSc.'+d+'.'+utilityName, self)
            utilityObj.headerPrefix       = self.headerPrefix
            utilityObj.archProvider       = self.arch
            utilityObj.languageProvider   = self.languages
            utilityObj.installDirProvider = self.petscdir
            setattr(self, utilityName.lower(), utilityObj)

    for package in config.packages.all:
      if not package == 'PETSc':
        packageObj                    = framework.require('config.packages.'+package, self)
        packageObj.archProvider       = self.arch
        packageObj.languageProvider   = self.languages
        packageObj.installDirProvider = self.petscdir
        setattr(self, package.lower(), packageObj)
    # Force blaslapack to depend on scalarType so precision is set before BlasLapack is built
    framework.require('PETSc.utilities.scalarTypes', self.f2cblaslapack)
    self.f2cblaslapack.precisionProvider = self.scalartypes
    framework.require('PETSc.utilities.scalarTypes', self.blaslapack)
    self.blaslapack.precisionProvider = self.scalartypes

    self.compilers.headerPrefix  = self.headerPrefix
    self.types.headerPrefix      = self.headerPrefix
    self.headers.headerPrefix    = self.headerPrefix
    self.functions.headerPrefix  = self.headerPrefix
    self.libraries.headerPrefix  = self.headerPrefix
    self.blaslapack.headerPrefix = self.headerPrefix
    self.mpi.headerPrefix        = self.headerPrefix
    headersC = map(lambda name: name+'.h', ['setjmp','dos', 'endian', 'fcntl', 'float', 'io', 'limits', 'malloc', 'pwd', 'search', 'strings',
                                            'unistd', 'sys/sysinfo', 'machine/endian', 'sys/param', 'sys/procfs', 'sys/resource',
                                            'sys/systeminfo', 'sys/times', 'sys/utsname','string', 'stdlib','memory',
                                            'sys/socket','sys/wait','netinet/in','netdb','Direct','time','Ws2tcpip','sys/types',
                                            'WindowsX', 'cxxabi','float','ieeefp','stdint','fenv','sched','pthread'])
    functions = ['access', '_access', 'clock', 'drand48', 'getcwd', '_getcwd', 'getdomainname', 'gethostname', 'getpwuid',
                 'gettimeofday', 'getwd', 'memalign', 'memmove', 'mkstemp', 'popen', 'PXFGETARG', 'rand', 'getpagesize',
                 'readlink', 'realpath',  'sigaction', 'signal', 'sigset', 'usleep', 'sleep', '_sleep', 'socket',
                 'times', 'gethostbyname', 'uname','snprintf','_snprintf','_fullpath','lseek','_lseek','time','fork','stricmp',
                 'strcasecmp', 'bzero', 'dlopen', 'dlsym', 'dlclose', 'dlerror',
                 '_intel_fast_memcpy','_intel_fast_memset']
    libraries1 = [(['socket', 'nsl'], 'socket'), (['fpe'], 'handle_sigfpes')]
    self.headers.headers.extend(headersC)
    self.functions.functions.extend(functions)
    self.libraries.libraries.extend(libraries1)

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

      # F90 Modules
      if self.setCompilers.fortranModuleIncludeFlag:
        self.addMakeMacro('FC_MODULE_FLAG', self.setCompilers.fortranModuleIncludeFlag)
      else: # for non-f90 compilers like g77
        self.addMakeMacro('FC_MODULE_FLAG', '-I')
      if self.setCompilers.fortranModuleIncludeFlag:
        self.addMakeMacro('FC_MODULE_OUTPUT_FLAG', self.setCompilers.fortranModuleOutputFlag)
    else:
      self.addMakeMacro('FC','')

    if hasattr(self.compilers, 'CUDAC'):
      self.setCompilers.pushLanguage('CUDA')
      self.addMakeMacro('CUDAC_FLAGS',self.setCompilers.getCompilerFlags())
      self.setCompilers.popLanguage()

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
      self.addDefine('SLSUFFIX','""')
    else:
      self.addMakeMacro('SL_LINKER_SUFFIX', self.setCompilers.sharedLibraryExt)
      self.addDefine('SLSUFFIX','"'+self.setCompilers.sharedLibraryExt+'"')
      
    self.addMakeMacro('SL_LINKER_LIBS','${PETSC_EXTERNAL_LIB_BASIC}')

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
    if self.setCompilers.isCrayVector('CC'):
      self.addDefine('HAVE_CRAY_VECTOR','1')

#-----------------------------------------------------------------------------------------------------
    if self.functions.haveFunction('gethostbyname') and self.functions.haveFunction('socket') and self.headers.haveHeader('netinet/in.h'):
      self.addDefine('USE_SOCKET_VIEWER','1')
      if self.checkCompile('#include <sys/socket.h>','setsockopt(0,SOL_SOCKET,SO_REUSEADDR,0,0)'):
        self.addDefine('HAVE_SO_REUSEADDR','1')

#-----------------------------------------------------------------------------------------------------
    # print include and lib for makefiles
    self.framework.packages.reverse()
    includes = [os.path.join(self.petscdir.dir,'include'),os.path.join(self.petscdir.dir,self.arch.arch,'include')]
    libs = []
    for i in self.framework.packages:
      if i.useddirectly:
        self.addDefine('HAVE_'+i.PACKAGE, 1)  # ONLY list package if it is used directly by PETSc (and not only by another package)
      if not isinstance(i.lib, list):
        i.lib = [i.lib]
      libs.extend(i.lib)
      self.addMakeMacro(i.PACKAGE+'_LIB', self.libraries.toStringNoDupes(i.lib))
      if hasattr(i,'include'):
        if not isinstance(i.include,list):
          i.include = [i.include]
        includes.extend(i.include)
        self.addMakeMacro(i.PACKAGE+'_INCLUDE',self.headers.toStringNoDupes(i.include))
    if self.framework.argDB['with-single-library']:
      self.addMakeMacro('PETSC_WITH_EXTERNAL_LIB',self.libraries.toStringNoDupes(['-L'+os.path.join(self.petscdir.dir,self.arch.arch,'lib'),' -lpetsc']+libs+self.libraries.math+self.compilers.flibs+self.compilers.cxxlibs+self.compilers.LIBS.split(' '))+self.CHUD.LIBS)
    self.addMakeMacro('PETSC_EXTERNAL_LIB_BASIC',self.libraries.toStringNoDupes(libs+self.libraries.math+self.compilers.flibs+self.compilers.cxxlibs+self.compilers.LIBS.split(' '))+self.CHUD.LIBS)
    self.PETSC_EXTERNAL_LIB_BASIC = self.libraries.toStringNoDupes(libs+self.libraries.math+self.compilers.flibs+self.compilers.cxxlibs+self.compilers.LIBS.split(' '))+self.CHUD.LIBS
    self.addMakeMacro('PETSC_CC_INCLUDES',self.headers.toStringNoDupes(includes))
    self.PETSC_CC_INCLUDES = self.headers.toStringNoDupes(includes)
    if hasattr(self.compilers, 'FC'):
      if self.compilers.fortranIsF90:
        self.addMakeMacro('PETSC_FC_INCLUDES',self.headers.toStringNoDupes(includes,includes))
      else:
        self.addMakeMacro('PETSC_FC_INCLUDES',self.headers.toStringNoDupes(includes))
    
    self.addMakeMacro('DESTDIR',self.installdir)
    self.addDefine('LIB_DIR','"'+os.path.join(self.installdir,'lib')+'"')

    if self.framework.argDB['with-single-library']:
      # overrides the values set in conf/variables
      self.addMakeMacro('LIBNAME','${INSTALL_LIB_DIR}/libpetsc.${AR_LIB_SUFFIX}')
      self.addMakeMacro('SHLIBS','libpetsc')
      self.addMakeMacro('PETSC_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_KSP_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_TS_LIB_BASIC','-lpetsc')
      self.addDefine('USE_SINGLE_LIBRARY', '1')
      if self.sharedlibraries.useShared:
        self.addMakeMacro('PETSC_SYS_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_VEC_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_MAT_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_DM_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_KSP_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_SNES_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_TS_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_CHARACTERISTIC_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_CONTRIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
      else:
        self.addMakeMacro('PETSC_SYS_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_VEC_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_MAT_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_DM_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_KSP_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_SNES_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_TS_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_CHARACTERISTIC_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_CONTRIB','${PETSC_WITH_EXTERNAL_LIB}')
      
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

  def dumpMachineInfo(self):
    import platform
    import time
    import script
    fd = file(os.path.join(self.arch.arch,'include','petscmachineinfo.h'),'w')
    fd.write('static const char *petscmachineinfo = \"\\n\"\n')
    fd.write('\"-----------------------------------------\\n\"\n')
    fd.write('\"Libraries compiled on %s on %s \\n\"\n' % (time.ctime(time.time()), platform.node()))
    fd.write('\"Machine characteristics: %s\\n\"\n' % (platform.platform()))
    fd.write('\"Using PETSc directory: %s\\n\"\n' % (self.petscdir.dir))
    fd.write('\"Using PETSc arch: %s\\n\"\n' % (self.arch.arch))
    fd.write('\"-----------------------------------------\\n\";\n')
    fd.write('static const char *petsccompilerinfo = \"\\n\"\n')
    self.setCompilers.pushLanguage(self.languages.clanguage)
    fd.write('\"Using C compiler: %s %s ${COPTFLAGS} ${CFLAGS}\\n\"\n' % (self.setCompilers.getCompiler(), self.setCompilers.getCompilerFlags()))
    self.setCompilers.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      fd.write('\"Using Fortran compiler: %s %s ${FOPTFLAGS} ${FFLAGS} %s\\n\"\n' % (self.setCompilers.getCompiler(), self.setCompilers.getCompilerFlags(), self.setCompilers.CPPFLAGS))
      self.setCompilers.popLanguage()
    fd.write('\"-----------------------------------------\\n\";\n')
    fd.write('static const char *petsccompilerflagsinfo = \"\\n\"\n')
    fd.write('\"Using include paths: %s %s %s\\n\"\n' % ('-I'+os.path.join(self.petscdir.dir, self.arch.arch, 'include'), '-I'+os.path.join(self.petscdir.dir, 'include'), self.PETSC_CC_INCLUDES.replace('\\ ','\\\\ ')))
    fd.write('\"-----------------------------------------\\n\";\n')
    fd.write('static const char *petsclinkerinfo = \"\\n\"\n')
    self.setCompilers.pushLanguage(self.languages.clanguage)
    fd.write('\"Using C linker: %s\\n\"\n' % (self.setCompilers.getLinker()))
    self.setCompilers.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      fd.write('\"Using Fortran linker: %s\\n\"\n' % (self.setCompilers.getLinker()))
      self.setCompilers.popLanguage()
    if self.framework.argDB['with-single-library']:
      petsclib = '-lpetsc'
    else:
      petsclib = '-lpetscts -lpetscsnes -lpetscksp -lpetscdm -lpetscmat -lpetscvec -lpetscsys'
    fd.write('\"Using libraries: %s%s -L%s %s %s\\n\"\n' % (self.setCompilers.CSharedLinkerFlag, os.path.join(self.petscdir.dir, self.arch.arch, 'lib'), os.path.join(self.petscdir.dir, self.arch.arch, 'lib'), petsclib, self.PETSC_EXTERNAL_LIB_BASIC.replace('\\ ','\\\\ ')))
    fd.write('\"-----------------------------------------\\n\";\n')
    fd.close()
    return

  def dumpCMakeConfig(self):
    '''
    Writes configuration-specific values to ${PETSC_ARCH}/conf/PETScConfig.cmake.
    This file is private to PETSc and should not be included by third parties
    (a suitable file can be produced later by CMake, but this is not it).
    '''
    def cmakeset(fd,key,val=True):
      if val == True: val = 'YES'
      if val == False: val = 'NO'
      fd.write('set (' + key + ' ' + val + ')\n')
    def ensurelist(a):
      if isinstance(a,list):
        return a
      else:
        return [a]
    def libpath(lib):
      'Returns a search path if that is what this item provides, else "" which will be cleaned out later'
      if not isinstance(lib,str): return ''
      if lib.startswith('-L'): return lib[2:]
      if lib.startswith('-R'): return lib[2:]
      if lib.startswith('-Wl,-rpath,'):
        # This case occurs when an external package needs a specific system library that is normally provided by the compiler.
        # In other words, the -L path is builtin to the wrapper or compiler, here we provide it so that CMake can locate the
        # corresponding library.
        return lib[len('-Wl,-rpath,'):]
      if lib.startswith('-'): return ''
      return os.path.dirname(lib)
    def cleanlib(lib):
      'Returns a library name if that is what this item provides, else "" which will be cleaned out later'
      if not isinstance(lib,str): return ''
      if lib.startswith('-l'):  return lib[2:]
      if lib.startswith('-Wl') or lib.startswith('-L'): return ''
      lib = os.path.splitext(os.path.basename(lib))[0]
      if lib.startswith('lib'): return lib[3:]
      return lib
    def nub(lst):
      'Return a list containing the first occurrence of each unique element'
      unique = []
      for elem in lst:
        if elem not in unique and elem != '':
          unique.append(elem)
      return unique
    try: reversed # reversed was added in Python-2.4
    except NameError:
      def reversed(lst): return lst[::-1]
    def nublast(lst):
      'Return a list containing the last occurrence of each unique entry in a list'
      return reversed(nub(reversed(lst)))
    def cmakeexpand(varname):
      return r'"${' + varname + r'}"'
    def uniqextend(lst,new):
      for x in ensurelist(new):
        if x not in lst:
          lst.append(x)
    def notstandardinclude(path):
      return path not in '/usr/include'.split() # /usr/local/include is not automatically included on FreeBSD
    def writeMacroDefinitions(fd):
      if self.mpi.usingMPIUni:
        cmakeset(fd,'PETSC_HAVE_MPIUNI')
      for pkg in self.framework.packages:
        if pkg.useddirectly:
          cmakeset(fd,'PETSC_HAVE_' + pkg.PACKAGE)
        for pair in pkg.defines.items():
          if pair[0].startswith('HAVE_') and pair[1]:
            cmakeset(fd, self.framework.getFullDefineName(pkg, pair[0]), pair[1])
      for name,val in self.functions.defines.items():
        cmakeset(fd,'PETSC_'+name,val)
      for dct in [self.defines, self.libraryoptions.defines]:
        for k,v in dct.items():
          if k.startswith('USE_'):
            cmakeset(fd,'PETSC_' + k, v)
      cmakeset(fd,'PETSC_USE_COMPLEX', self.scalartypes.scalartype == 'complex')
      cmakeset(fd,'PETSC_USE_REAL_' + self.scalartypes.precision.upper())
      cmakeset(fd,'PETSC_CLANGUAGE_'+self.languages.clanguage)
      if hasattr(self.compilers, 'FC'):
        cmakeset(fd,'PETSC_HAVE_FORTRAN')
        if self.compilers.fortranIsF90:
          cmakeset(fd,'PETSC_USING_F90')
        if self.compilers.fortranIsF2003:
          cmakeset(fd,'PETSC_USING_F2003')
      if hasattr(self.compilers, 'CXX'):
        cmakeset(fd,'PETSC_HAVE_CXX')
      if self.sharedlibraries.useShared:
        cmakeset(fd,'BUILD_SHARED_LIBS')
    def writeBuildFlags(fd):
      def extendby(lib):
        libs = ensurelist(lib)
        lib_paths.extend(map(libpath,libs))
        lib_libs.extend(map(cleanlib,libs))
      lib_paths = []
      lib_libs  = []
      includes  = []
      libvars   = []
      for pkg in self.framework.packages:
        extendby(pkg.lib)
        uniqextend(includes,pkg.include)
      extendby(self.libraries.math)
      extendby(self.libraries.rt)
      extendby(self.compilers.flibs)
      extendby(self.compilers.cxxlibs)
      extendby(self.compilers.LIBS.split())
      for libname in nublast(lib_libs):
        libvar = 'PETSC_' + libname.upper() + '_LIB'
        addpath = ''
        for lpath in nublast(lib_paths):
          addpath += '"' + str(lpath) + '" '
        fd.write('find_library (' + libvar + ' ' + libname + ' HINTS ' + addpath + ')\n')
        libvars.append(libvar)
      fd.write('mark_as_advanced (' + ' '.join(libvars) + ')\n')
      fd.write('set (PETSC_PACKAGE_LIBS ' + ' '.join(map(cmakeexpand,libvars)) + ')\n')
      includes = filter(notstandardinclude,includes)
      fd.write('set (PETSC_PACKAGE_INCLUDES ' + ' '.join(map(lambda i: '"'+i+'"',includes)) + ')\n')
    fd = open(os.path.join(self.arch.arch,'conf','PETScConfig.cmake'), 'w')
    writeMacroDefinitions(fd)
    writeBuildFlags(fd)
    fd.close()
    return

  def dumpCMakeLists(self):
    import sys
    if sys.version_info >= (2,5):
      import cmakegen
      try:
        cmakegen.main(self.petscdir.dir, log=self.framework.log)
      except (OSError), e:
        self.framework.logPrint('Generating CMakeLists.txt failed:\n' + str(e))
    else:
      self.framework.logPrint('Skipping cmakegen due to old python version: ' +str(sys.version_info) )

  def cmakeBoot(self):
    import sys
    self.cmakeboot_success = False
    if sys.version_info >= (2,5) and hasattr(self.cmake,'cmake'):
      try:
        import cmakeboot
        self.cmakeboot_success = cmakeboot.main(petscdir=self.petscdir.dir,petscarch=self.arch.arch,argDB=self.argDB,framework=self.framework,log=self.framework.log)
      except (OSError), e:
        self.framework.logPrint('Booting CMake in PETSC_ARCH failed:\n' + str(e))
      except (ImportError, KeyError), e:
        self.framework.logPrint('Importing cmakeboot failed:\n' + str(e))
      if self.cmakeboot_success:
        if self.framework.argDB['with-cuda']: # Our CMake build does not support CUDA at this time
          self.framework.logPrint('CMake configured successfully, but could not be used by default because --with-cuda was used\n')
        elif hasattr(self.compilers, 'FC') and self.compilers.fortranIsF90 and not self.setCompilers.fortranModuleOutputFlag:
          self.framework.logPrint('CMake configured successfully, but could not be used by default because of missing fortranModuleOutputFlag\n')
        else:
          self.framework.logPrint('CMake configured successfully, using as default build\n')
          self.addMakeMacro('PETSC_BUILD_USING_CMAKE',1)
      else:
        self.framework.logPrint('CMake configuration was unsuccessful\n')
    else:
      self.framework.logPrint('Skipping cmakeboot due to old python version: ' +str(sys.version_info) )
    return

  def configurePrefetch(self):
    '''Sees if there are any prefetch functions supported'''
    if config.setCompilers.Configure.isSolaris() or self.framework.argDB['with-ios']:
      self.addDefine('Prefetch(a,b,c)', ' ')
      return
    self.pushLanguage(self.languages.clanguage)      
    if self.checkLink('#include <xmmintrin.h>', 'void *v = 0;_mm_prefetch((const char*)v,_MM_HINT_NTA);\n'):
      # The Intel Intrinsics manual [1] specifies the prototype
      #
      #   void _mm_prefetch(char const *a, int sel);
      #
      # but other vendors seem to insist on using subtly different
      # prototypes, including void* for the pointer, and an enum for
      # sel.  These are both reasonable changes, but negatively impact
      # portability.
      #
      # [1] http://software.intel.com/file/6373
      self.addDefine('HAVE_XMMINTRIN_H', 1)
      self.addDefine('Prefetch(a,b,c)', '_mm_prefetch((const char*)(a),(c))')
      self.addDefine('PREFETCH_HINT_NTA', '_MM_HINT_NTA')
      self.addDefine('PREFETCH_HINT_T0',  '_MM_HINT_T0')
      self.addDefine('PREFETCH_HINT_T1',  '_MM_HINT_T1')
      self.addDefine('PREFETCH_HINT_T2',  '_MM_HINT_T2')
    elif self.checkLink('#include <xmmintrin.h>', 'void *v = 0;_mm_prefetch(v,_MM_HINT_NTA);\n'):
      self.addDefine('HAVE_XMMINTRIN_H', 1)
      self.addDefine('Prefetch(a,b,c)', '_mm_prefetch((const void*)(a),(c))')
      self.addDefine('PREFETCH_HINT_NTA', '_MM_HINT_NTA')
      self.addDefine('PREFETCH_HINT_T0',  '_MM_HINT_T0')
      self.addDefine('PREFETCH_HINT_T1',  '_MM_HINT_T1')
      self.addDefine('PREFETCH_HINT_T2',  '_MM_HINT_T2')
    elif self.checkLink('', 'void *v = 0;__builtin_prefetch(v,0,0);\n'):
      # From GCC docs: void __builtin_prefetch(const void *addr,int rw,int locality)
      #
      #   The value of rw is a compile-time constant one or zero; one
      #   means that the prefetch is preparing for a write to the memory
      #   address and zero, the default, means that the prefetch is
      #   preparing for a read. The value locality must be a compile-time
      #   constant integer between zero and three. A value of zero means
      #   that the data has no temporal locality, so it need not be left
      #   in the cache after the access. A value of three means that the
      #   data has a high degree of temporal locality and should be left
      #   in all levels of cache possible. Values of one and two mean,
      #   respectively, a low or moderate degree of temporal locality.
      #
      # Here we adopt Intel's x86/x86-64 naming scheme for the locality
      # hints.  Using macros for these values in necessary since some
      # compilers require an enum.
      self.addDefine('Prefetch(a,b,c)', '__builtin_prefetch((a),(b),(c))')
      self.addDefine('PREFETCH_HINT_NTA', '0')
      self.addDefine('PREFETCH_HINT_T0',  '3')
      self.addDefine('PREFETCH_HINT_T1',  '2')
      self.addDefine('PREFETCH_HINT_T2',  '1')
    else:
      self.addDefine('Prefetch(a,b,c)', ' ')
    self.popLanguage()

  def configureFeatureTestMacros(self):
    '''Checks if certain feature test macros are support'''
    if self.checkCompile('#define _POSIX_C_SOURCE 200112L\n#include <sysctl.h>',''):
       self.addDefine('_POSIX_C_SOURCE_200112L', '1')
    if self.checkCompile('#define _BSD_SOURCE\n#include<stdlib.h>',''):
       self.addDefine('_BSD_SOURCE', '1')
    if self.checkCompile('#define _GNU_SOURCE\n#include <sched.h>','cpu_set_t mset;\nCPU_ZERO(&mset);'):
       self.addDefine('_GNU_SOURCE', '1')

  def configureAtoll(self):
    '''Checks if atoll exists'''
    if self.checkCompile('#define _POSIX_C_SOURCE 200112L\n#include <stdlib.h>','long v = atoll("25")') or self.checkCompile ('#include <stdlib.h>','long v = atoll("25")'):
       self.addDefine('HAVE_ATOLL', '1')

  def configureUnused(self):
    '''Sees if __attribute((unused)) is supported'''
    if self.framework.argDB['with-ios']:
      self.addDefine('UNUSED', ' ')
      return
    self.pushLanguage(self.languages.clanguage)      
    if self.checkLink('__attribute((unused)) static int myfunc(__attribute((unused)) void *name){ return 1;}', 'int i = 0;\nint j = myfunc(&i);\ntypedef void* atype;\n__attribute((unused))  atype a;\n'):
      self.addDefine('UNUSED', '__attribute((unused))')
    else:
      self.addDefine('UNUSED', ' ')
    self.popLanguage()

  def configureExpect(self):
    '''Sees if the __builtin_expect directive is supported'''
    self.pushLanguage(self.languages.clanguage)
    if self.checkLink('', 'if (__builtin_expect(0,1)) return 1;'):
      self.addDefine('HAVE_BUILTIN_EXPECT', 1)
    self.popLanguage()

  def configureFunctionName(self):
    '''Sees if the compiler supports __func__ or a variant.  Falls back
    on __FUNCT__ which PETSc source defines, but most users do not, thus
    stack traces through user code are better when the compiler's
    variant is used.'''
    def getFunctionName(lang):
      name = '__FUNCT__'
      self.pushLanguage(lang)
      if self.checkLink('', "if (__func__[0] != 'm') return 1;"):
        name = '__func__'
      elif self.checkLink('', "if (__FUNCTION__[0] != 'm') return 1;"):
        name = '__FUNCTION__'
      self.popLanguage()
      return name
    langs = []

    self.addDefine('FUNCTION_NAME_C', getFunctionName('C'))
    if hasattr(self.compilers, 'CXX'):
      self.addDefine('FUNCTION_NAME_CXX', getFunctionName('Cxx'))
    else:
      self.addDefine('FUNCTION_NAME_CXX', '__FUNCT__')

  def configureIntptrt(self):
    '''Determine what to use for uintptr_t'''
    def staticAssertSizeMatchesVoidStar(inc,typename):
      # The declaration is an error if either array size is negative.
      # It should be okay to use an int that is too large, but it would be very unlikely for this to be the case
      return self.checkCompile(inc, ('#define STATIC_ASSERT(cond) char negative_length_if_false[2*(!!(cond))-1]\n'
                                     + 'STATIC_ASSERT(sizeof(void*) == sizeof(%s));'%typename))
    self.pushLanguage(self.languages.clanguage)
    if self.checkCompile('#include <stdint.h>', 'int x; uintptr_t i = (uintptr_t)&x;'):
      self.addDefine('UINTPTR_T', 'uintptr_t')
    elif staticAssertSizeMatchesVoidStar('','unsigned long long'):
      self.addDefine('UINTPTR_T', 'unsigned long long')
    elif staticAssertSizeMatchesVoidStar('#include <stdlib.h>','size_t') or staticAssertSizeMatchesVoidStar('#include <string.h>', 'size_t'):
      self.addDefine('UINTPTR_T', 'size_t')
    elif staticAssertSizeMatchesVoidStar('','unsigned long'):
      self.addDefine('UINTPTR_T', 'unsigned long')
    elif staticAssertSizeMatchesVoidStar('','unsigned'):
      self.addDefine('UINTPTR_T', 'unsigned')
    else:
      raise RuntimeError('Could not find any unsigned integer type matching void*')
    self.popLanguage()
      
  def configureInline(self):
    '''Get a generic inline keyword, depending on the language'''
    if self.languages.clanguage == 'C':
      self.addDefine('STATIC_INLINE', self.compilers.cStaticInlineKeyword)
      self.addDefine('RESTRICT', self.compilers.cRestrict)
    elif self.languages.clanguage == 'Cxx':
      self.addDefine('STATIC_INLINE', self.compilers.cxxStaticInlineKeyword)
      self.addDefine('RESTRICT', self.compilers.cxxRestrict)

    if self.checkCompile('#include <dlfcn.h>\n void *ptr =  RTLD_DEFAULT;'):
      self.addDefine('RTLD_DEFAULT','1')
    return

  def configureSolaris(self):
    '''Solaris specific stuff'''
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
    # TODO: Test for this by mallocing an odd number of floats and checking the address
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
    if self.checkCompile('#include <Windows.h>\n#include <fcntl.h>\n', 'int flags = O_BINARY;'):
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
  def configureDefaultArch(self):
    conffile = os.path.join('conf', 'petscvariables')
    if self.framework.argDB['with-default-arch']:
      fd = file(conffile, 'w')
      fd.write('PETSC_ARCH='+self.arch.arch+'\n')
      fd.write('PETSC_DIR='+self.petscdir.dir+'\n')
      fd.write('include '+os.path.join(self.petscdir.dir,self.arch.arch,'conf','petscvariables')+'\n')
      fd.close()
      self.framework.actions.addArgument('PETSc', 'Build', 'Set default architecture to '+self.arch.arch+' in '+conffile)
    elif os.path.isfile(conffile):
      try:
        os.unlink(conffile)
      except:
        raise RuntimeError('Unable to remove file '+conffile+'. Did a different user create it?')
    return

#-----------------------------------------------------------------------------------------------------
  def configureScript(self):
    '''Output a script in the conf directory which will reproduce the configuration'''
    import nargs
    import sys
    scriptName = os.path.join(self.arch.arch,'conf', 'reconfigure-'+self.arch.arch+'.py')
    args = dict([(nargs.Arg.parseArgument(arg)[0], arg) for arg in self.framework.clArgs])
    if 'configModules' in args:
      if nargs.Arg.parseArgument(args['configModules'])[1] == 'PETSc.Configure':
        del args['configModules']
    if 'optionsModule' in args:
      if nargs.Arg.parseArgument(args['optionsModule'])[1] == 'PETSc.compilerOptions':
        del args['optionsModule']
    if not 'PETSC_ARCH' in args:
      args['PETSC_ARCH'] = 'PETSC_ARCH='+str(self.arch.arch)
    f = file(scriptName, 'w')
    f.write('#!'+sys.executable+'\n')
    f.write('if __name__ == \'__main__\':\n')
    f.write('  import sys\n')
    f.write('  import os\n')
    f.write('  sys.path.insert(0, os.path.abspath(\'config\'))\n')
    f.write('  import configure\n')
    # pretty print repr(args.values())
    f.write('  configure_options = [\n')
    for itm in sorted(args.values()):
      f.write('    \''+str(itm)+'\',\n')
    f.write('  ]\n')
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
      self.addMakeRule('shared_install','',['-@echo "Now to install the libraries do:"',\
                                              '-@echo "make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} install"',\
                                              '-@echo "========================================="'])
    else:
      self.installdir = os.path.join(self.petscdir.dir,self.arch.arch)
      self.addMakeRule('shared_install','',['-@echo "Now to check if the libraries are working do:"',\
                                              '-@echo "make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} test"',\
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

  def postProcessPackages(self):
    postPackages=[]
    for i in self.framework.packages:
      if hasattr(i,'postProcess'): postPackages.append(i)
    if postPackages:
      # ctetgen needs petsc conf files. so attempt to create them early
      self.framework.dumpConfFiles()
      for i in postPackages: i.postProcess()
    return

  def configure(self):
    if not os.path.samefile(self.petscdir.dir, os.getcwd()):
      raise RuntimeError('Wrong PETSC_DIR option specified: '+str(self.petscdir.dir) + '\n  Configure invoked in: '+os.path.realpath(os.getcwd()))
    if self.framework.argDB['prefix'] and os.path.isdir(self.framework.argDB['prefix']) and os.path.samefile(self.framework.argDB['prefix'],self.petscdir.dir):
      raise RuntimeError('Incorrect option --prefix='+self.framework.argDB['prefix']+' specified. It cannot be same as PETSC_DIR!')
    if self.framework.argDB['prefix'] and os.path.isdir(self.framework.argDB['prefix']) and os.path.samefile(self.framework.argDB['prefix'],os.path.join(self.petscdir.dir,self.arch.arch)):
      raise RuntimeError('Incorrect option --prefix='+self.framework.argDB['prefix']+' specified. It cannot be same as PETSC_DIR/PETSC_ARCH!')
    self.framework.header          = os.path.join(self.arch.arch,'include','petscconf.h')
    self.framework.cHeader         = os.path.join(self.arch.arch,'include','petscfix.h')
    self.framework.makeMacroHeader = os.path.join(self.arch.arch,'conf','petscvariables')
    self.framework.makeRuleHeader  = os.path.join(self.arch.arch,'conf','petscrules')
    if self.libraries.math is None:
      raise RuntimeError('PETSc requires a functional math library. Please send configure.log to petsc-maint@mcs.anl.gov.')
    if self.languages.clanguage == 'Cxx' and not hasattr(self.compilers, 'CXX'):
      raise RuntimeError('Cannot set C language to C++ without a functional C++ compiler.')
    self.executeTest(self.configureInline)
    self.executeTest(self.configurePrefetch)
    self.executeTest(self.configureUnused)
    self.executeTest(self.configureExpect);
    self.executeTest(self.configureFunctionName);
    self.executeTest(self.configureIntptrt);
    self.executeTest(self.configureSolaris)
    self.executeTest(self.configureLinux)
    self.executeTest(self.configureWin32)
    self.executeTest(self.configureDefaultArch)
    self.executeTest(self.configureScript)
    self.executeTest(self.configureInstall)
    self.executeTest(self.configureGCOV)
    self.executeTest(self.configureFortranFlush)
    self.executeTest(self.configureFeatureTestMacros)
    self.executeTest(self.configureAtoll)
    # dummy rules, always needed except for remote builds
    self.addMakeRule('remote','')
    self.addMakeRule('remoteclean','')
    
    self.Dump()
    self.dumpConfigInfo()
    self.dumpMachineInfo()
    self.postProcessPackages()
    self.dumpCMakeConfig()
    self.dumpCMakeLists()
    self.cmakeBoot()
    self.framework.log.write('================================================================================\n')
    self.logClear()
    return
