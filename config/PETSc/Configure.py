import config.base

import os
import sys
import re
import pickle

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.installed    = 0 # 1 indicates that Configure itself has already compiled and installed PETSc
    self.found        = 1
    return

  def __str2__(self):
    desc = ['  Using GNU make: ' + self.make.make]
    if self.defines.get('USE_COVERAGE'):
      desc.extend([
        '  Code coverage: yes',
        '  Using code coverage executable: {}'.format(self.getMakeMacro('PETSC_COVERAGE_EXEC'))
      ])
    if not self.installed:
      desc.append('xxx=========================================================================xxx')
      desc.append(' Configure stage complete. Now build PETSc libraries with:')
      desc.append('   %s PETSC_DIR=%s PETSC_ARCH=%s all' % (self.make.make_user, self.petscdir.dir, self.arch.arch))
      desc.append('xxx=========================================================================xxx')
    else:
      desc.append('xxx=========================================================================xxx')
      desc.append(' Installation complete. You do not need to run make to compile or install the software')
      desc.append('xxx=========================================================================xxx')
    desc.append('')
    return '\n'.join(desc)

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc',  '-prefix=<dir>',                              nargs.Arg(None, '', 'Specifiy location to install PETSc (eg. /usr/local)'))
    help.addArgument('PETSc',  '-with-prefetch=<bool>',                      nargs.ArgBool(None, 1,'Enable checking for prefetch instructions'))
    help.addArgument('Windows','-with-windows-graphics=<bool>',              nargs.ArgBool(None, 1,'Enable check for Windows Graphics'))
    help.addArgument('PETSc', '-with-default-arch=<bool>',                   nargs.ArgBool(None, 1, 'Allow using the last configured arch without setting PETSC_ARCH'))
    help.addArgument('PETSc','-with-single-library=<bool>',                  nargs.ArgBool(None, 1,'Put all PETSc code into the single -lpetsc library'))
    help.addArgument('PETSc','-with-fortran-bindings=<bool>',                nargs.ArgBool(None, 1,'Build PETSc fortran bindings in the library and corresponding module files'))
    help.addArgument('PETSc', '-with-ios=<bool>',                            nargs.ArgBool(None, 0, 'Build an iPhone/iPad version of PETSc library'))
    help.addArgument('PETSc', '-with-display=<x11display>',                  nargs.Arg(None, '', 'Specifiy DISPLAY env variable for use with matlab test)'))
    help.addArgument('PETSc', '-with-package-scripts=<pyscripts>',           nargs.ArgFileList(None,None,'Specify configure package scripts for user provided packages'))
    help.addArgument('PETSc', '-with-coverage=<bool>',                       nargs.ArgFuzzyBool(None, value=0, help='Enable or disable code-coverage collection'))
    help.addArgument('PETSc', '-with-coverage-exec=<executable>',            nargs.ArgExecutable(None, value='default-auto', mustExist=0, help='Name of executable to use for post-processing coverage data, e.g. \'gcov\' or \'llvm-cov\'. Pass \'auto\' to let configure infer from compiler'))
    help.addArgument('PETSc', '-with-tau-perfstubs=<bool>',                  nargs.ArgBool(None, 1,'Enable TAU profiler stubs'))
    help.addArgument('PETSc', '-with-strict-petscerrorcode=<bool>',          nargs.ArgFuzzyBool(None, value=0, help='Enable strict PetscErrorCode mode, which enables additional compile-time checking for misuse of PetscErrorCode and error handling'))
    return

  def registerPythonFile(self,filename,directory):
    ''' Add a python file to the framework and registers its headerprefix, ... externalpackagedir
        directory is the directory where the file relative to the BuildSystem or config path in python notation with . '''
    (utilityName, ext) = os.path.splitext(filename)
    if not utilityName.startswith('.') and not utilityName.startswith('#') and ext == '.py' and not utilityName == '__init__':
      if directory: directory = directory+'.'
      utilityObj                             = self.framework.require(directory+utilityName, self)
      utilityObj.headerPrefix                = self.headerPrefix
      utilityObj.archProvider                = self.arch
      utilityObj.languageProvider            = self.languages
      utilityObj.installDirProvider          = self.installdir
      utilityObj.externalPackagesDirProvider = self.externalpackagesdir
      utilityObj.precisionProvider           = self.scalartypes
      utilityObj.indexProvider               = self.indexTypes
      setattr(self, utilityName.lower(), utilityObj)
      return utilityObj
    return None

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.programs      = framework.require('config.programs',           self)
    self.setCompilers  = framework.require('config.setCompilers',       self)
    self.compilerFlags = framework.require('config.compilerFlags',      self)
    self.compilers     = framework.require('config.compilers',          self)
    self.arch          = framework.require('PETSc.options.arch',        self.setCompilers)
    self.petscdir      = framework.require('PETSc.options.petscdir',    self.arch)
    self.installdir    = framework.require('PETSc.options.installDir',  self)
    self.dataFilesPath = framework.require('PETSc.options.dataFilesPath',self)
    self.scalartypes   = framework.require('PETSc.options.scalarTypes', self)
    self.indexTypes    = framework.require('PETSc.options.indexTypes',  self)
    self.languages     = framework.require('PETSc.options.languages',   self.setCompilers)
    self.indexTypes    = framework.require('PETSc.options.indexTypes',  self.compilers)
    self.types         = framework.require('config.types',              self)
    self.headers       = framework.require('config.headers',            self)
    self.functions     = framework.require('config.functions',          self)
    self.libraries     = framework.require('config.libraries',          self)
    self.atomics       = framework.require('config.atomics',            self)
    self.make          = framework.require('config.packages.make',      self)
    self.blasLapack    = framework.require('config.packages.BlasLapack',self)
    self.mpi           = framework.require('config.packages.MPI',       self)
    self.fortran       = framework.require('config.compilersFortran',   self)
    self.externalpackagesdir = framework.require('PETSc.options.externalpackagesdir',self)

    for utility in sorted(os.listdir(os.path.join('config','PETSc','options'))):
      self.registerPythonFile(utility,'PETSc.options')

    for utility in sorted(os.listdir(os.path.join('config','BuildSystem','config','utilities'))):
      self.registerPythonFile(utility,'config.utilities')

    for package in sorted(os.listdir(os.path.join('config', 'BuildSystem', 'config', 'packages'))):
      obj = self.registerPythonFile(package,'config.packages')
      if obj:
        obj.archProvider                = self.framework.requireModule(obj.archProvider, obj)
        obj.languageProvider            = self.framework.requireModule(obj.languageProvider, obj)
        obj.installDirProvider          = self.framework.requireModule(obj.installDirProvider, obj)
        obj.externalPackagesDirProvider = self.framework.requireModule(obj.externalPackagesDirProvider, obj)
        obj.precisionProvider           = self.framework.requireModule(obj.precisionProvider, obj)
        obj.indexProvider               = self.framework.requireModule(obj.indexProvider, obj)

    # Force blaslapack and opencl to depend on scalarType so precision is set before BlasLapack is built
    framework.require('PETSc.options.scalarTypes', self.f2cblaslapack)
    framework.require('PETSc.options.scalarTypes', self.fblaslapack)
    framework.require('PETSc.options.scalarTypes', self.blaslapack)
    framework.require('PETSc.options.scalarTypes', self.opencl)

    self.programs.headerPrefix     = self.headerPrefix
    self.setCompilers.headerPrefix = self.headerPrefix
    self.compilers.headerPrefix    = self.headerPrefix
    self.fortran.headerPrefix      = self.headerPrefix
    self.types.headerPrefix        = self.headerPrefix
    self.headers.headerPrefix      = self.headerPrefix
    self.functions.headerPrefix    = self.headerPrefix
    self.libraries.headerPrefix    = self.headerPrefix

    # Register user provided package scripts
    if 'with-package-scripts' in self.framework.argDB:
      for script in self.framework.argDB['with-package-scripts']:
        if os.path.splitext(script)[1] != '.py':
          raise RuntimeError('Only python scripts compatible with configure package script format should be specified! Invalid option -with-package-scripts='+script)
        self.framework.logPrint('User is registering a new package script: '+script)
        dname,fname = os.path.split(script)
        if dname: sys.path.append(dname)
        self.registerPythonFile(fname,'')

    # test for a variety of basic headers and functions
    headersC = map(lambda name: name+'.h',['setjmp','dos','fcntl','float','io','malloc','pwd','strings',
                                            'unistd','machine/endian','sys/param','sys/procfs','sys/resource',
                                            'sys/systeminfo','sys/times','sys/utsname',
                                            'sys/socket','sys/wait','netinet/in','netdb','direct','time','Ws2tcpip','sys/types',
                                            'WindowsX','float','ieeefp','stdint','inttypes','immintrin'])
    functions = ['access','_access','clock','drand48','getcwd','_getcwd','getdomainname','gethostname',
                 'getwd','posix_memalign','popen','PXFGETARG','rand','getpagesize',
                 'readlink','realpath','usleep','sleep','_sleep',
                 'uname','snprintf','_snprintf','lseek','_lseek','time','fork','stricmp',
                 'strcasecmp','bzero','dlopen','dlsym','dlclose','dlerror',
                 '_set_output_format','_mkdir','socket','gethostbyname','fpresetsticky',
                 'fpsetsticky','__gcov_dump']
    libraries = [(['fpe'],'handle_sigfpes')]
    librariessock = [(['socket','nsl'],'socket')]
    self.headers.headers.extend(headersC)
    self.functions.functions.extend(functions)
    self.libraries.libraries.extend(libraries)
    if not hasattr(self,'socket'):
      self.libraries.libraries.extend(librariessock)
    return

  def DumpPkgconfig(self, petsc_pc):
    ''' Create a pkg-config file '''
    if not os.path.exists(os.path.join(self.petscdir.dir,self.arch.arch,'lib','pkgconfig')):
      os.makedirs(os.path.join(self.petscdir.dir,self.arch.arch,'lib','pkgconfig'))
    with open(os.path.join(self.petscdir.dir,self.arch.arch,'lib','pkgconfig',petsc_pc),'w') as fd:
      cflags_inc = ['-I${includedir}']
      if self.framework.argDB['prefix']:
        fd.write('prefix='+self.installdir.dir+'\n')
      else:
        fd.write('prefix='+os.path.join(self.petscdir.dir, self.arch.arch)+'\n')
        cflags_inc.append('-I' + os.path.join(self.petscdir.dir, 'include'))
      fd.write('exec_prefix=${prefix}\n')
      fd.write('includedir=${prefix}/include\n')
      fd.write('libdir=${prefix}/lib\n')

      with self.setCompilers.Language('C'):
        fd.write('ccompiler='+self.setCompilers.getCompiler()+'\n')
        fd.write('cflags_extra='+self.setCompilers.getCompilerFlags().strip()+'\n')
        fd.write('cflags_dep='+self.compilers.dependenciesGenerationFlag.get('C','')+'\n')
        fd.write('ldflag_rpath='+self.setCompilers.CSharedLinkerFlag+'\n')
      if hasattr(self.compilers, 'CXX'):
        with self.setCompilers.Language('C++'):
          fd.write('cxxcompiler='+self.setCompilers.getCompiler()+'\n')
          fd.write('cxxflags_extra='+self.setCompilers.getCompilerFlags().strip()+'\n')
      if hasattr(self.compilers, 'FC'):
        with self.setCompilers.Language('FC'):
          fd.write('fcompiler='+self.setCompilers.getCompiler()+'\n')
          fd.write('fflags_extra='+self.setCompilers.getCompilerFlags().strip()+'\n')
      if hasattr(self.compilers, 'CUDAC'):
        with self.setCompilers.Language('CUDA'):
          fd.write('cudacompiler='+self.setCompilers.getCompiler()+'\n')
          fd.write('cudaflags_extra='+self.setCompilers.getCompilerFlags().strip()+'\n')
          p = self.framework.require('config.packages.cuda')
          fd.write('cudalib='+self.libraries.toStringNoDupes(p.lib)+'\n')
          fd.write('cudainclude='+self.headers.toStringNoDupes(p.include)+'\n')
          if hasattr(self.setCompilers,'CUDA_CXX'):
            fd.write('cuda_cxx='+self.setCompilers.CUDA_CXX+'\n')
            fd.write('cuda_cxxflags='+self.setCompilers.CUDA_CXXFLAGS+'\n')

      fd.write('\n')
      fd.write('Name: PETSc\n')
      fd.write('Description: Library to solve ODEs and algebraic equations\n')
      fd.write('Version: %s\n' % self.petscdir.version)
      fd.write('Cflags: ' + ' '.join([self.setCompilers.CPPFLAGS] + cflags_inc) + '\n')
      fd.write('Libs: '+self.libraries.toStringNoDupes(['-L${libdir}', self.petsclib], with_rpath=False)+'\n')
      # Remove RPATH flags from library list.  User can add them using
      # pkg-config --variable=ldflag_rpath and pkg-config --libs-only-L
      fd.write('Libs.private: '+self.libraries.toStringNoDupes([f for f in self.packagelibs+self.complibs if not f.startswith(self.setCompilers.CSharedLinkerFlag)], with_rpath=False)+'\n')
    return

  def DumpModule(self):
    ''' Create a module file '''
    if not os.path.exists(os.path.join(self.petscdir.dir,self.arch.arch,'lib','petsc','conf','modules')):
      os.makedirs(os.path.join(self.petscdir.dir,self.arch.arch,'lib','petsc','conf','modules'))
    if not os.path.exists(os.path.join(self.petscdir.dir,self.arch.arch,'lib','petsc','conf','modules','petsc')):
      os.makedirs(os.path.join(self.petscdir.dir,self.arch.arch,'lib','petsc','conf','modules','petsc'))
    if self.framework.argDB['prefix']:
      installdir  = self.installdir.dir
      installarch = ''
      installpath = os.path.join(installdir,'bin')
    else:
      installdir  = self.petscdir.dir
      installarch = self.arch.arch
      installpath = os.path.join(installdir,installarch,'bin')+':'+os.path.join(installdir,'bin')
    fd = open(os.path.join(self.petscdir.dir,self.arch.arch,'lib','petsc','conf','modules','petsc',self.petscdir.version),'w')
    fd.write('''\
#%%Module

proc ModulesHelp { } {
    puts stderr "This module sets the path and environment variables for petsc-%s"
    puts stderr "     see https://petsc.org/ for more information      "
    puts stderr ""
}
module-whatis "PETSc - Portable, Extensible Toolkit for Scientific Computation"

set petsc_dir   "%s"
set petsc_arch  "%s"

setenv PETSC_ARCH "$petsc_arch"
setenv PETSC_DIR "$petsc_dir"
prepend-path PATH "%s"
''' % (self.petscdir.version, installdir, installarch, installpath))
    fd.close()
    return

  def Dump(self):
    ''' Actually put the values into the configuration files '''
    # eventually everything between -- should be gone
    if self.mpi.usingMPIUni:
      #
      # Remove any MPI/MPICH include files that may have been put here by previous runs of ./configure
      self.executeShellCommand('rm -rf  '+os.path.join(self.petscdir.dir,self.arch.arch,'include','mpi*')+' '+os.path.join(self.petscdir.dir,self.arch.arch,'include','opa*'), log = self.log)

    self.logPrintDivider()
    # Test for compiler-specific macros that need to be defined.
    if self.setCompilers.isCrayVector('CC', self.log):
      self.addDefine('HAVE_CRAY_VECTOR','1')

    if self.functions.haveFunction('gethostbyname') and self.functions.haveFunction('socket') and self.headers.haveHeader('netinet/in.h'):
      self.addDefine('USE_SOCKET_VIEWER','1')
      if self.checkCompile('#include <sys/socket.h>','setsockopt(0,SOL_SOCKET,SO_REUSEADDR,0,0)'):
        self.addDefine('HAVE_SO_REUSEADDR','1')

    self.logPrintDivider()
    self.setCompilers.pushLanguage('C')
    compiler = self.setCompilers.getCompiler()
    if [s for s in ['mpicc','mpiicc'] if os.path.basename(compiler).find(s)>=0]:
      try:
        output   = self.executeShellCommand(compiler + ' -show', log = self.log)[0]
        compiler = output.split(' ')[0]
        self.addDefine('MPICC_SHOW','"'+output.strip().replace('\n','\\\\n').replace('"','')+'"')
      except:
        self.addDefine('MPICC_SHOW','"Unavailable"')
    else:
      self.addDefine('MPICC_SHOW','"Unavailable"')
    self.setCompilers.popLanguage()
#-----------------------------------------------------------------------------------------------------

    # Sometimes we need C compiler, even if built with C++
    self.setCompilers.pushLanguage('C')
    # do not use getCompilerFlags() because that automatically includes the CPPFLAGS so one ends up with duplication flags in makefile usage
    self.addMakeMacro('CC_FLAGS',self.setCompilers.CFLAGS)
    self.setCompilers.popLanguage()

    # And sometimes we need a C++ compiler even when PETSc is built with C
    if hasattr(self.compilers, 'CXX'):
      self.setCompilers.pushLanguage('Cxx')
      self.addDefine('HAVE_CXX','1')
      self.addMakeMacro('CXXPP_FLAGS',self.setCompilers.CXXPPFLAGS)
      # do not use getCompilerFlags() because that automatically includes the CXXPPFLAGS so one ends up with duplication flags in makefile usage
      self.addMakeMacro('CXX_FLAGS',self.setCompilers.CXXFLAGS+' '+self.setCompilers.CXX_CXXFLAGS)
      cxx_linker = self.setCompilers.getLinker()
      self.addMakeMacro('CXX_LINKER',cxx_linker)
      self.addMakeMacro('CXX_LINKER_FLAGS',self.setCompilers.getLinkerFlags())
      self.setCompilers.popLanguage()
    else:
      self.addMakeMacro('CXX','')

    # C preprocessor values
    self.addMakeMacro('CPP_FLAGS',self.setCompilers.CPPFLAGS)

    # compiler values
    self.setCompilers.pushLanguage(self.languages.clanguage)
    self.addMakeMacro('PCC',self.setCompilers.getCompiler())
    # do not use getCompilerFlags() because that automatically includes the preprocessor flags so one ends up with duplication flags in makefile usage
    if self.languages.clanguage == 'C':
      self.addMakeMacro('PCC_FLAGS','$(CC_FLAGS)')
    else:
      self.addMakeMacro('PCC_FLAGS','$(CXX_FLAGS)')
    self.setCompilers.popLanguage()
    # .o or .obj
    self.addMakeMacro('CC_SUFFIX','o')

    # executable linker values
    self.setCompilers.pushLanguage(self.languages.clanguage)
    pcc_linker = self.setCompilers.getLinker()
    self.addMakeMacro('PCC_LINKER',pcc_linker)
    # We need to add sycl flags when linking petsc. See more in sycl.py.
    if hasattr(self.compilers, 'SYCLC'):
      self.addMakeMacro('PCC_LINKER_FLAGS',self.setCompilers.getLinkerFlags()+' '+self.setCompilers.SYCLFLAGS+' '+self.setCompilers.SYCLC_LINKER_FLAGS)
    else:
      self.addMakeMacro('PCC_LINKER_FLAGS',self.setCompilers.getLinkerFlags())
    self.setCompilers.popLanguage()
    # '' for Unix, .exe for Windows
    self.addMakeMacro('CC_LINKER_SUFFIX','')

    if hasattr(self.compilers, 'FC'):
      if self.framework.argDB['with-fortran-bindings']:
        if not self.fortran.fortranIsF90:
          raise RuntimeError('Error! Fortran compiler "'+self.compilers.FC+'" does not support F90! PETSc fortran bindings require a F90 compiler')
        self.addDefine('HAVE_FORTRAN','1')
      self.setCompilers.pushLanguage('FC')
      # need FPPFLAGS in config/setCompilers
      self.addMakeMacro('FPP_FLAGS',self.setCompilers.FPPFLAGS)

      # compiler values
      self.addMakeMacro('FC_FLAGS',self.setCompilers.getCompilerFlags())
      self.setCompilers.popLanguage()
      # .o or .obj
      self.addMakeMacro('FC_SUFFIX','o')

      # executable linker values
      self.setCompilers.pushLanguage('FC')
      self.addMakeMacro('FC_LINKER',self.setCompilers.getLinker())
      self.addMakeMacro('FC_LINKER_FLAGS',self.setCompilers.getLinkerFlags())
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
      self.addMakeMacro('CUDAPP_FLAGS',self.setCompilers.CUDAPPFLAGS)
      self.setCompilers.popLanguage()

    if hasattr(self.compilers, 'HIPC'):
      self.setCompilers.pushLanguage('HIP')
      self.addMakeMacro('HIPC_FLAGS',self.setCompilers.getCompilerFlags())
      self.addMakeMacro('HIPPP_FLAGS',self.setCompilers.HIPPPFLAGS)
      self.setCompilers.popLanguage()

    if hasattr(self.compilers, 'SYCLC'):
      self.setCompilers.pushLanguage('SYCL')
      self.addMakeMacro('SYCLC_FLAGS',self.setCompilers.getCompilerFlags())
      self.addMakeMacro('SYCLC_LINKER_FLAGS',self.setCompilers.getLinkerFlags())
      self.addMakeMacro('SYCLPP_FLAGS',self.setCompilers.SYCLPPFLAGS)
      self.setCompilers.popLanguage()

    # Avoid picking CFLAGS etc from env - but support 'make CFLAGS=-Werror' etc..
    self.addMakeMacro('CFLAGS','')
    self.addMakeMacro('CPPFLAGS','')
    self.addMakeMacro('CXXFLAGS','')
    self.addMakeMacro('CXXPPFLAGS','')
    self.addMakeMacro('FFLAGS','')
    self.addMakeMacro('FPPFLAGS','')
    self.addMakeMacro('CUDAFLAGS','')
    self.addMakeMacro('CUDAPPFLAGS','')
    self.addMakeMacro('HIPFLAGS','')
    self.addMakeMacro('HIPPPFLAGS','')
    self.addMakeMacro('SYCLFLAGS','')
    self.addMakeMacro('SYCLPPFLAGS','')
    self.addMakeMacro('LDFLAGS','')

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

#-----------------------------------------------------------------------------------------------------
    # print include and lib for makefiles
    self.logPrintDivider()
    self.framework.packages.reverse()
    petscincludes = [os.path.join(self.petscdir.dir,'include'),os.path.join(self.petscdir.dir,self.arch.arch,'include')]
    petscincludes_install = [os.path.join(self.installdir.dir, 'include')] if self.framework.argDB['prefix'] else petscincludes
    includes = []
    self.packagelibs = []
    for i in self.framework.packages:
      if not i.required:
        if i.devicePackage:
          self.addDefine('HAVE_DEVICE',1)
        self.addDefine('HAVE_'+i.PACKAGE.replace('-','_'), 1)  # ONLY list package if it is used directly by PETSc (and not only by another package)
      if not isinstance(i.lib, list):
        i.lib = [i.lib]
      if i.linkedbypetsc: self.packagelibs.extend(i.lib)
      self.addMakeMacro(i.PACKAGE.replace('-','_')+'_LIB', self.libraries.toStringNoDupes(i.lib))
      if hasattr(i,'include'):
        if not isinstance(i.include,list):
          i.include = [i.include]
        includes.extend(i.include)
        self.addMakeMacro(i.PACKAGE.replace('-','_')+'_INCLUDE',self.headers.toStringNoDupes(i.include))
    if self.framework.argDB['with-single-library']:
      self.petsclib = '-lpetsc'
    else:
      self.petsclib = '-lpetscts -lpetscsnes -lpetscksp -lpetscdm -lpetscmat -lpetscvec -lpetscsys'
    self.complibs = self.compilers.flibs+self.compilers.cxxlibs+self.compilers.LIBS.split()
    self.PETSC_WITH_EXTERNAL_LIB = self.libraries.toStringNoDupes(['-L${PETSC_DIR}/${PETSC_ARCH}/lib', self.petsclib]+self.packagelibs+self.complibs)
    self.PETSC_EXTERNAL_LIB_BASIC = self.libraries.toStringNoDupes(self.packagelibs+self.complibs)

    self.addMakeMacro('PETSC_EXTERNAL_LIB_BASIC',self.PETSC_EXTERNAL_LIB_BASIC)
    allincludes = petscincludes + includes
    allincludes_install = petscincludes_install + includes
    self.PETSC_CC_INCLUDES = self.headers.toStringNoDupes(allincludes)
    self.PETSC_CC_INCLUDES_INSTALL = self.headers.toStringNoDupes(allincludes_install)
    self.addMakeMacro('PETSC_CC_INCLUDES',self.PETSC_CC_INCLUDES)
    self.addMakeMacro('PETSC_CC_INCLUDES_INSTALL', self.PETSC_CC_INCLUDES_INSTALL)
    if hasattr(self.compilers, 'FC'):
      def modinc(includes):
        return includes if self.fortran.fortranIsF90 else []
      self.addMakeMacro('PETSC_FC_INCLUDES',self.headers.toStringNoDupes(allincludes,modinc(allincludes)))
      self.addMakeMacro('PETSC_FC_INCLUDES_INSTALL',self.headers.toStringNoDupes(allincludes_install,modinc(allincludes_install)))

    self.addDefine('LIB_DIR','"'+os.path.join(self.installdir.dir,'lib')+'"')

    if self.framework.argDB['with-single-library']:
      # overrides the values set in conf/variables
      self.addMakeMacro('LIBNAME','${INSTALL_LIB_DIR}/libpetsc.${AR_LIB_SUFFIX}')
      self.addMakeMacro('SHLIBS','libpetsc')
      self.addMakeMacro('PETSC_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_KSP_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_TS_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_TAO_LIB_BASIC','-lpetsc')
      self.addMakeMacro('PETSC_WITH_EXTERNAL_LIB',self.PETSC_WITH_EXTERNAL_LIB)
      self.addDefine('USE_SINGLE_LIBRARY', '1')
      if self.sharedlibraries.useShared:
        self.addMakeMacro('PETSC_SYS_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_VEC_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_MAT_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_DM_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_KSP_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_SNES_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_TS_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_TAO_LIB','${C_SH_LIB_PATH} ${PETSC_WITH_EXTERNAL_LIB}')
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
        self.addMakeMacro('PETSC_TAO_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_CHARACTERISTIC_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_LIB','${PETSC_WITH_EXTERNAL_LIB}')
        self.addMakeMacro('PETSC_CONTRIB','${PETSC_WITH_EXTERNAL_LIB}')

    if not os.path.exists(os.path.join(self.petscdir.dir,self.arch.arch,'lib')):
      os.makedirs(os.path.join(self.petscdir.dir,self.arch.arch,'lib'))

# add a makefile endtry for display
    if self.framework.argDB['with-display']:
      self.addMakeMacro('DISPLAY',self.framework.argDB['with-display'])

    # add a makefile entry for configure options
    self.addMakeMacro('CONFIGURE_OPTIONS', self.framework.getOptionsString(['configModules', 'optionsModule']).replace('\"','\\"'))

    if self.framework.argDB['with-tau-perfstubs']:
      self.addDefine('HAVE_TAU_PERFSTUBS',1)
    return

  def dumpConfigInfo(self):
    import time
    fd = open(os.path.join(self.arch.arch,'include','petscconfiginfo.h'),'w')
    fd.write('static const char *petscconfigureoptions = "'+self.framework.getOptionsString(['configModules', 'optionsModule']).replace('\"','\\"').replace('\\ ','\\\\ ')+'";\n')
    fd.close()
    return

  def dumpMachineInfo(self):
    import platform
    import datetime
    import time
    import script
    def escape(s):
      return s.replace('"',r'\"').replace(r'\ ',r'\\ ') # novermin
    fd = open(os.path.join(self.arch.arch,'include','petscmachineinfo.h'),'w')
    fd.write('static const char *petscmachineinfo = \"\\n\"\n')
    fd.write('\"-----------------------------------------\\n\"\n')
    buildhost = platform.node()
    if os.environ.get('SOURCE_DATE_EPOCH'):
      buildhost = "reproducible"
    buildtime = datetime.datetime.utcfromtimestamp(int(os.environ.get('SOURCE_DATE_EPOCH', time.time())))
    fd.write('\"Libraries compiled on %s on %s \\n\"\n' % (buildtime, buildhost))
    fd.write('\"Machine characteristics: %s\\n\"\n' % (platform.platform()))
    fd.write('\"Using PETSc directory: %s\\n\"\n' % (escape(self.installdir.petscDir)))
    fd.write('\"Using PETSc arch: %s\\n\"\n' % (escape(self.installdir.petscArch)))
    fd.write('\"-----------------------------------------\\n\";\n')
    fd.write('static const char *petsccompilerinfo = \"\\n\"\n')
    self.setCompilers.pushLanguage(self.languages.clanguage)
    fd.write('\"Using C compiler: %s %s \\n\"\n' % (escape(self.setCompilers.getCompiler()), escape(self.setCompilers.getCompilerFlags())))
    self.setCompilers.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      fd.write('\"Using Fortran compiler: %s %s  %s\\n\"\n' % (escape(self.setCompilers.getCompiler()), escape(self.setCompilers.getCompilerFlags()), escape(self.setCompilers.CPPFLAGS)))
      self.setCompilers.popLanguage()
    fd.write('\"-----------------------------------------\\n\";\n')
    fd.write('static const char *petsccompilerflagsinfo = \"\\n\"\n')
    fd.write('\"Using include paths: %s\\n\"\n' % (escape(self.PETSC_CC_INCLUDES_INSTALL.replace('${PETSC_DIR}', self.installdir.petscDir))))
    fd.write('\"-----------------------------------------\\n\";\n')
    fd.write('static const char *petsclinkerinfo = \"\\n\"\n')
    self.setCompilers.pushLanguage(self.languages.clanguage)
    fd.write('\"Using C linker: %s\\n\"\n' % (escape(self.setCompilers.getLinker())))
    self.setCompilers.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      fd.write('\"Using Fortran linker: %s\\n\"\n' % (escape(self.setCompilers.getLinker())))
      self.setCompilers.popLanguage()
    fd.write('\"Using libraries: %s%s -L%s %s %s\\n\"\n' % (escape(self.setCompilers.CSharedLinkerFlag), escape(os.path.join(self.installdir.petscDir, self.installdir.petscArch, 'lib')), escape(os.path.join(self.installdir.petscDir, self.installdir.petscArch, 'lib')), escape(self.petsclib), escape(self.PETSC_EXTERNAL_LIB_BASIC)))
    fd.write('\"-----------------------------------------\\n\";\n')
    fd.close()
    return

  def configurePrefetch(self):
    '''Sees if there are any prefetch functions supported'''
    if config.setCompilers.Configure.isSolaris(self.log) or self.framework.argDB['with-ios'] or not self.framework.argDB['with-prefetch']:
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
      # [1] https://software.intel.com/file/6373
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

  def delGenFiles(self):
    '''Delete generated files'''
    delfile = os.path.join(self.arch.arch,'lib','petsc','conf','files')
    try:
      os.unlink(delfile)
    except: pass

  def configureAtoll(self):
    '''Checks if atoll exists'''
    if self.checkLink('#define _POSIX_C_SOURCE 200112L\n#include <stdlib.h>','long v = atoll("25");\n(void)v') or self.checkLink ('#include <stdlib.h>','long v = atoll("25");\n(void)v'):
       self.addDefine('HAVE_ATOLL', '1')

  def configureUnused(self):
    '''Sees if __attribute((unused)) is supported'''
    if self.framework.argDB['with-ios']:
      self.addDefine('UNUSED', ' ')
      return
    self.pushLanguage(self.languages.clanguage)
    if self.checkLink('__attribute((unused)) static int myfunc(__attribute((unused)) void *name){ return 1;}', 'int i = 0;\nint j = myfunc(&i);\n(void)j;\ntypedef void* atype;\n__attribute((unused))  atype a'):
      self.addDefine('UNUSED', '__attribute((unused))')
    else:
      self.addDefine('UNUSED', ' ')
    self.popLanguage()

  def configureIsatty(self):
    '''Check if the Unix C function isatty() works correctly
       Actually just assumes it does not work correctly on batch systems'''
    if not self.framework.argDB['with-batch']:
      self.addDefine('USE_ISATTY',1)

  def configureDeprecated(self):
    '''Check if __attribute((deprecated)) is supported'''
    def checkDeprecated(macro_base, src, is_intel):
      '''
      run through the various attribute deprecated combinations and define MACRO_BAS(why) to the result
      it if it compiles.

      If none of the combos work, defines MACRO_BASE(why) as empty
      '''
      full_macro_name = macro_base + '(why)'
      for prefix in ('__attribute__', '__attribute','__declspec'):
        if prefix == '__declspec':
          # declspec does not have an extra set of brackets around the arguments
          attr_bodies = ('deprecated(why)', 'deprecated')
        else:
          attr_bodies = ('(deprecated(why))', '(deprecated)')

        for attr_body in attr_bodies:
          attr_def = '{}({})'.format(prefix, attr_body)
          test_src = '\n'.join((
            '#define {} {}'.format(full_macro_name, attr_def),
            src.format(macro_base + '("asdasdadsasd")')
          ))
          if self.checkCompile(test_src, ''):
            self.logPrint('configureDeprecated: \'{}\' appears to work'.format(attr_def))
            if is_intel and '(why)' in attr_body:
              self.logPrint('configureDeprecated: Intel has conspired to make a supremely environment-sensitive compiler. The Intel compiler looks at the gcc executable in the environment to determine the language compatibility that it should attempt to emulate. Some important Cray installations have built PETSc using the Intel compiler, but with a newer gcc module loaded (e.g. 4.7). Thus at PETSc configure time, the Intel compiler decides to support the string argument, but the gcc found in the default user environment is older and does not support the argument.\n'.format(attr_def))
              self.logPrint('*** WE WILL THEREFORE REJECT \'{}\' AND CONTINUE TESTING ***'.format(attr_def))
              continue
            self.addDefine(full_macro_name, attr_def)
            return

      self.addDefine(full_macro_name, ' ')
      return

    lang = self.languages.clanguage
    with self.Language(lang):
      is_intel = self.setCompilers.isIntel(self.getCompiler(lang=lang), self.log)
      checkDeprecated('DEPRECATED_FUNCTION', '{} int myfunc(void) {{ return 1; }}', is_intel)
      checkDeprecated('DEPRECATED_TYPEDEF', 'typedef int my_int {};', is_intel)
      checkDeprecated('DEPRECATED_ENUM', 'enum E {{ oldval {}, newval }};', is_intel)
      # I was unable to make a CPP macro that takes the old and new values as separate
      # arguments and builds the message needed by _Pragma hence the deprecation message is
      # handled as it is
      if self.checkCompile('#define TEST _Pragma("GCC warning \"Testing _Pragma\"") value'):
        self.addDefine('DEPRECATED_MACRO(why)', '_Pragma(why)')
      else:
        self.addDefine('DEPRECATED_MACRO(why)', ' ')

  def configureAlign(self):
    '''Check if __attribute(aligned) is supported'''
    code = '''\
struct mystruct {int myint;} __attribute((aligned(16)));
char assert_aligned[(sizeof(struct mystruct)==16)*2-1];
'''
    self.pushLanguage(self.languages.clanguage)
    if self.checkCompile(code):
      self.addDefine('ATTRIBUTEALIGNED(size)', '__attribute((aligned(size)))')
      self.addDefine('HAVE_ATTRIBUTEALIGNED', 1)
    else:
      self.framework.logPrint('Incorrect attribute(aligned)')
      self.addDefine('ATTRIBUTEALIGNED(size)', ' ')
    self.popLanguage()
    return

  def configureExpect(self):
    '''Sees if the __builtin_expect directive is supported'''
    self.pushLanguage(self.languages.clanguage)
    if self.checkLink('', 'if (__builtin_expect(0,1)) return 1;'):
      self.addDefine('HAVE_BUILTIN_EXPECT', 1)
    self.popLanguage()

  def configureFunctionName(self):
    '''Sees if the compiler supports __func__ or a variant.'''
    def getFunctionName(lang):
      name = '"unknown"'
      self.pushLanguage(lang)
      for fname in ['__func__','__FUNCTION__','__extension__ __func__']:
        code = "if ("+fname+"[0] != 'm') return 1;"
        if self.checkCompile('',code) and self.checkLink('',code):
          name = fname
          break
      self.popLanguage()
      return name
    langs = []

    self.addDefine('FUNCTION_NAME_C', getFunctionName('C'))
    if hasattr(self.compilers, 'CXX'):
      self.addDefine('FUNCTION_NAME_CXX', getFunctionName('Cxx'))

  def configureIntptrt(self):
    '''Determine what to use for uintptr_t and intptr_t'''
    def staticAssertSizeMatchesVoidStar(inc,typename):
      # The declaration is an error if either array size is negative.
      # It should be okay to use an int that is too large, but it would be very unlikely for this to be the case
      return self.checkCompile(inc, ('#define STATIC_ASSERT(cond) char negative_length_if_false[2*(!!(cond))-1]\n'
                                     + 'STATIC_ASSERT(sizeof(void*) == sizeof(%s));'%typename))

    def generate_uintptr_guesses():
      for suff in ('max', '64', '32', '16'):
        yield '#include <stdint.h>', 'uint{}_t'.format(suff), 'PRIx{}'.format(suff.upper())
      yield '#include <stdlib.h>\n#include <string.h>', 'size_t', 'zx'
      yield '', 'unsigned long long', 'llx'
      yield '', 'unsigned long', 'lx'
      yield '', 'unsigned', 'x'

    def generate_intptr_guesses():
      for suff in ('max', '64', '32', '16'):
        yield '#include <stdint.h>', 'int{}_t'.format(suff), 'PRIx{}'.format(suff.upper())
      yield '', 'long long', 'llx'
      yield '', 'long', 'lx'
      yield '', 'int', 'x'

    def check(default_typename, generator):
      macro_name = default_typename.upper()
      with self.Language(self.languages.clanguage):
        if self.checkCompile(
            '#include <stdint.h>',
            'int x; {type_name} i = ({type_name})&x; (void)i'.format(type_name=default_typename)
        ):
          typename     = default_typename
          print_format = 'PRIxPTR'
        else:
          for include, typename, print_format in generator():
            if staticAssertSizeMatchesVoidStar(include, typename):
              break
          else:
            raise RuntimeError('Could not find any {} type matching void*'.format(macro_name))
      self.addDefine(macro_name         , typename)
      self.addDefine(macro_name + '_FMT', '\"#\" ' + print_format)
      return

    check('uintptr_t', generate_uintptr_guesses)
    check('intptr_t', generate_intptr_guesses)
    return

  def configureRTLDDefault(self):
    '''Check for dynamic library feature'''
    if self.checkCompile('#include <dlfcn.h>\n void *ptr =  RTLD_DEFAULT;'):
      self.addDefine('HAVE_RTLD_DEFAULT','1')
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

  def configureDarwin(self):
    '''Log brew configuration for Apple systems'''
    try:
      self.executeShellCommand(['brew', 'config'], log = self.log)
      self.executeShellCommand(['brew', 'info', 'gcc'], log = self.log)
    except:
      pass
    return

  def configureLinux(self):
    '''Linux specific stuff'''
    # TODO: Test for this by mallocing an odd number of floats and checking the address
    self.addDefine('HAVE_DOUBLE_ALIGN_MALLOC', 1)
    return

  def configureWin32(self):
    '''Win32 non-cygwin specific stuff'''
    kernel32=0
    if self.libraries.add('Kernel32.lib','GetComputerName',prototype='#include <windows.h>', call='GetComputerName(NULL,NULL);'):
      self.addDefine('HAVE_WINDOWS_H',1)
      self.addDefine('HAVE_GETCOMPUTERNAME',1)
      kernel32=1
    elif self.libraries.add('kernel32','GetComputerName',prototype='#include <windows.h>', call='GetComputerName(NULL,NULL);'):
      self.addDefine('HAVE_WINDOWS_H',1)
      self.addDefine('HAVE_GETCOMPUTERNAME',1)
      kernel32=1
    if kernel32:
      if self.framework.argDB['with-windows-graphics']:
        self.addDefine('USE_WINDOWS_GRAPHICS',1)
      if self.checkLink('#include <windows.h>','LoadLibrary(0)'):
        self.addDefine('HAVE_LOADLIBRARY',1)
      if self.checkLink('#include <windows.h>','GetProcAddress(0,0)'):
        self.addDefine('HAVE_GETPROCADDRESS',1)
      if self.checkLink('#include <windows.h>','FreeLibrary(0)'):
        self.addDefine('HAVE_FREELIBRARY',1)
      if self.checkLink('#include <windows.h>','GetLastError()'):
        self.addDefine('HAVE_GETLASTERROR',1)
      if self.checkLink('#include <windows.h>','SetLastError(0)'):
        self.addDefine('HAVE_SETLASTERROR',1)
      if self.checkLink('#include <windows.h>\n','QueryPerformanceCounter(0);\n'):
        self.addDefine('USE_MICROSOFT_TIME',1)
    if self.libraries.add('Advapi32.lib','GetUserName',prototype='#include <windows.h>', call='GetUserName(NULL,NULL);'):
      self.addDefine('HAVE_GET_USER_NAME',1)
    elif self.libraries.add('advapi32','GetUserName',prototype='#include <windows.h>', call='GetUserName(NULL,NULL);'):
      self.addDefine('HAVE_GET_USER_NAME',1)

    if not self.libraries.add('User32.lib','GetDC',prototype='#include <windows.h>',call='GetDC(0);'):
      self.libraries.add('user32','GetDC',prototype='#include <windows.h>',call='GetDC(0);')
    if not self.libraries.add('Gdi32.lib','CreateCompatibleDC',prototype='#include <windows.h>',call='CreateCompatibleDC(0);'):
      self.libraries.add('gdi32','CreateCompatibleDC',prototype='#include <windows.h>',call='CreateCompatibleDC(0);')

    self.types.check('int32_t', 'int')
    if not self.checkCompile('#include <sys/types.h>\n','uid_t u;\n(void)u'):
      self.addTypedef('int', 'uid_t')
      self.addTypedef('int', 'gid_t')
    if not self.checkLink('#if defined(PETSC_HAVE_UNISTD_H)\n#include <unistd.h>\n#endif\n','int a=R_OK;\n(void)a'):
      self.framework.addDefine('R_OK', '04')
      self.framework.addDefine('W_OK', '02')
      self.framework.addDefine('X_OK', '01')
    if not self.checkLink('#include <sys/stat.h>\n','int a=0;\nif (S_ISDIR(a)){}\n'):
      self.framework.addDefine('S_ISREG(a)', '(((a)&_S_IFMT) == _S_IFREG)')
      self.framework.addDefine('S_ISDIR(a)', '(((a)&_S_IFMT) == _S_IFDIR)')
    if self.checkCompile('#include <windows.h>\n','LARGE_INTEGER a;\nDWORD b=a.u.HighPart;\n'):
      self.addDefine('HAVE_LARGE_INTEGER_U',1)

    # Windows requires a Binary file creation flag when creating/opening binary files.  Is a better test in order?
    if self.checkCompile('#include <windows.h>\n#include <fcntl.h>\n', 'int flags = O_BINARY;'):
      self.addDefine('HAVE_O_BINARY',1)

    if self.compilers.CC.find('win32fe') >= 0:
      self.addDefine('HAVE_WINDOWS_COMPILERS',1)
      self.addDefine('DIR_SEPARATOR','\'\\\\\'')
      self.addDefine('REPLACE_DIR_SEPARATOR','\'/\'')
      self.addDefine('CANNOT_START_DEBUGGER',1)
      (petscdir,error,status) = self.executeShellCommand('cygpath -w '+self.installdir.petscDir, log = self.log)
      self.addDefine('DIR','"'+petscdir.replace('\\','\\\\')+'"')
      (petscdir,error,status) = self.executeShellCommand('cygpath -m '+self.installdir.petscDir, log = self.log)
      self.addMakeMacro('wPETSC_DIR',petscdir)
      if self.dataFilesPath.datafilespath:
        (datafilespath,error,status) = self.executeShellCommand('cygpath -m '+self.dataFilesPath.datafilespath, log = self.log)
        self.addMakeMacro('DATAFILESPATH',datafilespath)

    else:
      self.addDefine('REPLACE_DIR_SEPARATOR','\'\\\\\'')
      self.addDefine('DIR_SEPARATOR','\'/\'')
      self.addDefine('DIR','"'+self.installdir.petscDir+'"')
      self.addMakeMacro('wPETSC_DIR',self.installdir.petscDir)
      if self.dataFilesPath.datafilespath:
        self.addMakeMacro('DATAFILESPATH',self.dataFilesPath.datafilespath)
    self.addDefine('ARCH','"'+self.installdir.petscArch+'"')
    return

  def configureCoverageForLang(self, log_printer_cls, lang, extra_coverage_flags=None, extra_debug_flags=None):
    """
    Check that a compiler accepts code-coverage flags. If the compiler does accept code-coverage flags
    try to set debugging flags equivalent to -Og.

    Arguments:
    - lang: the language to check the coverage flag for
    - extra_coverage_flags: a list of extra flags to use when checking the coverage flags
    - extra_debug_flags: a list of extra flags to try when setting debug flags

    On success:
    - defines PETSC_USE_COVERAGE to 1
    """
    log_print = log_printer_cls(self)

    def quoted(string):
      return string.join(("'", "'"))

    def make_flag_list(default, extra):
      ret = [default]
      if extra is not None:
        assert isinstance(extra, list)
        ret.extend(extra)
      return ret

    log_print('Checking coverage flag for language {}'.format(lang))

    compiler = self.getCompiler(lang=lang)
    if self.setCompilers.isGNU(compiler, self.log):
      is_gnuish = True
    elif self.setCompilers.isClang(compiler, self.log):
      is_gnuish = True
    else:
      is_gnuish = False

    # if not gnuish and we don't have a set of extra flags, bail
    if not is_gnuish and extra_coverage_flags is None:
      log_print('Don\'t know how to add coverage for compiler {}. Only know how to add coverage for gnu-like compilers (either gcc or clang). Skipping it!'.format(quoted(compiler)))
      return

    coverage_flags = make_flag_list('--coverage', extra_coverage_flags)
    log_print('Checking set of coverage flags: {}'.format(coverage_flags))

    found = None
    with self.Language(lang):
      with self.setCompilers.Language(lang):
        for flag in coverage_flags:
          # the linker also needs to see the coverage flag
          with self.setCompilers.extraCompilerFlags([flag], compilerOnly=False) as skip_flags:
            if not skip_flags and self.checkRun():
              # flag was accepted
              found = flag
              break

          log_print(
            'Compiler {} did not accept coverage flag {}'.format(quoted(compiler), quoted(flag))
          )

        if found is None:
          log_print(
            'Compiler {} did not accept ANY coverage flags: {}, bailing!'.format(
              quoted(compiler), coverage_flags
            )
          )
          return

        # must do this exactly here since:
        #
        # 1. setCompilers.extraCompilerFlags() will reset the compiler flags on __exit__()
        #    (so cannot do it in the loop)
        # 2. we need to set the compiler flag while setCompilers.Language() is still in
        #    effect (so cannot do it outside the with statements)
        self.setCompilers.insertCompilerFlag(flag, False)

    if not self.functions.haveFunction('__gcov_dump'):
      self.functions.checkClassify(['__gcov_dump'])

    # now check if we can override the optimization level. It is only kosher to do so if
    # the user did not explicitly set the optimization flags (via CFLAGS, CXXFLAGS,
    # CXXOPTFLAGS, etc). If they have done so, we sternly warn them about their lapse in
    # judgement
    with self.Language(lang):
      compiler_flags = self.getCompilerFlags()

    user_set          = 0
    allowed_opt_flags = re.compile(r'|'.join((r'-O[01g]', r'-g[1-9]*')))
    for flagsname in [self.getCompilerFlagsName(lang), self.compilerFlags.getOptionalFlagsName(lang)]:
      if flagsname in self.argDB:
        opt_flags = [
          f for f in self.compilerFlags.findOptFlags(compiler_flags) if not allowed_opt_flags.match(f)
        ]
        if opt_flags:
          self.logPrintWarning('Coverage requested, but optimization flag(s) {} found in {}. Coverage collection will work, but may be less accurate. Suggest removing the flag and/or using -Og (or equivalent) instead'.format(', '.join(map(quoted, opt_flags)), quoted(flagsname)))
          user_set = 1
          break

    # disable this for now, the warning should be sufficient. If the user still chooses to
    # ignore it, then that's on them
    if 0 and not user_set:
      debug_flags = make_flag_list('-Og', extra_debug_flags)
      with self.setCompilers.Language(lang):
        for flag in debug_flags:
          try:
            self.setCompilers.addCompilerFlag(flag)
          except RuntimeError:
            continue
          break

    self.addDefine('USE_COVERAGE', 1)
    return

  def configureCoverage(self):
    """
    Configure coverage for all available languages.

    If user did not request coverage, this function does nothing and returns immediatel.
    Therefore the following only apply to the case where the user requested coverage.

    On success:
    - defines PETSC_USE_COVERAGE to 1

    On failure:
    - If no compilers supported the coverage flag, throws RuntimeError
    -
    """
    class LogPrinter:
      def __init__(self, cfg):
        self.cfg = cfg
        try:
          import inspect

          calling_func_stack = inspect.stack()[1]
          if sys.version_info >= (3, 5):
            func_name = calling_func_stack.function
          else:
            func_name = calling_func_stack[3]
        except:
          func_name = 'Unknown'
        self.fmt_str = func_name + '(): {}'

      def __call__(self, msg, *args, **kwargs):
        return self.cfg.logPrint(self.fmt_str.format(msg), *args, **kwargs)

    argdb_flag = 'with-coverage'
    log_print  = LogPrinter(self)
    if not self.argDB[argdb_flag]:
      log_print('coverage was disabled from command line or default')
      return

    tested_langs = []
    for LANG in ['C', 'Cxx', 'CUDA', 'HIP', 'SYCL', 'FC']:
      compilerName = LANG.upper() if LANG in {'Cxx', 'FC'} else LANG + 'C'
      if hasattr(self.setCompilers, compilerName):
        kwargs = {}
        if LANG in {'CUDA'}:
          # nvcc preprocesses the base file into a bunch of intermediate files, which are
          # then compiled by the host compiler. Why is this a problem?  Because the
          # generated coverage data is based on these preprocessed source files! So gcov
          # tries to read it later, but since its in the tmp directory it cannot. Thus we
          # need to keep them around (in a place we know about).
          nvcc_tmp_dir = os.path.join(self.petscdir.dir, self.arch.arch, 'nvcc_tmp')
          try:
            os.mkdir(nvcc_tmp_dir)
          except FileExistsError:
            pass
          kwargs['extra_coverage_flags'] = [
            '-Xcompiler --coverage -Xcompiler -fPIC --keep --keep-dir={}'.format(nvcc_tmp_dir)
          ]
          if self.kokkos.found:
            # yet again the kokkos nvcc_wrapper goes out of its way to be as useless as
            # possible. Its default arch (sm_35) is actually too low to compile kokkos,
            # for whatever reason this works if you dont use the --keep and --keep-dir
            # flags above.
            kwargs['extra_coverage_flags'].append('-arch=native')
            kwargs['extra_debug_flags'] = ['-Xcompiler -Og']
        tested_langs.append(LANG)
        self.executeTest(self.configureCoverageForLang, args=[LogPrinter, LANG], kargs=kwargs)

    if not self.defines.get('USE_COVERAGE'):
      # coverage was requested but no compilers accepted it, this is an error
      raise RuntimeError(
        'Coverage was requested (--{}={}) but none of the compilers supported it:\n{}\n'.format(
          argdb_flag, self.argDB[argdb_flag],
          '\n'.join(['  - {} ({})'.format(self.getCompiler(lang=lang), lang) for lang in tested_langs])
        )
      )

    return
    # Disabled for now, since this does not really work. It solves the problem of
    # "undefined reference to __gcov_flush()" but if we add -lgcov we get:
    #
    # duplicate symbol '___gcov_reset' in:
    #     /Library/.../libclang_rt.profile_osx.a(GCDAProfiling.c.o)
    #     /opt/.../libgcov.a(_gcov_reset.o)
    # duplicate symbol '___gcov_dump' in:
    #     /opt/.../libgcov.a(_gcov_dump.o)
    #     /Library/.../libclang_rt.profile_osx.a(GCDAProfiling.c.o)
    # duplicate symbol '___gcov_fork' in:
    #     /opt/.../libgcov.a(_gcov_fork.o)
    #     /Library/.../libclang_rt.profile_osx.a(GCDAProfiling.c.o)
    #
    # I don't know how to solve this.

    log_print('Checking if compilers can cross-link disparate coverage libraries')
    # At least one of the compilers has coverage enabled. Now need to make sure multiple
    # code coverage impls work together, specifically when using clang C/C++ compiler with
    # gfortran.
    if not hasattr(self.setCompilers, 'FC'):
      log_print('No fortran compiler detected. No need to check cross-linking!')
      return

    c_lang = self.languages.clanguage
    if not self.setCompilers.isClang(self.getCompiler(lang=c_lang), self.log):
      # must be GCC
      log_print('C-language ({}) compiler is not clang, assuming it is GCC, so cross-linking with FC ({}) assumed to be OK'.format(c_lang, self.getCompiler(lang='FC')))
      return

    # If we are here we:
    #   1. Have both C/C++ compiler and fortran compiler
    #   2. The C/C++ compiler is *not* the same as the fortran compiler (unless we start
    #      using flang)
    #
    # Now we check if we can cross-link
    def can_cross_link(**kwargs):
      f_body = "      subroutine foo()\n      print*,'testing'\n      return\n      end\n"
      c_body = "int main() { }"

      return self.compilers.checkCrossLink(
        f_body, c_body, language1='FC', language2=c_lang, extralibs=self.compilers.flibs, **kwargs
      )

    log_print('Trying to cross-link WITHOUT extra libs')
    if can_cross_link():
      log_print('Successfully cross-linked WITHOUT extra libs')
      # success, we already can cross-link
      return

    extra_libs = ['-lgcov']
    log_print('Trying to cross-link with extra libs: {}'.format(extra_libs))
    if can_cross_link(extraObjs=extra_libs):
      log_print(
        'Successfully cross-linked using extra libs: {}, adding them to LIBS'.format(extra_libs)
      )
      self.setCompilers.LIBS += ' ' + ' '.join(extra_libs)
    else:
      # maybe should be an error?
      self.logPrintWarning("Could not successfully cross-link covered code between {} and FC. Sometimes this is a false positive. Assuming this does eventually end up working when the full link-line is assembled when building PETSc. If you later encounter linker errors about missing __gcov_exit(), __gcov_init(), __llvm_cov_flush() etc. this is why!".format(c_lang))
    return

  def configureCoverageExecutable(self):
    """
    Check that a code-coverage collecting tool exists and is on PATH.

    On success:
    - Adds PETSC_COVERAGE_EXEC make macro containing the full path to the coverage tool executable.

    Raises RuntimeError if:
    - User explicitly requests auto-detection of the coverage tool from command line, and this
      routine fails to guess the suitable tool name.
    - The routine fails to find the tool, and --with-coverage is true
    """
    def log_print(msg, *args, **kwargs):
      self.logPrint('checkCoverage: '+str(msg), *args, **kwargs)
      return

    def quoted(string):
      return string.join(("'", "'"))

    required         = bool(self.argDB['with-coverage'])
    arg_opt          = self.argDB['with-coverage-exec']
    use_default_path = True
    search_path      = ''

    log_print('{} to find an executable'.format('REQUIRED' if required else 'NOT required'))
    if arg_opt in {'auto', 'default-auto', '1'}:
      # detect it based on the C language compiler, hopefully this does not clash!
      lang     = self.setCompilers.languages.clanguage
      compiler = self.getCompiler(lang=lang)
      log_print('User did not explicitly set coverage exec (got {}), trying to auto-detect based on compiler {}'.format(quoted(arg_opt), quoted(compiler)))
      if self.setCompilers.isGNU(compiler, self.log):
        compiler_version_re = re.compile(r'[gG][cC\+\-]+[0-9]* \(.+\) (\d+)\.(\d+)\.(\d+)')
        exec_names          = ['gcov']
      elif self.setCompilers.isClang(compiler, self.log):
        compiler_version_re = re.compile(r'clang version (\d+)\.(\d+)\.(\d+)')
        exec_names          = ['llvm-cov']
        if self.setCompilers.isDarwin(self.log):
          # macOS masquerades llvm-cov as just 'gcov', so we add this to the list in case
          # bare llvm-cov does not work
          exec_names.append('gcov')
      elif arg_opt == 'default-auto' and not required:
        # default-auto implies the user did not set it via command line!
        log_print('Could not auto-detect coverage tool for {}, not a gnuish compiler. Bailing since user did not explicitly set exec on the commandline'.format(quoted(compiler)))
        return
      else:
        # implies 'auto' explicitly set by user, or we were required to find
        # something. either way we should error
        raise RuntimeError('Could not auto-detect coverage tool for {}, please set coverage tool name explicitly'.format(quoted(compiler)))

      try:
        compiler_version_str = self.compilerFlags.version[lang]
      except KeyError:
        compiler_version_str = 'Unknown'

      log_print('Searching version string {} (for compiler {}) using pattern {}'.format(quoted(compiler_version_str), quoted(compiler), quoted(compiler_version_re.pattern)))
      compiler_version = compiler_version_re.search(compiler_version_str)
      if compiler_version is not None:
        log_print('Found major = {}, minor = {}, patch = {}'.format(compiler_version.group(1), compiler_version.group(2), compiler_version.group(3)))
        # form [llvm-cov-14, llvm-cov-14.0, llvm-cov, etc.]
        cov_exec_name = exec_names[0]
        exec_names    = [
          # llvm-cov-14
          '{}-{}'.format(cov_exec_name, compiler_version.group(1)),
           # llvm-cov-14.0
          '{}-{}.{}'.format(cov_exec_name, compiler_version.group(1), compiler_version.group(2))
        ] + exec_names
    else:
      log_print('User explicitly set coverage exec as {}'.format(quoted(arg_opt)))
      par_dir = os.path.dirname(arg_opt)
      if os.path.exists(par_dir):
        # arg_opt is path-like, we should only search the provided directory when we go
        # looking for the tool
        use_default_path = False
        search_path      = par_dir
      exec_names = [arg_opt]

    make_macro_name = 'PETSC_COVERAGE_EXEC'
    log_print('Checking for coverage tool(s):\n{}'.format('\n'.join('- '+t for t in exec_names)))
    found_exec = self.getExecutables(
      exec_names,
      path=search_path, getFullPath=True, useDefaultPath=use_default_path, resultName=make_macro_name
    )

    if found_exec is None:
      # didn't find the coverage tool
      if required:
        raise RuntimeError('Coverage tool(s) {} could not be found. Please provide explicit path to coverage tool'.format(exec_names))
      return

    found_exec_name = os.path.basename(found_exec)
    if 'llvm-cov' in found_exec_name and 'gcov' not in found_exec_name:
      # llvm-cov needs to be called as 'llvm-cov gcov' to work
      self.addMakeMacro(make_macro_name, found_exec + ' gcov')
    return

  def configureStrictPetscErrorCode(self):
    """
    Enables or disables strict PetscErrorCode checking.

    If --with-strict-petscerrorcode = 1:
    - defines PETSC_USE_STRICT_PETSCERRORCODE to 1

    Else:
    - deletes any prior PETSC_USE_STRICT_PETSCERRORCODE definitions (if they exist)
    """
    define_name = 'USE_STRICT_PETSCERRORCODE'
    if self.argDB['with-strict-petscerrorcode']:
      self.addDefine(define_name, 1)
    else:
      # in case it was somehow added previously
      self.delDefine(define_name)
    return

#-----------------------------------------------------------------------------------------------------
  def configureCygwinBrokenPipe(self):
    '''Cygwin version <= 1.7.18 had issues with pipes and long commands invoked from gnu-make
    http://cygwin.com/ml/cygwin/2013-05/msg00340.html '''
    if config.setCompilers.Configure.isCygwin(self.log):
      import platform
      import re
      r=re.compile("([0-9]+).([0-9]+).([0-9]+)")
      m=r.match(platform.release())
      major=int(m.group(1))
      minor=int(m.group(2))
      subminor=int(m.group(3))
      if ((major < 1) or (major == 1 and minor < 7) or (major == 1 and minor == 7 and subminor <= 18)):
        self.addMakeMacro('PETSC_CYGWIN_BROKEN_PIPE','1')
    return

#-----------------------------------------------------------------------------------------------------
  def configureDefaultArch(self):
    conffile = os.path.join('lib','petsc','conf', 'petscvariables')
    if self.framework.argDB['with-default-arch']:
      fd = open(conffile, 'w')
      fd.write('PETSC_ARCH='+self.arch.arch+'\n')
      fd.write('PETSC_DIR='+self.petscdir.dir+'\n')
      fd.write('include '+os.path.join('$(PETSC_DIR)','$(PETSC_ARCH)','lib','petsc','conf','petscvariables')+'\n')
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
    scriptName = os.path.join(self.arch.arch,'lib','petsc','conf', 'reconfigure-'+self.arch.arch+'.py')
    args = dict([(nargs.Arg.parseArgument(arg)[0], arg) for arg in self.framework.clArgs])
    if 'with-clean' in args:
      del args['with-clean']
    if 'force' in args:
      del args['force']
    if 'configModules' in args:
      if nargs.Arg.parseArgument(args['configModules'])[1] == 'PETSc.Configure':
        del args['configModules']
    if 'optionsModule' in args:
      if nargs.Arg.parseArgument(args['optionsModule'])[1] == 'config.compilerOptions':
        del args['optionsModule']
    if not 'PETSC_ARCH' in args:
      args['PETSC_ARCH'] = 'PETSC_ARCH='+str(self.arch.arch)
    f = open(scriptName, 'w')
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
      os.chmod(scriptName, 0o775)
    except OSError as e:
      self.framework.logPrint('Unable to make reconfigure script executable:\n'+str(e))
    self.framework.actions.addArgument('PETSc', 'File creation', 'Created '+scriptName+' for automatic reconfiguration')
    return

  def configureInstall(self):
    '''Setup the directories for installation'''
    if self.framework.argDB['prefix']:
      self.addMakeRule('print_mesg_after_build','',
       ['-@echo "========================================="',
        '-@echo "Now to install the libraries do:"',
        '-@echo "%s${MAKE_USER} PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} install"' % self.installdir.installSudo,
        '-@echo "========================================="'])
    else:
      self.addMakeRule('print_mesg_after_build','',
       ['-@echo "========================================="',
        '-@echo "Now to check if the libraries are working do:"',
        '-@echo "${MAKE_USER} PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} check"',
        '-@echo "========================================="'])
      return

  def postProcessPackages(self):
    postPackages=[]
    for i in self.framework.packages:
      if hasattr(i,'postProcess'): postPackages.append(i)
    if postPackages:
      # ctetgen needs petsc conf files. so attempt to create them early
      self.framework.dumpConfFiles()
      # tacky fix for dependency of Aluimia on Pflotran; requested via petsc-dev Matt provide a correct fix
      for i in postPackages:
        if i.name.upper() in ['PFLOTRAN']:
          i.postProcess()
          postPackages.remove(i)
      for i in postPackages: i.postProcess()
      for i in postPackages:
        if i.installedpetsc:
          self.installed = 1
          break
    return

  def configure(self):
    if 'package-prefix-hash' in self.argDB:
      # turn off prefix if it was only used to for installing external packages.
      self.framework.argDB['prefix'] = ''
      self.dir = os.path.abspath(os.path.join(self.petscdir.dir, self.arch.arch))
      self.installdir.dir = self.dir
      self.installdir.petscDir = self.petscdir.dir
      self.petscDir = self.petscdir.dir
      self.petscArch = self.arch.arch
      self.addMakeMacro('PREFIXDIR',self.dir)
      self.confDir = os.path.abspath(os.path.join(self.petscdir.dir, self.arch.arch))

    if not os.path.samefile(self.petscdir.dir, os.getcwd()):
      raise RuntimeError('Wrong PETSC_DIR option specified: '+str(self.petscdir.dir) + '\n  Configure invoked in: '+os.path.realpath(os.getcwd()))
    if self.framework.argDB['prefix'] and os.path.isdir(self.framework.argDB['prefix']) and os.path.samefile(self.framework.argDB['prefix'],self.petscdir.dir):
      raise RuntimeError('Incorrect option --prefix='+self.framework.argDB['prefix']+' specified. It cannot be same as PETSC_DIR!')
    if self.framework.argDB['prefix'] and self.framework.argDB['prefix'].find(' ') > -1:
      raise RuntimeError('Your --prefix '+self.framework.argDB['prefix']+' has spaces in it; this is not allowed.\n Use a --prefix that does not have spaces in it')
    if self.framework.argDB['prefix'] and os.path.isdir(self.framework.argDB['prefix']) and os.path.samefile(self.framework.argDB['prefix'],os.path.join(self.petscdir.dir,self.arch.arch)):
      raise RuntimeError('Incorrect option --prefix='+self.framework.argDB['prefix']+' specified. It cannot be same as PETSC_DIR/PETSC_ARCH!')
    self.framework.header          = os.path.join(self.arch.arch,'include','petscconf.h')
    self.framework.cHeader         = os.path.join(self.arch.arch,'include','petscfix.h')
    self.framework.poisonheader    = os.path.join(self.arch.arch,'include','petscconf_poison.h')
    self.framework.pkgheader       = os.path.join(self.arch.arch,'include','petscpkg_version.h')
    self.framework.makeMacroHeader = os.path.join(self.arch.arch,'lib','petsc','conf','petscvariables')
    self.framework.makeRuleHeader  = os.path.join(self.arch.arch,'lib','petsc','conf','petscrules')
    if self.libraries.math is None:
      raise RuntimeError('PETSc requires a functional math library. Please send configure.log to petsc-maint@mcs.anl.gov.')
    if self.languages.clanguage == 'Cxx' and not hasattr(self.compilers, 'CXX'):
      raise RuntimeError('Cannot set C language to C++ without a functional C++ compiler.')
    self.executeTest(self.configureRTLDDefault)
    self.executeTest(self.configurePrefetch)
    self.executeTest(self.configureUnused)
    self.executeTest(self.configureDeprecated)
    self.executeTest(self.configureIsatty)
    self.executeTest(self.configureExpect)
    self.executeTest(self.configureAlign)
    self.executeTest(self.configureFunctionName)
    self.executeTest(self.configureIntptrt)
    self.executeTest(self.configureSolaris)
    self.executeTest(self.configureLinux)
    self.executeTest(self.configureDarwin)
    self.executeTest(self.configureWin32)
    self.executeTest(self.configureCygwinBrokenPipe)
    self.executeTest(self.configureDefaultArch)
    self.executeTest(self.configureScript)
    self.executeTest(self.configureInstall)
    self.executeTest(self.configureAtoll)
    self.executeTest(self.configureCoverage)
    self.executeTest(self.configureCoverageExecutable)
    self.executeTest(self.configureStrictPetscErrorCode)

    self.Dump()
    self.dumpConfigInfo()
    self.dumpMachineInfo()
    self.delGenFiles()
    # need to save the current state of BuildSystem so that postProcess() packages can read it in and perhaps run make install
    self.framework.storeSubstitutions(self.framework.argDB)
    self.framework.argDB['configureCache'] = pickle.dumps(self.framework)
    self.framework.argDB.save(force = True)
    self.DumpPkgconfig('PETSc.pc')
    self.DumpPkgconfig('petsc.pc')
    self.DumpModule()
    self.postProcessPackages()
    self.framework.log.write('================================================================================\n')
    self.logClear()
    return
