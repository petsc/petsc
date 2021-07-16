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
    self.installed = 0 # 1 indicates that Configure itself has already compiled and installed PETSc
    return

  def __str2__(self):
    desc = ['  Using GNU make: ' + self.make.make]
    if not self.installed:
      desc.append('xxx=========================================================================xxx')
      desc.append(' Configure stage complete. Now build PETSc libraries with:')
      desc.append('   %s PETSC_DIR=%s PETSC_ARCH=%s all' % (self.make.make_user, self.petscdir.dir, self.arch.arch))
      desc.append('xxx=========================================================================xxx')
    else:
      desc.append('xxx=========================================================================xxx')
      desc.append(' Installation complete. You do not need to run make to compile or install the software')
      desc.append('xxx=========================================================================xxx')
    return '\n'.join(desc)+'\n'

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc',  '-prefix=<dir>',                              nargs.Arg(None, '', 'Specifiy location to install PETSc (eg. /usr/local)'))
    help.addArgument('PETSc',  '-with-prefetch=<bool>',                      nargs.ArgBool(None, 1,'Enable checking for prefetch instructions'))
    help.addArgument('Windows','-with-windows-graphics=<bool>',              nargs.ArgBool(None, 1,'Enable check for Windows Graphics'))
    help.addArgument('PETSc', '-with-default-arch=<bool>',                   nargs.ArgBool(None, 1, 'Allow using the last configured arch without setting PETSC_ARCH'))
    help.addArgument('PETSc','-with-single-library=<bool>',                  nargs.ArgBool(None, 1,'Put all PETSc code into the single -lpetsc library'))
    help.addArgument('PETSc','-with-fortran-bindings=<bool>',                nargs.ArgBool(None, 1,'Build PETSc fortran bindings in the library and corresponding module files'))
    help.addArgument('PETSc', '-with-ios=<bool>',                            nargs.ArgBool(None, 0, 'Build an iPhone/iPad version of PETSc library'))
    help.addArgument('PETSc', '-with-xsdk-defaults',                         nargs.ArgBool(None, 0, 'Set the following as defaults for the xSDK standard: --enable-debug=1, --enable-shared=1, --with-precision=double, --with-index-size=32, locate blas/lapack automatically'))
    help.addArgument('PETSc', '-with-display=<x11display>',                  nargs.Arg(None, '', 'Specifiy DISPLAY env variable for use with matlab test)'))
    help.addArgument('PETSc', '-with-package-scripts=<pyscripts>',           nargs.ArgFileList(None,None,'Specify configure package scripts for user provided packages'))
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

    self.programs.headerPrefix   = self.headerPrefix
    self.compilers.headerPrefix  = self.headerPrefix
    self.fortran.headerPrefix    = self.headerPrefix
    self.types.headerPrefix      = self.headerPrefix
    self.headers.headerPrefix    = self.headerPrefix
    self.functions.headerPrefix  = self.headerPrefix
    self.libraries.headerPrefix  = self.headerPrefix

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
                                            'unistd','sys/sysinfo','machine/endian','sys/param','sys/procfs','sys/resource',
                                            'sys/systeminfo','sys/times','sys/utsname',
                                            'sys/socket','sys/wait','netinet/in','netdb','direct','time','Ws2tcpip','sys/types',
                                            'WindowsX','float','ieeefp','stdint','pthread','inttypes','immintrin','zmmintrin'])
    functions = ['access','_access','clock','drand48','getcwd','_getcwd','getdomainname','gethostname',
                 'getwd','memalign','popen','PXFGETARG','rand','getpagesize',
                 'readlink','realpath','usleep','sleep','_sleep',
                 'uname','snprintf','_snprintf','lseek','_lseek','time','fork','stricmp',
                 'strcasecmp','bzero','dlopen','dlsym','dlclose','dlerror',
                 '_set_output_format','_mkdir','socket','gethostbyname','_pipe','fpresetsticky','fpsetsticky']
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
    puts stderr "     see https://www.mcs.anl.gov/petsc/ for more information      "
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
        self.addDefine('MPICC_SHOW','"'+output.strip().replace('\n','\\\\n')+'"')
      except:
        self.addDefine('MPICC_SHOW','"Unavailable"')
    else:
      self.addDefine('MPICC_SHOW','"Unavailable"')
    self.setCompilers.popLanguage()
#-----------------------------------------------------------------------------------------------------

    # Sometimes we need C compiler, even if built with C++
    self.setCompilers.pushLanguage('C')
    self.addMakeMacro('CC_FLAGS',self.setCompilers.getCompilerFlags())
    self.setCompilers.popLanguage()

    # And sometimes we need a C++ compiler even when PETSc is built with C
    if hasattr(self.compilers, 'CXX'):
      self.setCompilers.pushLanguage('Cxx')
      self.addDefine('HAVE_CXX','1')
      self.addMakeMacro('CXXPP_FLAGS',self.setCompilers.CXXPPFLAGS)
      self.addMakeMacro('CXX_FLAGS',self.setCompilers.getCompilerFlags())
      cxx_linker = self.setCompilers.getLinker()
      self.addMakeMacro('CXX_LINKER',cxx_linker)
      self.addMakeMacro('CXX_LINKER_FLAGS',self.setCompilers.getLinkerFlags())
      self.setCompilers.popLanguage()

    # C preprocessor values
    self.addMakeMacro('CPP_FLAGS',self.setCompilers.CPPFLAGS)

    # compiler values
    self.setCompilers.pushLanguage(self.languages.clanguage)
    self.addMakeMacro('PCC',self.setCompilers.getCompiler())
    self.addMakeMacro('PCC_FLAGS',self.setCompilers.getCompilerFlags())
    self.addMakeMacro('PCPP_FLAGS',getattr(self.setCompilers,self.languages.clanguage.upper()+'PPFLAGS'))
    self.addMakeMacro('PFLAGS','${'+self.languages.clanguage.upper()+'FLAGS}')
    self.addMakeMacro('PPPFLAGS','${'+self.languages.clanguage.upper()+'PPFLAGS}')
    # ugly work-around for python3 distutils parse_makefile() issue with the above 2 lines
    self.addMakeMacro('PY_'+self.languages.clanguage.upper()+'FLAGS','')
    self.addMakeMacro('PY_'+self.languages.clanguage.upper()+'PPFLAGS','')
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

    if hasattr(self.compilers, 'HIPC'):
      self.setCompilers.pushLanguage('HIP')
      self.addMakeMacro('HIPC_FLAGS',self.setCompilers.getCompilerFlags())
      self.addMakeMacro('HIPPP_FLAGS',self.setCompilers.HIPPPFLAGS)
      self.setCompilers.popLanguage()

    if hasattr(self.compilers, 'SYCLCXX'):
      self.setCompilers.pushLanguage('SYCL')
      self.addMakeMacro('SYCLCXX_FLAGS',self.setCompilers.getCompilerFlags())
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
    if self.checkLink('#define _POSIX_C_SOURCE 200112L\n#include <stdlib.h>','long v = atoll("25")') or self.checkLink ('#include <stdlib.h>','long v = atoll("25")'):
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

  def configureIsatty(self):
    '''Check if the Unix C function isatty() works correctly
       Actually just assumes it does not work correctly on batch systems'''
    if not self.framework.argDB['with-batch']:
      self.addDefine('USE_ISATTY',1)

  def configureDeprecated(self):
    '''Check if __attribute((deprecated)) is supported'''
    self.pushLanguage(self.languages.clanguage)
    ## Recent versions of gcc and clang support __attribute((deprecated("string argument"))), which is very useful, but
    ## Intel has conspired to make a supremely environment-sensitive compiler.  The Intel compiler looks at the gcc
    ## executable in the environment to determine the language compatibility that it should attempt to emulate.  Some
    ## important Cray installations have built PETSc using the Intel compiler, but with a newer gcc module loaded (e.g.,
    ## 4.7).  Thus at PETSc configure time, the Intel compiler decides to support the string argument, but the gcc
    ## found in the default user environment is older and does not support the argument.  If GCC and Intel were cool
    ## like Clang and supported __has_attribute, we could avoid configure tests entirely, but they don't.  And that is
    ## why we can't have nice things.
    #
    # if self.checkCompile("""__attribute((deprecated("Why you shouldn't use myfunc"))) static int myfunc(void) { return 1;}""", ''):
    #   self.addDefine('DEPRECATED_FUNCTION(why)', '__attribute((deprecated(why)))')
    #   self.addDefine('DEPRECATED_TYPEDEF(why)', '__attribute((deprecated(why)))')
    if self.checkCompile("""__attribute((deprecated)) static int myfunc(void) { return 1;}""", ''):
      self.addDefine('DEPRECATED_FUNCTION(why)', '__attribute((deprecated))')
      self.addDefine('DEPRECATED_TYPEDEF(why)', '__attribute((deprecated))')
    else:
      self.addDefine('DEPRECATED_FUNCTION(why)', ' ')
      self.addDefine('DEPRECATED_TYPEDEF(why)', ' ')
    if self.checkCompile("""enum E {oldval __attribute((deprecated)), newval };""", ''):
      self.addDefine('DEPRECATED_ENUM(why)', '__attribute((deprecated))')
    else:
      self.addDefine('DEPRECATED_ENUM(why)', ' ')
    # I was unable to make a CPP macro that takes the old and new values as separate arguments and builds the message needed by _Pragma
    # hence the deprecation message is handled as it is
    if self.checkCompile('#define TEST _Pragma("GCC warning \"Testing _Pragma\"") value'):
      self.addDefine('DEPRECATED_MACRO(why)', '_Pragma(why)')
    else:
      self.addDefine('DEPRECATED_MACRO(why)', ' ')
    self.popLanguage()

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

  def configureRTLDDefault(self):
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

  def configureGCOV(self):
    if self.framework.argDB['with-gcov']:
      self.addDefine('USE_GCOV','1')
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
    self.executeTest(self.configureWin32)
    self.executeTest(self.configureCygwinBrokenPipe)
    self.executeTest(self.configureDefaultArch)
    self.executeTest(self.configureScript)
    self.executeTest(self.configureInstall)
    self.executeTest(self.configureGCOV)
    self.executeTest(self.configureAtoll)

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
