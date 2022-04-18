import config.package

class Configure(config.package.Package):
  def __init__(self,framework):
    config.package.Package.__init__(self,framework)
    self.gitcommit              = '547451c0742d15bcafee673cfa2f749efc0ac6fa' # main apr-18-2022
    self.download               = ['git://https://github.com/hpddm/hpddm','https://github.com/hpddm/hpddm/archive/'+self.gitcommit+'.tar.gz']
    self.minversion             = '2.2.1'
    self.versionname            = 'HPDDM_VERSION'
    self.versioninclude         = 'HPDDM_define.hpp'
    self.buildLanguages         = ['Cxx']
    self.functions              = []
    self.includes               = ['HPDDM.hpp']
    self.skippackagewithoptions = 1
    self.linkedbypetsc          = 0
    self.builtafterpetsc        = 1
    self.precisions             = ['single','double']
    self.hastestsdatafiles      = 1
    return

  def setupDependencies(self,framework):
    config.package.Package.setupDependencies(self,framework)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries',self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.cxxlibs         = framework.require('config.packages.cxxlibs',self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.flibs           = framework.require('config.packages.flibs',self)
    self.deps            = [self.blasLapack,self.cxxlibs,self.mathlib,self.flibs] # KSPHPDDM
    self.mpi             = framework.require('config.packages.MPI',self)
    self.slepc           = framework.require('config.packages.slepc',self)
    self.odeps           = [self.mpi,self.slepc] # KSPHPDDM + PCHPDDM
    return

  def Install(self):
    import os
    if self.blasLapack.mkl and not self.blasLapack.mkl_spblas_h:
      raise RuntimeError('Cannot use HPDDM with the MKL as \'mkl_spblas.h\' was not found, check for missing --with-blaslapack-include=/opt/intel/mkl/include (or similar)')
    buildDir = os.path.join(self.packageDir,'petsc-build')
    self.pushLanguage('Cxx')
    cxx = self.getCompiler()
    cxxflags = self.getCompilerFlags()
    self.popLanguage()
    if self.framework.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      PETSC_DIR  = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
      PETSC_ARCH = ''
      prefix     = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
      PETSC_DIR  = self.petscdir.dir
      PETSC_ARCH = self.arch
      prefix     = os.path.join(self.petscdir.dir,self.arch)
    incDir = os.path.join(prefix,'include')
    libDir = os.path.join(prefix,'lib')
    self.addMakeMacro('HPDDM','yes')
    self.include = [incDir]
    if not hasattr(self.framework,'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)
    try:
      self.logPrintBox('Copying HPDDM; this may take several seconds')
      output,err,ret = config.package.Package.executeShellCommand(['cp','-rf',os.path.join(self.packageDir,'include'),prefix],timeout=100,log=self.log) # cannot use shutil.copytree since target directory likely exists
      self.log.write(output+err)
    except RuntimeError as e:
      raise RuntimeError('Error copying HPDDM: '+str(e))
    # SLEPc dependency
    if self.slepc.found:
      if self.checkSharedLibrariesEnabled():
        slepcbuilddep = ''
        ldflags = ' '.join(self.setCompilers.sharedLibraryFlags)
        cxxflags += ' '+self.headers.toStringNoDupes(self.dinclude+[os.path.join(PETSC_DIR,'include'),os.path.join(PETSC_DIR,PETSC_ARCH,'include')])
        ldflags += ' '+self.libraries.toStringNoDupes(self.dlib)
        slepcbuilddep = 'slepc-install slepc-build'
        oldFlags = self.compilers.CXXPPFLAGS
        self.compilers.CXXPPFLAGS += ' -I'+incDir
        self.checkVersion()
        self.compilers.CXXPPFLAGS = oldFlags
        # check for Windows-specific define
        if self.sharedLibraries.getMakeMacro('PETSC_DLL_EXPORTS'):
          cxxflags += ' -Dpetsc_EXPORTS'
          # need to explicitly link to PETSc and BLAS on Windows
          ldflags += ' '+self.libraries.toStringNoDupes([os.path.join(libDir,'libpetsc.'+self.setCompilers.sharedLibraryExt),self.libraries.toStringNoDupes(self.blasLapack.lib)])
        self.addMakeRule('hpddmbuild',slepcbuilddep,\
                           ['@echo "*** Building and installing HPDDM ***"',\
                            '@${RM} -f ${PETSC_ARCH}/lib/petsc/conf/hpddm.errorflg',\
                            '@'+cxx+' '+cxxflags+' '+os.path.join(self.packageDir,'interface','hpddm_petsc.cpp')+' '+ldflags+' -o '+os.path.join(libDir,'libhpddm_petsc.'+self.setCompilers.sharedLibraryExt)+' > ${PETSC_ARCH}/lib/petsc/conf/hpddm.log 2>&1 || \\\n\
                 (echo "**************************ERROR*************************************" && \\\n\
                 echo "Error building HPDDM. Check ${PETSC_ARCH}/lib/petsc/conf/hpddm.log" && \\\n\
                 echo "********************************************************************" && \\\n\
                 touch '+os.path.join('${PETSC_ARCH}','lib','petsc','conf','hpddm.errorflg')+' && \\\n\
                 exit 1)'])
        if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
          self.addMakeRule('hpddm-build','')
          self.addMakeRule('hpddm-install','hpddmbuild')
          return self.installDir
        else:
          self.addMakeRule('hpddm-build','hpddmbuild')
          self.addMakeRule('hpddm-install','')
          return self.installDir
      else:
        self.logPrintBox('***** WARNING: Skipping PCHPDDM installation,\n\
remove --with-shared-libraries=0 *****')
    else:
      self.logPrintBox('***** WARNING: Compiling HPDDM without SLEPc,\n\
PCHPDDM won\'t be available, unless reconfiguring with --download-slepc *****')
    self.addMakeRule('hpddm-build','')
    self.addMakeRule('hpddm-install','')
    return self.installDir

  def alternateConfigureLibrary(self):
    self.addMakeRule('hpddm-build','')
    self.addMakeRule('hpddm-install','')
