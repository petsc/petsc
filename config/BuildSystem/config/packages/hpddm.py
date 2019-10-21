import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = '7d80961' #master sep-29-2019
    self.download               = ['git://https://github.com/hpddm/hpddm','https://github.com/hpddm/hpddm/archive/'+self.gitcommit+'.tar.gz']
    self.minversion             = '2.0.0'
    self.versionname            = 'HPDDM_VERSION'
    self.versioninclude         = 'HPDDM_define.hpp'
    self.requirescxx11          = 1
    self.noMPIUni               = 1
    self.cxx                    = 1
    self.functions              = []
    self.includes               = ['HPDDM.hpp']
    self.skippackagewithoptions = 1
    self.useddirectly           = 1
    self.linkedbypetsc          = 0
    self.builtafterpetsc        = 1
    self.precisions             = ['single','double']
    self.hastestsdatafiles      = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries',self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.cxxlibs         = framework.require('config.packages.cxxlibs',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.slepc           = framework.require('config.packages.slepc',self)
    self.odeps           = [self.slepc]
    self.deps            = [self.mpi,self.blasLapack,self.cxxlibs,self.mathlib]
    return

  def Install(self):
    import os
    if not self.checkSharedLibrariesEnabled():
      raise RuntimeError('Shared libraries enabled needed to build HPDDM')
    if self.framework.argDB['with-64-bit-blas-indices']:
      raise RuntimeError('32-bit BLAS needed to build HPDDM')
    buildDir   = os.path.join(self.packageDir,'petsc-build')
    self.setCompilers.pushLanguage('Cxx')
    cxx = self.setCompilers.getCompiler()
    cxxflags = self.setCompilers.getCompilerFlags()
    self.setCompilers.popLanguage()
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
    PETSC_OPT = self.headers.toStringNoDupes([os.path.join(PETSC_DIR,'include'),os.path.join(PETSC_DIR,PETSC_ARCH,'include')])
    # SLEPc dependency
    ldflags = ' '.join(self.setCompilers.sharedLibraryFlags)
    slepcbuilddep = ''
    if self.slepc.found:
      # how can we get the slepc lib? Eventually, we may want to use the variables from the framework
      #cxxflags += self.headers.toStringNoDupes(self.slepc.dinclude)
      #ldflags += self.libraries.toString(self.slepc.dlib)
      dinclude = [incDir]
      dlib = [os.path.join(libDir,'libslepc.'+self.setCompilers.sharedLibraryExt)]
      cxxflags += ' '+self.headers.toStringNoDupes(dinclude)
      ldflags += ' '+self.libraries.toString(dlib)
      slepcbuilddep = 'slepc-install slepc-build'
    if self.installSudo:
       newuser = self.installSudo+' -u $${SUDO_USER} '
    else:
       newuser = ''
    self.addMakeMacro('HPDDM','yes')
    self.include = [incDir]
    if not hasattr(self.framework, 'packages'):
      self.framework.packages = []
    self.framework.packages.append(self)
    cpstr = newuser+' mkdir -p '+incDir+' && '+newuser+' cp '+os.path.join(self.packageDir,'include','*')+' '+incDir
    self.logPrintBox('Copying HPDDM; this may take several seconds')
    output,err,ret  = config.package.Package.executeShellCommand(cpstr,timeout=100,log=self.log)
    self.log.write(output+err)
    oldFlags = self.compilers.CXXPPFLAGS
    self.compilers.CXXPPFLAGS += ' -I'+incDir
    self.checkVersion()
    self.compilers.CXXPPFLAGS = oldFlags
    self.addMakeRule('hpddmcopy','',\
                       ['@echo "*** Copying HPDDM ***"',\
                        '@${RM} -f ${PETSC_ARCH}/lib/petsc/conf/hpddm.errorflg',\
                        '@'+cpstr+' > ${PETSC_ARCH}/lib/petsc/conf/hpddm.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error copying HPDDM. Check ${PETSC_ARCH}/lib/petsc/conf/hpddm.log" && \\\n\
             echo "********************************************************************" && \\\n\
             touch '+os.path.join('${PETSC_ARCH}','lib','petsc','conf','hpddm.errorflg')+' && \\\n\
             exit 1)'])
    self.addMakeRule('hpddmbuild',slepcbuilddep,\
                       ['@echo "*** Building and installing HPDDM ***"',\
                        '@${RM} -f ${PETSC_ARCH}/lib/petsc/conf/hpddm.errorflg',\
                        '@'+newuser+cxx+' '+cxxflags+' '+self.headers.toStringNoDupes(self.dinclude)+' '+PETSC_OPT+' -I'+self.packageDir+'/include '+self.packageDir+'/interface/hpddm_petsc.cpp '+ldflags+' -o '+libDir+os.path.join('/libhpddm_petsc.'+self.setCompilers.sharedLibraryExt)+' > ${PETSC_ARCH}/lib/petsc/conf/hpddm.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building HPDDM. Check ${PETSC_ARCH}/lib/petsc/conf/hpddm.log" && \\\n\
             echo "********************************************************************" && \\\n\
             touch '+os.path.join('${PETSC_ARCH}','lib','petsc','conf','hpddm.errorflg')+' && \\\n\
             exit 1)'])
    if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      self.addMakeRule('hpddm-build','')
      self.addMakeRule('hpddm-install','hpddmbuild')
    else:
      self.addMakeRule('hpddm-build','hpddmbuild')
      self.addMakeRule('hpddm-install','')
    return self.installDir

  def alternateConfigureLibrary(self):
    self.addMakeRule('hpddm-build','')
    self.addMakeRule('hpddm-install','')
