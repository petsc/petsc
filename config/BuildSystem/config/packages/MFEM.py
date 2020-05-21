import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    #disable version checking
    #self.minversion             = '4'
    #self.version                = '4.0.0'
    #self.versionname            = 'MFEM_VERSION_STRING'
    #self.versioninclude         = 'mfem/config.hpp'
    self.gitcommit              = 'v4.0' # tags do not include subminor
    self.download               = ['git://https://github.com/mfem/mfem.git']
    self.linkedbypetsc          = 0
    self.downloadonWindows      = 1
    self.cxx                    = 1
    self.requirescxx11          = 1
    self.skippackagewithoptions = 1
    self.builtafterpetsc        = 1
    self.noMPIUni               = 1
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('MFEM', '-download-mfem-ghv-cxx=<prog>', nargs.Arg(None, None, 'CXX Front-end compiler to compile get_hypre_version'))
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.hypre = framework.require('config.packages.hypre',self)
    self.mpi   = framework.require('config.packages.MPI',self)
    self.metis = framework.require('config.packages.metis',self)
    self.deps  = [self.mpi,self.hypre,self.metis]
    return

  def Install(self):
    import os

    buildDir = os.path.join(self.packageDir,'petsc-build')
    configDir = os.path.join(buildDir,'config')
    if not os.path.exists(configDir):
      os.makedirs(configDir)

    if self.framework.argDB['prefix']:
      PETSC_DIR  = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
      PETSC_ARCH = ''
      prefix     = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
      PETSC_DIR  = self.petscdir.dir
      PETSC_ARCH = self.arch
      prefix     = os.path.join(self.petscdir.dir,self.arch)

    PETSC_OPT = self.headers.toStringNoDupes([os.path.join(PETSC_DIR,'include'),os.path.join(PETSC_DIR,PETSC_ARCH,'include')])

    self.setCompilers.pushLanguage('Cxx')
    cxx = self.setCompilers.getCompiler()
    cxxflags = self.setCompilers.getCompilerFlags()
    cxxflags = cxxflags.replace('-fvisibility=hidden','') # MFEM is currently broken with -fvisibility=hidden
    # MFEM uses the macro MFEM_BUILD_DIR that builds a path by combining the directory plus other stuff but if the
    # directory name contains  "-linux'" this is converted by CPP to the value 1 since that is defined in Linux header files
    # unless the -std=C++11 or -std=C++14 flag is used; we want to support MFEM without this flag
    cxxflags += ' -Dlinux=linux'
    self.setCompilers.popLanguage()
    if 'download-mfem-ghv-cxx' in self.argDB and self.argDB['download-mfem-ghv-cxx']:
      ghv = self.argDB['download-mfem-ghv-cxx']
    else:
      ghv = cxx

    # On CRAY with shared libraries, libmfem.so is linked as
    # $ cc -shared -o libmfem.so ...a bunch of .o files.... ...libraries.... -dynamic
    # The -dynamic at the end makes cc think it is creating an executable
    ldflags = self.setCompilers.LDFLAGS.replace('-dynamic','')

    with open(os.path.join(configDir,'user.mk'),'w') as g:
      g.write('PREFIX = '+prefix+'\n')
      g.write('MPICXX = '+cxx+'\n')
      g.write('export GHV_CXX = '+ghv+'\n')
      g.write('CXXFLAGS = '+cxxflags+'\n')
      if self.argDB['with-shared-libraries']:
        g.write('SHARED = YES\n')
        g.write('STATIC = NO\n')
      else:
        g.write('SHARED = NO\n')
        g.write('STATIC = YES\n')
      g.write('AR = '+self.setCompilers.AR+'\n')
      g.write('ARFLAGS = '+self.setCompilers.AR_FLAGS+'\n')
      g.write('LDFLAGS = '+ldflags+'\n')
      g.write('MFEM_USE_MPI = YES\n')
      g.write('MFEM_MPIEXEC = '+self.mpi.getMakeMacro('MPIEXEC')+'\n')
      g.write('MFEM_USE_METIS_5 = YES\n')
      g.write('MFEM_USE_METIS = YES\n')
      g.write('MFEM_USE_PETSC = YES\n')
      g.write('HYPRE_OPT = '+self.headers.toString(self.hypre.include)+'\n')
      g.write('HYPRE_LIB = '+self.libraries.toString(self.hypre.lib)+'\n')
      g.write('METIS_OPT = '+self.headers.toString(self.metis.include)+'\n')
      g.write('METIS_LIB = '+self.libraries.toString(self.metis.lib)+'\n')
      g.write('PETSC_VARS ='+prefix+'/lib/petsc/conf/petscvariables\n')
      g.write('PETSC_OPT = '+PETSC_OPT+'\n')
      # Adding all externals should not be needed when PETSc is a shared library, but it is no harm.
      # When the HYPRE library is built statically, we need to resolve blas symbols
      g.write('PETSC_LIB = $(shell sed -n "s/PETSC_WITH_EXTERNAL_LIB = *//p" $(PETSC_VARS))\n')
      g.close()

    #  if installing as Superuser than want to return to regular user for clean and build
    if self.installSudo:
       newuser = self.installSudo+' -u $${SUDO_USER} '
    else:
       newuser = ''

    self.addDefine('HAVE_MFEM',1)
    self.addMakeMacro('MFEM','yes')
    self.addMakeRule('mfembuild','', \
                       ['@echo "*** Building mfem ***"',\
                          '@${RM} -f ${PETSC_ARCH}/lib/petsc/conf/mfem.errorflg',\
                          '@(cd '+buildDir+' && \\\n\
           ${OMAKE} -f '+self.packageDir+'/makefile config && \\\n\
           ${OMAKE} clean && \\\n\
           '+self.make.make_jnp+') > ${PETSC_ARCH}/lib/petsc/conf/mfem.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building mfem. Check ${PETSC_ARCH}/lib/petsc/conf/mfem.log" && \\\n\
             echo "********************************************************************" && \\\n\
             touch ${PETSC_ARCH}/lib/petsc/conf/mfem.errorflg && \\\n\
             exit 1)'])
    self.addMakeRule('mfeminstall','', \
                       ['@echo "*** Installing mfem ***"',\
                          '@(cd '+buildDir+' && \\\n\
           '+newuser+'${OMAKE} install) >> ${PETSC_ARCH}/lib/petsc/conf/mfem.log 2>&1 || \\\n\
             (echo "**************************ERROR*************************************" && \\\n\
             echo "Error building mfem. Check ${PETSC_ARCH}/lib/petsc/conf/mfem.log" && \\\n\
             echo "********************************************************************" && \\\n\
             exit 1)'])

    if self.argDB['prefix']:
      self.addMakeRule('mfem-build','')
      self.addMakeRule('mfem-install','mfembuild mfeminstall')
    else:
      self.addMakeRule('mfem-build','mfembuild mfeminstall')
      self.addMakeRule('mfem-install','')

    exampleDir = os.path.join(self.packageDir,'examples')
    self.logClearRemoveDirectory()
    self.logPrintBox('MFEM examples are available at '+exampleDir)
    self.logResetRemoveDirectory()

    return self.installDir

  def alternateConfigureLibrary(self):
    self.addMakeRule('mfem-build','')
    self.addMakeRule('mfem-install','')
