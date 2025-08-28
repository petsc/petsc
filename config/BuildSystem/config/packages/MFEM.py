import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    #disable version checking
    #self.minversion             = '4.6'
    #self.version                = '4.6'
    #self.versionname            = 'MFEM_VERSION_STRING'
    #self.versioninclude         = 'mfem/config.hpp'
    self.gitcommit              = '0795d119cfb37a8cbae7ba8c1b189cc88048248b' # stefanozampini/for-petsc3.24-release sep-27-2025 (includes hypre-3.0 https://github.com/mfem/mfem/pull/4975 with https://github.com/mfem/mfem/pull/5036)
    self.download               = ['git://https://github.com/mfem/mfem.git','https://github.com/mfem/mfem/archive/'+self.gitcommit+'.tar.gz']
    self.linkedbypetsc          = 0
    self.downloadonWindows      = 1
    self.buildLanguages         = ['Cxx']
    self.minCxxVersion          = 'c++17'
    self.skippackagewithoptions = 1
    self.builtafterpetsc        = 1
    self.noMPIUni               = 1
    self.precisions             = ['single', 'double']
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('MFEM', '-download-mfem-ghv-cxx=<prog>', nargs.Arg(None, None, 'CXX Front-end compiler to compile get_hypre_version'))
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.hypre        = framework.require('config.packages.hypre',self)
    self.mpi          = framework.require('config.packages.MPI',self)
    self.metis        = framework.require('config.packages.METIS',self)
    self.slepc        = framework.require('config.packages.SLEPc',self)
    self.ceed         = framework.require('config.packages.libCEED',self)
    self.cuda         = framework.require('config.packages.CUDA',self)
    self.hip          = framework.require('config.packages.HIP',self)
    self.openmp       = framework.require('config.packages.OpenMP',self)
    self.superlu_dist = framework.require('config.packages.SuperLU_DIST',self)
    self.netcdf       = framework.require('config.packages.netCDF',self)
    self.scalar = framework.require('PETSc.options.scalarTypes',self)
    self.deps   = [self.mpi,self.hypre,self.metis]
    self.odeps  = [self.slepc,self.ceed,self.cuda,self.openmp,self.superlu_dist,self.netcdf]
    return

  def writeConfig(self, g, lib_name, lib_data):
    lib_name_upper = lib_name.upper()
    if lib_data.found:
      g.write('MFEM_USE_{0} = YES\n'.format(lib_name_upper))
      g.write('{0}_DIR = {1}\n'.format(lib_name_upper, lib_data.directory))
      g.write('{0}_OPT = {1}\n'.format(lib_name_upper, self.headers.toString(lib_data.include)))
      g.write('{0}_LIB = {1}\n'.format(lib_name_upper, self.libraries.toString(lib_data.lib)))
      if self.cuda.found:
        g.write('{0}_LIB := $(subst -Wl,-Xlinker=,$({0}_LIB))\n'.format(lib_name_upper))

  def Install(self):
#    return self.installDir
#
#  TODO: In order to use postProcess, we need to fix package.py and add these lines
#  in configureLibrary if builtafterpetsc is true. However, these caused duplicated entries
#  in the petscconf.h macros. Not sure if PETSC_HAVE_XXX will conflict when building XXX after petsc
#+        if not hasattr(self.framework, 'packages'):
#+          self.framework.packages = []
#+        self.framework.packages.append(self)

#  def postProcess(self):
    import os
    import re

    buildDir = os.path.join(self.packageDir,'petsc-build')
    configDir = os.path.join(buildDir,'config')
    if not os.path.exists(configDir):
      os.makedirs(configDir)

    if self.framework.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      PETSC_DIR  = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
      PETSC_ARCH = ''
      prefix     = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
      PETSC_DIR  = self.petscdir.dir
      PETSC_ARCH = self.arch
      prefix     = os.path.join(self.petscdir.dir,self.arch)

    PETSC_OPT = self.headers.toStringNoDupes([os.path.join(PETSC_DIR,'include'),os.path.join(PETSC_DIR,PETSC_ARCH,'include')])

    self.pushLanguage('Cxx')
    cxx = self.getCompiler()
    cxxflags = self.updatePackageCxxFlags(self.getCompilerFlags())
    self.popLanguage()
    if 'download-mfem-ghv-cxx' in self.argDB and self.argDB['download-mfem-ghv-cxx']:
      ghv = self.argDB['download-mfem-ghv-cxx']
    else:
      ghv = cxx

    # On CRAY with shared libraries, libmfem.so is linked as
    # $ cc -shared -o libmfem.so ...a bunch of .o files.... ...libraries.... -dynamic
    # The -dynamic at the end makes cc think it is creating an executable
    ldflags = self.setCompilers.LDFLAGS.replace('-dynamic','')

    strip_rpath=''
    if self.cuda.found:
      strip_rpath=' | sed "s/-Wl,-rpath,/-Xlinker=-rpath,/g"'
      if self.openmp.found:
        ldflags = ldflags.replace(self.openmp.ompflag,'')

    makedepend = ''
    with open(os.path.join(configDir,'user.mk'),'w') as g:
      g.write('PREFIX = '+prefix+'\n')
      g.write('MPICXX = '+cxx+'\n')
      g.write('GHV_CXX = '+ghv+'\n')
      if not self.hip.found: #MFEM uses hipcc as compiler for everything
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
      if self.cuda.found:
        g.write('LDFLAGS := $(addprefix -Xlinker ,$(LDFLAGS))\n')
      g.write('MFEM_USE_MPI = YES\n')
      g.write('MFEM_MPIEXEC = '+self.mpi.getMakeMacro('MPIEXEC')+'\n')
      g.write('MFEM_USE_METIS_5 = YES\n')
      g.write('MFEM_USE_METIS = YES\n')
      if self.scalar.precision == 'single':
        g.write('MFEM_PRECISION = single\n')
      g.write('MFEM_USE_PETSC = YES\n')
      g.write('HYPRE_OPT = '+self.headers.toString(self.hypre.include)+'\n')
      g.write('HYPRE_LIB = '+self.libraries.toString(self.hypre.lib)+'\n')
      g.write('METIS_OPT = '+self.headers.toString(self.metis.include)+'\n')
      g.write('METIS_LIB = '+self.libraries.toString(self.metis.lib)+'\n')
      if self.cuda.found:
        g.write('HYPRE_LIB := $(subst -Wl,-Xlinker=,$(HYPRE_LIB))\n')
        g.write('METIS_LIB := $(subst -Wl,-Xlinker=,$(METIS_LIB))\n')
      g.write('PETSC_VARS = '+prefix+'/lib/petsc/conf/petscvariables\n')
      g.write('PETSC_OPT = '+PETSC_OPT+'\n')
      # MFEM's config/defaults.mk overwrites these
      g.write('PETSC_DIR = '+PETSC_DIR+'\n')
      g.write('PETSC_ARCH = '+PETSC_ARCH+'\n')
      # Adding all externals should not be needed when PETSc is a shared library, but it is no harm.
      # When the HYPRE library is built statically, we need to resolve blas symbols
      # It would be nice to have access to the conf variables during postProcess, and access petsclib and other variables, instead of using a shell here
      # but I do not know how to do so
      petscext = '$(shell sed -n "s/PETSC_EXTERNAL_LIB_BASIC = *//p" $(PETSC_VARS)'+strip_rpath+')'

      if self.argDB['with-single-library']:
        petsclib = '-L'+prefix+'/lib -lpetsc'
      else:
        petsclib = '-L'+prefix+'/lib -lpetscml -lpetsctao -lpetscts -lpetscsnes -lpetscksp -lpetscdm -lpetscmat -lpetscvec -lpetscsys'
      if self.argDB['with-shared-libraries']:
        if self.cuda.found:
          petscrpt = '-Xlinker=-rpath,'+prefix+'/lib'
        else:
          petscrpt = '-Wl,-rpath,'+prefix+'/lib'
      else:
        petscrpt = ''
      g.write('PETSC_LIB = '+petscrpt+' '+petsclib+' '+petscext+'\n')
      if self.slepc.found:
        g.write('MFEM_USE_SLEPC = YES\n')
        g.write('SLEPC_OPT = '+PETSC_OPT+'\n')
        g.write('SLEPC_DIR = '+PETSC_DIR+'\n')
        g.write('SLEPC_ARCH = '+PETSC_ARCH+'\n')
        g.write('SLEPC_VARS = '+prefix+'/lib/slepc/conf/slepc_variables\n')
        slepclib = '-L'+prefix+'/lib -lslepc'
        slepcext = ''
        g.write('SLEPC_LIB = '+petscrpt+' '+slepclib+' '+slepcext+' $(PETSC_LIB)\n')
        if self.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
          makedepend = 'slepc-install'
        else:
          makedepend = 'slepc-build'
      self.writeConfig(g, 'ceed', self.ceed)
      self.writeConfig(g, 'superlu', self.superlu_dist)
      self.writeConfig(g, 'netcdf', self.netcdf)

      if self.cuda.found:
        self.pushLanguage('CUDA')
        petscNvcc = self.getCompiler()
        cudaFlags = self.updatePackageCUDAFlags(self.getCompilerFlags())
        self.popLanguage()
        cudaFlags = re.sub(r'-std=([^\s]+) ','',cudaFlags)
        g.write('MFEM_USE_CUDA = YES\n')
        g.write('CUDA_CXX = '+petscNvcc+'\n')
        g.write('CXXFLAGS := '+cudaFlags+' $(addprefix -Xcompiler ,$(CXXFLAGS))\n')
        g.write('CUDA_ARCH = sm_' + self.cuda.cudaArchSingle() + '\n')
      if self.hip.found:
        self.pushLanguage('HIP')
        hipcc = self.getCompiler()
        hipFlags = self.updatePackageCxxFlags(self.getCompilerFlags())
        self.popLanguage()
        hipFlags = re.sub(r'-std=([^\s]+) ','',hipFlags)
        g.write('MFEM_USE_HIP = YES\n')
        g.write('HIP_CXX = '+hipcc+'\n')
        hipFlags = hipFlags.replace('-fvisibility=hidden','')
        g.write('HIP_FLAGS = '+hipFlags+'\n')
        g.write('MPI_OPT = '+self.mpi.includepaths+'\n')
        g.write('MPI_LIB = '+self.mpi.libpaths+' '+self.mpi.mpilibs+'\n')
      g.close()

    with open(os.path.join(configDir,'petsc.mk'),'w') as f:
      f.write('''
MAKEOVERRIDES := $(filter-out CXXFLAGS=%,$(MAKEOVERRIDES))
unexport CXXFLAGS
unexport CPPFLAGS
.PHONY: run-config
run-config:
\t$(MAKE) -f {mfile} config MFEM_DIR={mfemdir}
'''.format(mfile=os.path.join(self.packageDir,'makefile'), mfemdir=self.packageDir))

    self.addDefine('HAVE_MFEM',1)
    self.addMakeMacro('MFEM','yes')
    self.addPost(buildDir, ['${OMAKE} -f ' + os.path.join(configDir,'petsc.mk') + ' run-config',
                            '${OMAKE} clean',
                            self.make.make_jnp,
                            '${OMAKE} install'])
    # this checks MFEM using the pre-installed libraries, I think that is wrong and it should use the post-installed prefix location
    self.addMakeCheck(os.path.join(buildDir, 'examples', 'petsc'), '${OMAKE} -i ex1p-test-par ex9p-test-par')
    self.logClearRemoveDirectory()
    self.logPrintBox('MFEM examples are available at ' + os.path.join(self.packageDir,'examples'))
    self.logResetRemoveDirectory()
    return self.installDir
