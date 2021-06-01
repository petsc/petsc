import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = 'Unknown' # opt_dist_sz March 29, 2020
    self.download               = ['git://https://github.com/wajihboukaram/hara']
    self.precisions             = ['single','double']
    self.skippackagewithoptions = 1
    self.cxx                    = 1
    self.requirescxx11          = 1
    self.liblist                = [['libhara.a']]
    self.includes               = ['hara.h']
    self.complex                = 0
    return

  # TODO
  #def setupHelp(self, help):
  #  config.package.Package.setupHelp(self, help)
  #  return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.scalartypes = framework.require('PETSc.options.scalarTypes',self)
    self.cuda        = framework.require('config.packages.cuda',self)
    self.magma       = framework.require('config.packages.magma',self)
    self.blas        = framework.require('config.packages.BlasLapack',self)
    self.kblas       = framework.require('config.packages.kblas',self)
    self.openmp      = framework.require('config.packages.openmp',self)
    self.mpi         = framework.require('config.packages.MPI',self)
    self.thrust      = framework.require('config.packages.thrust',self)
    self.deps        = [self.blas,self.openmp]
    self.odeps       = [self.cuda,self.kblas,self.magma,self.mpi,self.thrust]
    return

  def Install(self):
    import os

    if not self.blas.has_cheaders and not self.blas.mkl:
      raise RuntimeError('HARA requires cblas.h and lapacke.h headers')

    if self.openmp.found:
      self.usesopenmp = 'yes'

    self.pushLanguage('Cxx')
    cxx = self.getCompiler()
    cxxflags = self.getCompilerFlags()
    cxxflags = cxxflags.replace('-fvisibility=hidden','')
    self.popLanguage()

    with_gpu=False
    if self.cuda.found and self.magma.found and self.kblas.found:
      self.pushLanguage('CUDA')
      nvcc = self.getCompiler()
      nvopts = self.getCompilerFlags()
      self.popLanguage()
      self.getExecutable(nvcc,getFullPath=1,resultName='systemNvcc')
      if hasattr(self,'systemNvcc'):
        nvccDir = os.path.dirname(self.systemNvcc)
        cudaDir = os.path.split(nvccDir)[0]
      else:
        raise RuntimeError('Unable to locate CUDA NVCC compiler')
      with_gpu=True

    if not with_gpu and not (self.thrust.found or self.cuda.found):
      raise RuntimeError('Missing THRUST. Run with --download-thrust or specify the location of the package')

    if with_gpu:
      self.setCompilers.CUDAPPFLAGS += ' -std=c++11'

    with open(os.path.join(self.packageDir,'make.inc'),'w') as g:
      g.write('HARA_DIR = '+self.packageDir+'\n')
      g.write('HARA_INSTALL_DIR = '+self.installDir+'\n')
      g.write('prefix-root = '+self.packageDir+'\n')
      g.write('CC = '+cxx+'\n')
      g.write('OBJ_DIR = '+self.packageDir+'/obj\n')
      g.write('CCFLAGS = -DHLIB_PROFILING_ENABLED '+cxxflags+'\n')
      g.write('LIBHARA = '+self.packageDir+'/lib/libhara.a\n')
      g.write('INCLUDES = -I'+self.packageDir+'/include\n')

      if self.blas.mkl:
        g.write('CCFLAGS += -DMKL_INT=int\n')
        g.write('with_mkl = 1\n')

      g.write('INCLUDES += '+self.headers.toString(self.blas.include)+'\n')
      g.write('LIBRARIES = '+self.libraries.toString(self.blas.lib)+'\n')

      if with_gpu:
        g.write('with_gpu = 1\n')
        g.write('CUDA_PATH = '+cudaDir+'\n')
        g.write('MAGMA_DIR = '+self.magma.directory+'\n')
        g.write('KBLASROOT = '+self.kblas.directory+'\n')
        g.write('NVCC = '+nvcc+'\n')
        g.write('NVCCFLAGS := $(addprefix -Xcompiler ,$(CCFLAGS))\n')
        g.write('NVCCFLAGS += -DHLIB_PROFILING_ENABLED -std=c++11 --expt-relaxed-constexpr ' + nvopts+'\n')
        if self.cuda.gencodearch:
          g.write('GENCODE_FLAGS = -gencode arch=compute_'+self.cuda.gencodearch+',code=sm_'+self.cuda.gencodearch+'\n')
        g.write('INCLUDES += '+self.headers.toString(self.cuda.include)+'\n')
        g.write('INCLUDES += '+self.headers.toString(self.magma.include)+'\n')
        g.write('INCLUDES += '+self.headers.toString(self.kblas.include)+'\n')
        g.write('LIBRARIES += '+self.libraries.toString(self.magma.lib)+'\n')
        g.write('LIBRARIES += '+self.libraries.toString(self.kblas.lib)+'\n')
      else:
        if self.thrust.found:
          g.write('INCLUDES += '+self.headers.toString(self.thrust.include)+'\n')
        elif self.cuda.found:
          g.write('INCLUDES += '+self.headers.toString(self.cuda.include)+'\n')
        g.write('with_gpu = 0\n')

      if self.scalartypes.precision == 'single':
        g.write('double_prec = 0\n')
      else:
        g.write('double_prec = 1\n')

      if self.mpi.found and not self.mpi.usingMPIUni:
        g.write('with_dist = 1\n')
      else:
        g.write('with_dist = 0\n')

    # makefile include for examples
    includeDir = os.path.join(self.installDir, self.includedir)
    libDir = os.path.join(self.installDir, self.libdir)
    libhara = os.path.join(self.installDir, self.libdir, 'libhara.a')
    includeDirb = os.path.join(self.packageDir, 'include')
    libharab = os.path.join(self.packageDir, 'lib', 'libhara.a')
    if self.framework.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      petscdir = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
      petscdir = os.path.join(self.petscdir.dir,self.arch)

    with open(os.path.join(self.packageDir,'examples','make-example.inc'),'w') as g:
        g.write('-include '+petscdir+'/lib/petsc/conf/petscvariables\n')
        g.write('CCFLAGS = -DHLIB_PROFILING_ENABLED '+cxxflags+'\n')
        g.write('HARA_DIR = '+self.installDir+'\n')
        g.write('ifdef HARA_USE_BUILD\n')
        g.write('LIBHARA = '+libharab+'\n')
        g.write('INCLUDES = -I'+includeDirb+'\n')
        g.write('else\n')
        g.write('LIBHARA = '+libhara+'\n')
        g.write('INCLUDES = -I'+includeDir+'\n')
        g.write('endif\n')
        g.write('LDFLAGS := '+self.openmp.ompflag+' $(PETSC_EXTERNAL_LIB_BASIC)\n')
        if with_gpu:
          g.write('INCLUDES += '+self.headers.toString(self.cuda.include)+'\n')
          g.write('INCLUDES += '+self.headers.toString(self.magma.include)+'\n')
          g.write('INCLUDES += '+self.headers.toString(self.kblas.include)+'\n')
          g.write('LIBRARIES += '+self.libraries.toString(self.magma.lib)+'\n')
          g.write('LIBRARIES += '+self.libraries.toString(self.kblas.lib)+'\n')
          g.write('LIBRARIES += '+self.libraries.toString(self.cuda.lib)+'\n')
        else:
          if self.thrust.found:
            g.write('INCLUDES += '+self.headers.toString(self.thrust.include)+'\n')
          elif self.cuda.found:
            g.write('INCLUDES += '+self.headers.toString(self.cuda.include)+'\n')
        g.write('INCLUDES += '+self.headers.toString(self.blas.include)+'\n')
        g.write('LIBRARIES += '+self.libraries.toString(self.blas.dlib)+'\n')

    if self.installNeeded('make.inc'):
      try:
        output1,err1,ret1  = config.package.Package.executeShellCommand('mkdir -p lib && mkdir -p obj && make clean', cwd=self.packageDir, timeout=60, log = self.log)
      except RuntimeError as e:
        self.logPrint('Error running make clean on HARA: '+str(e))
        raise RuntimeError('Error running make clean on HARA')
      try:
        self.logPrintBox('Compiling HARA; this may take several minutes')
        output2,err2,ret2 = config.package.Package.executeShellCommand('make config && ' + self.make.make_jnp, cwd=self.packageDir, timeout=2500, log = self.log)
        self.logPrintBox('Installing HARA; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommandSeq(
          [self.installSudo+'mkdir -p '+libDir+' '+includeDir,
           self.installSudo+'cp -f lib/*.* '+libDir+'/.',
           self.installSudo+'cp -rf include/* '+includeDir+'/.'
          ], cwd=self.packageDir, timeout=60, log = self.log) #TODO namespace/folder include files?
      except RuntimeError as e:
        self.logPrint('Error running make on HARA: '+str(e))
        raise RuntimeError('Error running make on HARA')
      self.postInstall(output1+err1+output2+err2,'make.inc')
    return self.installDir
