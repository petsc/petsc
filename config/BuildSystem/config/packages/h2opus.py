import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = 'cd09022' # Sep 30 2021, VMM handling
    self.download               = ['git://https://github.com/ecrc/h2opus']
    self.precisions             = ['single','double']
    self.skippackagewithoptions = 1
    self.cxx                    = 1
    self.requirescxx14          = 1
    self.liblist                = [['libh2opus.a']]
    self.includes               = ['h2opusconf.h']
    self.functionsCxx           = [1,'','h2opusCreateHandle']
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
    self.hip         = framework.require('config.packages.hip',self)
    self.magma       = framework.require('config.packages.magma',self)
    self.blas        = framework.require('config.packages.BlasLapack',self)
    self.kblas       = framework.require('config.packages.kblas',self)
    self.openmp      = framework.require('config.packages.openmp',self)
    self.mpi         = framework.require('config.packages.MPI',self)
    self.thrust      = framework.require('config.packages.thrust',self)
    self.math        = framework.require('config.packages.mathlib',self)
    self.deps        = [self.blas]
    self.odeps       = [self.mpi,self.openmp,self.cuda,self.kblas,self.magma,self.mpi,self.hip,self.thrust,self.math]
    return

  def Install(self):
    import os

    with_gpu = self.cuda.found and self.magma.found and self.kblas.found
    if not with_gpu and not (self.thrust.found or self.cuda.found or self.hip.found):
      raise RuntimeError('Missing THRUST. Run with --download-thrust or specify the location of the package')

    if self.openmp.found:
      self.usesopenmp = 'yes'

    self.pushLanguage('Cxx')
    cxx = self.getCompiler()
    cxxflags = self.getCompilerFlags()
    cxxflags = cxxflags.replace('-fvisibility=hidden','')
    cxxflags = cxxflags.replace('-std=gnu++14','-std=c++14')
    ldflags = self.setCompilers.LIBS + ' ' + self.setCompilers.LDFLAGS
    self.popLanguage()

    if self.blas.mangling == 'underscore':
      mangle = '-DH2OPUS_FMANGLE_ADD'
    elif self.blas.mangling == 'caps':
      mangle = '-DH2OPUS_FMANGLE_UPPER'
    else:
      mangle = '-DH2OPUS_FMANGLE_NOCHANGE'

    if with_gpu:
      self.pushLanguage('CUDA')
      nvcc = self.getCompiler()
      nvopts = self.getCompilerFlags()
      self.popLanguage()
      self.getExecutable(nvcc,getFullPath=1,resultName='systemNvcc',setMakeMacro=0)
      if hasattr(self,'systemNvcc'):
        nvccDir = os.path.dirname(self.systemNvcc)
        cudaDir = os.path.split(nvccDir)[0]
      else:
        raise RuntimeError('Unable to locate CUDA NVCC compiler')
      with_gpu=True

    with open(os.path.join(self.packageDir,'make.inc'),'w') as g:
      g.write('H2OPUS_INSTALL_DIR = '+self.installDir+'\n')
      g.write('CXX = '+cxx+'\n')
      g.write('CXXFLAGS = '+cxxflags+'\n')
      g.write('AR = '+self.setCompilers.AR+'\n')
      g.write('AR_FLAGS = '+self.setCompilers.AR_FLAGS+'\n')
      g.write('AR_SUFFIX = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
      g.write('RANLIB = '+self.setCompilers.RANLIB+'\n')
      g.write('SL = '+self.setCompilers.getLinker()+'\n')
      if self.argDB['with-shared-libraries']:
        g.write('SL_FLAGS = '+self.setCompilers.getSharedLinkerFlags()+'\n')
        g.write('SL_SUFFIX = '+self.setCompilers.sharedLibraryExt+'\n')
        g.write('SL_LINK_FLAG = '+self.setCompilers.CxxSharedLinkerFlag+'\n')
      if self.blas.mkl:
        g.write('H2OPUS_USE_MKL = 1\n')
      if hasattr(self.blas,'essl'):
        g.write('H2OPUS_USE_ESSL = 1\n')
      if self.libraries.check(self.blas.dlib, 'bli_init'):
        g.write('H2OPUS_USE_BLIS = 1\n')
      if self.libraries.check(self.blas.dlib, 'FLA_Init'):
        g.write('H2OPUS_USE_FLAME = 1\n')
      if config.setCompilers.Configure.isNEC(cxx, self.log):
        g.write('H2OPUS_USE_NEC = 1\n')
        g.write('H2OPUS_DISABLE_SHARED = 1\n')
      if config.setCompilers.Configure.isNVC(cxx, self.log):
        g.write('H2OPUS_USE_NVOMP = 1\n')
      cppfixes = ''
      if config.setCompilers.Configure.isIBM(cxx, self.log):
        cppfixes += ' -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT'

      g.write('CXXCPPFLAGS = ' + cppfixes + ' ' + mangle + ' -DH2OPUS_PROFILING_ENABLED '+self.headers.toString(self.blas.include)+'\n')
      g.write('BLAS_LIBS = '+self.libraries.toString(self.blas.dlib)+'\n')

      if self.setCompilers.isDarwin(self.log):
        self.getExecutable('dsymutil', getFullPath=1, resultName='dsymutil', setMakeMacro = 0)
        g.write('DSYMUTIL = '+self.dsymutil+'\n')

      if with_gpu:
        g.write('H2OPUS_USE_GPU = 1\n')
        if self.libraries.check(self.cuda.dlib, 'cuMemRelease'):
          g.write('H2OPUS_USE_GPU_VMM = 1\n')
        g.write('H2OPUS_USE_MAGMA_POTRF = 1\n')
        g.write('NVCC = '+nvcc+'\n')
        g.write('NVCCFLAGS = '+nvopts+' --expt-relaxed-constexpr\n')
        if self.cuda.cudaArch:
          g.write('GENCODE_FLAGS = -gencode arch=compute_'+self.cuda.cudaArch+',code=sm_'+self.cuda.cudaArch+'\n')
        g.write('CXXCPPFLAGS += '+self.headers.toString(self.cuda.include)+'\n')
        g.write('CXXCPPFLAGS += '+self.headers.toString(self.magma.include)+'\n')
        g.write('CXXCPPFLAGS += '+self.headers.toString(self.kblas.include)+'\n')
        g.write('CUDA_LIBS = '+self.libraries.toString(self.cuda.dlib)+'\n')
        g.write('MAGMA_LIBS = '+self.libraries.toString(self.magma.dlib)+'\n')
        g.write('KBLAS_LIBS = '+self.libraries.toString(self.kblas.dlib)+'\n')
      else:
        if self.thrust.found:
          g.write('CXXCPPFLAGS += '+self.headers.toString(self.thrust.include)+'\n')
        elif self.cuda.found:
          g.write('CXXCPPFLAGS += '+self.headers.toString(self.cuda.include)+'\n')
        elif self.hip.found:
          g.write('CXXCPPFLAGS += '+self.headers.toString(self.hip.include)+'\n')

      if self.scalartypes.precision == 'single':
        g.write('H2OPUS_USE_SINGLE_PRECISION = 1\n')

      if not self.mpi.usingMPIUni:
        g.write('H2OPUS_USE_MPI = 1\n')

      g.write('LDFLAGS = '+ldflags+' '+ self.libraries.toString(self.math.dlib)+'\n')

      if not self.argDB['with-shared-libraries']:
        g.write('H2OPUS_DISABLE_SHARED = 1\n')

    if self.installNeeded('make.inc'):
      try:
        output1,err1,ret1  = config.package.Package.executeShellCommand('make distclean', cwd=self.packageDir, timeout=60, log = self.log)
      except RuntimeError as e:
        self.logPrint('Error running make clean on H2OPUS: '+str(e))
        raise RuntimeError('Error running make clean on H2OPUS')
      try:
        self.logPrintBox('Compiling H2OPUS; this may take several minutes')
        output2,err2,ret2 = config.package.Package.executeShellCommand('make config && make', cwd=self.packageDir, timeout=2500, log = self.log)
        self.logPrintBox('Installing H2OPUS; this may take several minutes')
        output,err,ret = config.package.Package.executeShellCommand('make install', cwd=self.packageDir, timeout=60, log = self.log)
      except RuntimeError as e:
        self.logPrint('Error running make on H2OPUS: '+str(e))
        raise RuntimeError('Error running make on H2OPUS')
      self.postInstall(output1+err1+output2+err2,'make.inc')
    return self.installDir
