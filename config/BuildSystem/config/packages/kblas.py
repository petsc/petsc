import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version                = '4.0.0'
    #self.gitcommit              = 'v'+self.version
    self.gitcommit              = '8af76dc862c74cbe880569ff2ccf6e5e54245430' # mar-27,2023 master
    self.download               = ['git://https://github.com/ecrc/kblas-gpu.git']
    self.buildLanguages         = ['CUDA'] # uses nvcc to compile everything
    self.functionsCxx           = [1,'struct KBlasHandle; typedef struct KBlasHandle *kblasHandle_t;extern "C" int kblasCreate(kblasHandle_t*);','kblasHandle_t h; kblasCreate(&h)']
    self.liblist                = [['libkblas.a']]
    self.includes               = ['kblas.h']
    return

  # TODO
  #def setupHelp(self, help):
  #  config.package.Package.setupHelp(self, help)
  #  return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cub    = framework.require('config.packages.cub',self)
    self.cuda   = framework.require('config.packages.cuda',self)
    self.magma  = framework.require('config.packages.magma',self)
    self.openmp = framework.require('config.packages.openmp',self)
    self.deps   = [self.cuda,self.magma]
    self.odeps  = [self.openmp,self.cub]
    return

  def Install(self):
    import os

    if not self.cub.found and not self.cuda.version_tuple[0] >= 11:
      raise RuntimeError('Package kblas requested but dependency cub not requested. Perhaps you want --download-cub')

    if self.openmp.found:
      self.usesopenmp = 'yes'

    self.pushLanguage('Cxx')
    cxx = self.getCompiler()
    cxxflags = self.getCompilerFlags()
    self.popLanguage()

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

    with open(os.path.join(self.packageDir,'make.inc'),'w') as g:
      g.write('_SUPPORT_BLAS2_ = TRUE\n')
      g.write('_SUPPORT_BLAS3_ = TRUE\n')
      g.write('_SUPPORT_BATCH_TR_ = TRUE\n')
      g.write('_SUPPORT_TLR_ = TRUE\n')
      g.write('_SUPPORT_SVD_ = TRUE\n')
      g.write('_SUPPORT_LAPACK_ = TRUE\n')
      if self.cub.found:
        g.write('_CUB_DIR_ = '+self.cub.directory+'/include\n')
      else:
        g.write('_CUB_DIR_ = '+cudaDir+'/include\n')
      g.write('_USE_MAGMA_ = TRUE\n')
      g.write('_MAGMA_ROOT_ = '+self.magma.directory+'\n')
      g.write('_CUDA_ROOT_ = '+cudaDir+'\n')
      if self.cuda.cudaArch:
        # TARGET_SM is just used to check min version compatibility, so we pass
        # it the smallest version (or 35 if "all" etc are specified)
        if self.cuda.cudaArchIsVersionList():
          gencodestr = '-DTARGET_SM='+str(min(int(v) for v in self.cuda.cudaArchList()))
        else:
          gencodestr = '-DTARGET_SM=35'
        gencodestr += self.cuda.nvccArchFlags()
      else:
        # kblas as of v4.0.0 uses __ldg intrinsics, available starting from 35
        gencodestr = '-DTARGET_SM=35 -arch sm_35'
      g.write('NVCC = '+nvcc+'\n')
      g.write('CC = '+cxx+'\n')
      g.write('CXX = '+cxx+'\n')
      g.write('LIB_KBLAS_NAME = kblas\n')
      g.write('COPTS = '+cxxflags+' -DUSE_MAGMA\n')
      if config.setCompilers.Configure.isIBM(cxx, self.log):
        g.write('NVOPTS = -Xcompiler -Wno-c++11-narrowing '+nvopts+' -DUSE_MAGMA\n')
      else:
        g.write('NVOPTS = '+nvopts+' -DUSE_MAGMA\n')
      g.write('NVOPTS_2 = '+gencodestr+'\n')
      if self.openmp.found:
        g.write('NVOPTS_3 = '+gencodestr+' -Xcompiler '+self.openmp.ompflag+'\n')
      else:
        g.write('NVOPTS_3 = '+gencodestr+'\n')
      g.close()

    if self.installNeeded('make.inc'):
      try:
        output1,err1,ret1  = config.package.Package.executeShellCommand('make clean', cwd=self.packageDir, timeout=60, log = self.log)
      except RuntimeError as e:
        pass
      try:
        self.logPrintBox('Compiling KBLAS; this may take several minutes')
        output2,err2,ret2 = config.package.Package.executeShellCommand('cd src && ' + self.make.make_jnp, cwd=self.packageDir, timeout=2500, log = self.log)
        libDir     = os.path.join(self.installDir, self.libdir)
        includeDir = os.path.join(self.installDir, self.includedir)
        self.logPrintBox('Installing KBLAS; this may take several minutes')
        output,err,ret = config.package.Package.executeShellCommandSeq(
          ['mkdir -p '+libDir+' '+includeDir,
           'cp -f lib/*.* '+libDir+'/.',
           'cp -f include/*.* '+includeDir+'/.'
          ], cwd=self.packageDir, timeout=60, log = self.log) #TODO namespace include files in kblas!
      except RuntimeError as e:
        self.logPrint('Error running make on KBLAS: '+str(e))
        raise RuntimeError('Error running make on KBLAS')
      self.postInstall(output1+err1+output2+err2,'make.inc')
    return self.installDir
